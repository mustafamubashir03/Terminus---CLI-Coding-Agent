import os
import re
from contextlib import contextmanager

from langchain.tools import tool

from terminus.permissions import Operation

_MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024

# grep is a whole-tree walk, so it needs its own bounds.  The skip list mirrors
# the junk that code_parser.get_source_files also skips, plus a blanket rule
# for hidden entries so .git/.venv/.terminus and credential files like .env are
# never swept into the model's context.
_GREP_SKIP_DIRS = {"node_modules", "__pycache__", "venv", "dist", "build", "target"}
_GREP_MAX_MATCHES = 100
_GREP_MAX_FILE_BYTES = 1024 * 1024

_TMP_SUFFIX = ".terminus-tmp"


def _remove_quietly(path: str) -> None:
    """Best-effort cleanup of a temp file left behind by a failed write."""
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


@contextmanager
def _write_guard(
    tool_name: str, file_path: str, operation: Operation = Operation.WRITE
):
    """Authorise a mutation, and hold the project writer lock while it happens.

    Yields None to proceed, or a model-visible refusal string. Every mutating
    filesystem tool goes through here, so a tool cannot be added later that
    forgets to ask permission *or* forgets to take the project write lock.

    Two separate things happen, in this order:

    1. ``authorize_operation`` decides whether this execution may mutate at all,
       using the same policy ``run_command`` uses, so a mutation cannot be
       authorised (or refused) by a different rule than a shell command. A
       refusal names the execution that hit it, so a failure during a task is
       attributable to that task rather than looking like a global rule.
    2. If it is allowed and the operation is not read-only, the project writer
       lock is held until the body finishes, so two concurrent tasks in one
       project cannot mutate at the same time.

    The lock is released in a ``finally``, so a tool that raises leaves no stale
    lock behind.

    Note step 1 is *authorisation*, not path containment.

    KNOWN ISSUE (deliberately deferred): these tools accept absolute paths,
    ``..`` traversal and paths outside the project, and follow symlinks. There
    is no workspace sandbox anywhere in Terminus, so "where" a file may be
    written is currently unconstrained - only "whether" is governed. Confinement
    is a semantic change (a coding agent legitimately writes to temp dirs and to
    absolute paths in tests) and is deferred rather than half-built here. Once
    this policy is in force the escape is at least *authorised* rather than
    silent: a context that denies WRITE can no longer write anywhere at all.
    """
    from terminus.execution import current_execution
    from terminus.coordination import project_write_guard

    running = current_execution()
    with project_write_guard(
        operation,
        target=file_path,
        context=running.label if running else None,
    ) as grant:
        yield grant


@tool
def read_file(file_path:str )->str:
    """ Read a file """
    if not file_path or not file_path.strip():
        return "No file path provided"
    file_path = file_path.strip()
    if not os.path.exists(file_path):
        return f"File not found: {file_path}"
    if os.path.getsize(file_path) > _MAX_FILE_SIZE_BYTES:
        return f"File is too large: {file_path}"
    # Read as UTF-8 to match write_file/edit_file/grep. Falling back to
    # errors="replace" keeps genuinely non-UTF-8 files readable instead of
    # failing outright, and the substitution is flagged so the model knows the
    # text is not byte-exact.
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except PermissionError:
            return f"Permission denied: {file_path}"
        except Exception as e:
            return f"Error reading file: {str(e)}"
        return (
            f"[note: {file_path} is not valid UTF-8; undecodable bytes are shown as "
            f"the replacement character]\n{content}"
        )
    except PermissionError:
        return f"Permission denied: {file_path}"
    except Exception as e:
        return f"Error reading file: {str(e)}"



@tool
def write_file(file_path:str, content:str)->str:
    """ Write content to  a file, creating it and any parent directories as needed """
    if not file_path or not file_path.strip():
        return "No file path provided"
    if not content:
        return "No content provided"
    file_path = file_path.strip()

    with _write_guard("write_file", file_path) as grant:
        if grant.refused or grant.deferred:
            return grant.refused or grant.deferred

        tmp_path = f"{file_path}{_TMP_SUFFIX}"
        try:
            if os.path.dirname(file_path):
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
            # Write to a sibling temp file and swap it in, so an interrupted write
            # cannot leave the target truncated. Same approach as edit_file.
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(content)
            os.replace(tmp_path, file_path)
            return f"File written successfully: {file_path}"
        except PermissionError:
            _remove_quietly(tmp_path)
            return f"Could not write {file_path}: permission denied by the operating system."
        except Exception as e:
            _remove_quietly(tmp_path)
            return f"Could not write {file_path}: {type(e).__name__}: {e}"



@tool
def append_file(file_path:str, content:str)->str:
    """ Append content to  a file, creating it and any parent directories as needed """
    if not file_path or not file_path.strip():
        return "No file path provided"
    if not content:
        return "No content provided"
    file_path = file_path.strip()
    with _write_guard("append_file", file_path) as grant:
        if grant.refused or grant.deferred:
            return grant.refused or grant.deferred
        try:
            if os.path.dirname(file_path):
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "a", encoding="utf-8") as f:
                f.write(content)
            return f"File appended successfully: {file_path}"
        except PermissionError:
            return f"Permission denied: {file_path}"
        except Exception as e:
            return f"Error appending to file: {str(e)}"


@tool
def edit_file(file_path: str, old_text: str, new_text: str) -> str:
    """
    Replace an exact snippet inside an existing file. 'old_text' must appear
    exactly once, copied verbatim from the file including indentation; it is
    replaced by 'new_text'. Use this for targeted changes instead of rewriting a
    whole file. An empty 'new_text' deletes the matched snippet. The file is
    never created here - use 'write_file' for that. Nothing is changed unless
    the match is unique.
    """
    if not file_path or not file_path.strip():
        return "No file path provided"
    file_path = file_path.strip()
    if not old_text:
        # str.count("") returns len(text) + 1, so an empty needle would appear to
        # "match" an empty file. Reject it rather than corrupt anything.
        return "No change made: old_text must not be empty."

    if os.path.isdir(file_path):
        return f"No change made: path is a directory, not a file: {file_path}"
    if not os.path.exists(file_path):
        return (
            f"No change made: file not found: {file_path}. "
            "Use 'write_file' to create a new file."
        )

    # Authorisation happens after validation but before any read-modify-write,
    # so a refused edit cannot touch the file at all. The project writer lock is
    # held for the whole read-modify-write, so a concurrent task cannot edit the
    # same file (or any file) while this one is mid-update.
    with _write_guard("edit_file", file_path) as grant:
        if grant.refused or grant.deferred:
            return grant.refused or grant.deferred
        return _apply_edit(file_path, old_text, new_text)


def _apply_edit(file_path: str, old_text: str, new_text: str) -> str:
    """Perform the validated edit. Called with the project writer lock held."""
    try:
        if os.path.getsize(file_path) > _MAX_FILE_SIZE_BYTES:
            return f"No change made: file is too large: {file_path}"
    except OSError as e:
        return f"Could not edit {file_path}: {e}"

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except UnicodeDecodeError:
        return f"No change made: cannot decode {file_path} as UTF-8 text."
    except PermissionError:
        return f"Could not read {file_path}: permission denied by the operating system."
    except Exception as e:
        return f"Could not read {file_path}: {type(e).__name__}: {e}"

    occurrences = content.count(old_text)
    if occurrences == 0:
        return (
            f"No change made: old_text not found in {file_path}. "
            "Read the file again and copy the snippet verbatim, including indentation."
        )
    if occurrences > 1:
        return (
            f"No change made: old_text found {occurrences} times in {file_path}. "
            "Add surrounding context to make it unique."
        )

    index = content.index(old_text)
    line_number = content.count("\n", 0, index) + 1
    updated = content[:index] + new_text + content[index + len(old_text):]

    # Write via a temp file so a failed write cannot leave a half-written file.
    tmp_path = f"{file_path}{_TMP_SUFFIX}"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(updated)
        os.replace(tmp_path, file_path)
    except Exception as e:
        _remove_quietly(tmp_path)
        if isinstance(e, PermissionError):
            return f"Could not write {file_path}: permission denied by the operating system."
        return f"Could not write {file_path}: {type(e).__name__}: {e}"

    return (
        f"Replaced 1 occurrence in {file_path} at line {line_number}; "
        f"file is now {len(updated)} characters"
    )


@tool
def delete_file(file_path:str)->str:
    """ Delete a file """
    if not file_path or not file_path.strip():
        return "No file path provided"
    file_path = file_path.strip()
    # DELETE is a DESTRUCTIVE operation, so under a worker policy (no approver)
    # this is refused; it only proceeds in a context that permits it.
    with _write_guard(
        "delete_file", file_path, operation=Operation.DELETE
    ) as grant:
        if grant.refused or grant.deferred:
            return grant.refused or grant.deferred
        try:
            os.remove(file_path)
            return f"File deleted successfully: {file_path}"
        except PermissionError:
            return f"Permission denied: {file_path}"
        except FileNotFoundError:
            return f"File not found: {file_path}"
        except Exception as e:
            return f"Error deleting file: {str(e)}"


@tool
def list_directory(directory:str)->str:
    """ List directory contents """
    if not directory or not directory.strip():
        return "No directory provided"
    directory = directory.strip()
    if not os.path.exists(directory):
        return f"Directory not found: {directory}"
    if not os.path.isdir(directory):
        return f"Path is not a directory: {directory}"
    try:
        return "\n".join(os.listdir(directory))
    except PermissionError:
        return f"Permission denied: {directory}"
    except Exception as e:
        return f"Error listing directory: {str(e)}"


@tool
def file_exists(file_path:str)->str:
    """ Check if a file exists """
    if not file_path or not file_path.strip():
        return "No file path provided"
    file_path = file_path.strip()
    try:
        if os.path.exists(file_path) and os.path.isfile(file_path):
            return f"File exists: {file_path}"
        else:
            return f"File does not exist: {file_path}"
    except Exception as e:
        return f"Error checking file: {str(e)}"


@tool
def grep(pattern: str, path: str = ".") -> str:
    """
    Search files for a regular expression and return the matching lines as
    'file:line: text'. Use this when you already know the exact text you are
    looking for (a function name, an import, a config key). Use
    'search_codebase' instead for conceptual questions where you do not know the
    wording. Hidden files and directories (names starting with '.') and common
    build directories are skipped; 'path' may be a single file or a directory.
    """
    if not pattern or not pattern.strip():
        return "No pattern provided"
    pattern = pattern.strip()
    path = path.strip() if path and path.strip() else "."

    try:
        regex = re.compile(pattern)
    except re.error as e:
        return f"Invalid regular expression: {str(e)}"

    matches: list[str] = []
    scanned = 0
    truncated = False

    def _scan_file(file_path: str) -> bool:
        """Append matches from one file. Returns False once the cap is reached."""
        nonlocal scanned, truncated
        try:
            if os.path.getsize(file_path) > _GREP_MAX_FILE_BYTES:
                return True
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                scanned += 1
                for lineno, line in enumerate(f, 1):
                    if regex.search(line):
                        matches.append(f"{file_path}:{lineno}: {line.rstrip()}")
                        if len(matches) >= _GREP_MAX_MATCHES:
                            truncated = True
                            return False
        except OSError:
            return True
        return True

    if os.path.isfile(path):
        _scan_file(path)
    elif os.path.isdir(path):
        for root, dirs, files in os.walk(path):
            dirs[:] = [
                d for d in dirs
                if not d.startswith(".") and d not in _GREP_SKIP_DIRS
            ]
            for name in sorted(files):
                if name.startswith("."):
                    continue
                if not _scan_file(os.path.join(root, name)):
                    break
            if truncated:
                break
    else:
        return f"Path not found: {path}"

    if not matches:
        return f"No matches for {pattern!r} in {path} ({scanned} files searched)"

    result = "\n".join(matches)
    if truncated:
        result += (
            f"\n... stopped after {_GREP_MAX_MATCHES} matches "
            "(narrow the pattern or pass a more specific path)"
        )
    return result