"""Filesystem tools: the model-facing filesystem surface.

Every path a tool touches goes through :func:`terminus.workspace.resolve_in_workspace`
before any I/O happens. This module owns the *operations*; ``terminus.workspace``
owns *where they may point*. That split is deliberate - the rules about
containment have exactly one implementation, and no tool can forget to apply them
because there is no other way to turn a string into a path here.

Two result conventions, and the difference matters:

* **An outcome the model should react to is returned as text.** "File not found",
  "found 2 times", "Refused: ...", a non-zero exit code. These are results. The
  model has to read them and choose, and a successful ``ToolMessage`` carrying
  them is exactly right.
* **A workspace violation raises** :class:`langchain_core.tools.ToolException`.
  The tool is built by :func:`terminus.tools.refusing_tool`, so the framework
  turns that into a ``ToolMessage`` with ``status="error"``. It matters because
  reaching outside the workspace is not a result the model should adapt to - it
  is a rejected call, and the run's own record has to show that. Nothing about
  the boundary is expressible as "try a different absolute path instead".
"""

import os
import re

from langchain_core.tools import ToolException

from terminus.coordination import project_write_guard
from terminus.permissions import Operation
from terminus.tools import refusing_tool
from terminus.workspace import WorkspaceViolation, relative_to_workspace, resolve_in_workspace

_MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024

# grep is a whole-tree walk, so it needs its own bounds.  The skip list mirrors
# the junk that code_parser.get_source_files also skips, plus a blanket rule
# for hidden entries so .git/.venv/.terminus and credential files like .env are
# never swept into the model's context.
_GREP_SKIP_DIRS = {"node_modules", "__pycache__", "venv", "dist", "build", "target"}
_GREP_MAX_MATCHES = 100
_GREP_MAX_FILE_BYTES = 1024 * 1024

_TMP_SUFFIX = ".terminus-tmp"

_WORKSPACE_HINT = (
    "Filesystem paths are resolved inside this workspace only. Use a "
    "workspace-relative path."
)


def _blank(value) -> bool:
    """True for a missing or whitespace-only argument.

    A blank path is a malformed call the model can immediately correct, so it is
    reported as a result rather than raised - the convention for "the tool ran
    and here is what it has to say", as distinct from "this call was refused".
    """
    return not value or not str(value).strip()


def _resolve(path: str, argument: str = "file_path") -> str:
    """Resolve a model-supplied path inside the workspace, or refuse the call.

    Refusal is raised rather than returned: see the module docstring.
    """
    try:
        return str(resolve_in_workspace(path))
    except WorkspaceViolation as exc:
        raise ToolException(f"{exc} {_WORKSPACE_HINT}") from exc


def _shown(path: str) -> str:
    """A path as the model should recognise it: workspace-relative when inside."""
    return relative_to_workspace(path)


def _remove_quietly(path: str) -> None:
    """Best-effort cleanup of a temp file left behind by a failed write."""
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


@refusing_tool
def read_file(file_path:str )->str:
    """ Read a file from the workspace. The path is workspace-relative. """
    if _blank(file_path):
        return "No file path provided"
    file_path = _resolve(file_path)
    if not os.path.exists(file_path):
        return f"File not found: {_shown(file_path)}"
    if os.path.getsize(file_path) > _MAX_FILE_SIZE_BYTES:
        return f"File is too large: {_shown(file_path)}"
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
            return f"Permission denied: {_shown(file_path)}"
        except Exception as e:
            return f"Error reading file: {str(e)}"
        return (
            f"[note: {_shown(file_path)} is not valid UTF-8; undecodable bytes are "
            f"shown as the replacement character]\n{content}"
        )
    except PermissionError:
        return f"Permission denied: {_shown(file_path)}"
    except Exception as e:
        return f"Error reading file: {str(e)}"



@refusing_tool
def write_file(file_path:str, content:str)->str:
    """ Write content to a workspace file, creating it and any parent directories as needed """
    if _blank(file_path):
        return "No file path provided"
    if not content:
        return "No content provided"
    file_path = _resolve(file_path)
    shown = _shown(file_path)

    with project_write_guard(Operation.WRITE, target=shown) as grant:
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
            return f"File written successfully: {shown}"
        except PermissionError:
            _remove_quietly(tmp_path)
            return f"Could not write {shown}: permission denied by the operating system."
        except Exception as e:
            _remove_quietly(tmp_path)
            return f"Could not write {shown}: {type(e).__name__}: {e}"



@refusing_tool
def append_file(file_path:str, content:str)->str:
    """ Append content to a workspace file, creating it and any parent directories as needed """
    if _blank(file_path):
        return "No file path provided"
    if not content:
        return "No content provided"
    file_path = _resolve(file_path)
    shown = _shown(file_path)
    with project_write_guard(Operation.WRITE, target=shown) as grant:
        if grant.refused or grant.deferred:
            return grant.refused or grant.deferred
        try:
            if os.path.dirname(file_path):
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "a", encoding="utf-8") as f:
                f.write(content)
            return f"File appended successfully: {shown}"
        except PermissionError:
            return f"Permission denied: {shown}"
        except Exception as e:
            return f"Error appending to file: {str(e)}"


@refusing_tool
def edit_file(file_path: str, old_text: str, new_text: str) -> str:
    """
    Replace an exact snippet inside an existing workspace file. 'old_text' must
    appear exactly once, copied verbatim from the file including indentation; it
    is replaced by 'new_text'. Use this for targeted changes instead of rewriting
    a whole file. An empty 'new_text' deletes the matched snippet. The file is
    never created here - use 'write_file' for that. Nothing is changed unless
    the match is unique.
    """
    if _blank(file_path):
        return "No file path provided"
    if not old_text:
        # str.count("") returns len(text) + 1, so an empty needle would appear to
        # "match" an empty file. Reject it rather than corrupt anything.
        return "No change made: old_text must not be empty."

    file_path = _resolve(file_path)
    shown = _shown(file_path)

    if os.path.isdir(file_path):
        return f"No change made: path is a directory, not a file: {shown}"
    if not os.path.exists(file_path):
        return (
            f"No change made: file not found: {shown}. "
            "Use 'write_file' to create a new file."
        )

    # Authorisation happens after validation but before any read-modify-write,
    # so a refused edit cannot touch the file at all. The project writer lock is
    # held for the whole read-modify-write, so a concurrent task cannot edit the
    # same file (or any file) while this one is mid-update.
    with project_write_guard(Operation.WRITE, target=shown) as grant:
        if grant.refused or grant.deferred:
            return grant.refused or grant.deferred
        return _apply_edit(file_path, old_text, new_text, shown)


def _apply_edit(file_path: str, old_text: str, new_text: str, shown: str) -> str:
    """Perform the validated edit. Called with the project writer lock held."""
    try:
        if os.path.getsize(file_path) > _MAX_FILE_SIZE_BYTES:
            return f"No change made: file is too large: {shown}"
    except OSError as e:
        return f"Could not edit {shown}: {e}"

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except UnicodeDecodeError:
        return f"No change made: cannot decode {shown} as UTF-8 text."
    except PermissionError:
        return f"Could not read {shown}: permission denied by the operating system."
    except Exception as e:
        return f"Could not read {shown}: {type(e).__name__}: {e}"

    occurrences = content.count(old_text)
    if occurrences == 0:
        return (
            f"No change made: old_text not found in {shown}. "
            "Read the file again and copy the snippet verbatim, including indentation."
        )
    if occurrences > 1:
        return (
            f"No change made: old_text found {occurrences} times in {shown}. "
            "Add surrounding context to make it unique."
        )

    index = content.index(old_text)
    line_number = content.count("\n", 0, index) + 1
    updated = content[:index] + new_text + content[index + len(old_text):]

    tmp_path = f"{file_path}{_TMP_SUFFIX}"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(updated)
        os.replace(tmp_path, file_path)
    except Exception as e:
        _remove_quietly(tmp_path)
        if isinstance(e, PermissionError):
            return f"Could not write {shown}: permission denied by the operating system."
        return f"Could not write {shown}: {type(e).__name__}: {e}"

    return (
        f"Replaced 1 occurrence in {shown} at line {line_number}; "
        f"file is now {len(updated)} characters"
    )


@refusing_tool
def delete_file(file_path:str)->str:
    """ Delete a file in the workspace """
    if _blank(file_path):
        return "No file path provided"
    file_path = _resolve(file_path)
    shown = _shown(file_path)
    # DELETE is a DESTRUCTIVE operation, so under a worker policy (no approver)
    # this is refused; it only proceeds in a context that permits it.
    with project_write_guard(Operation.DELETE, target=shown) as grant:
        if grant.refused or grant.deferred:
            return grant.refused or grant.deferred
        try:
            os.remove(file_path)
            return f"File deleted successfully: {shown}"
        except PermissionError:
            return f"Permission denied: {shown}"
        except FileNotFoundError:
            return f"File not found: {shown}"
        except Exception as e:
            return f"Error deleting file: {str(e)}"


@refusing_tool
def list_directory(directory:str)->str:
    """ List workspace directory contents; the path is workspace-relative """
    if _blank(directory):
        return "No directory provided"
    directory = _resolve(directory, "directory")
    shown = _shown(directory)
    if not os.path.exists(directory):
        return f"Directory not found: {shown}"
    if not os.path.isdir(directory):
        return f"Path is not a directory: {shown}"
    try:
        return "\n".join(os.listdir(directory))
    except PermissionError:
        return f"Permission denied: {shown}"
    except Exception as e:
        return f"Error listing directory: {str(e)}"


@refusing_tool
def file_exists(file_path:str)->str:
    """ Check if a file exists in the workspace """
    if _blank(file_path):
        return "No file path provided"
    file_path = _resolve(file_path)
    shown = _shown(file_path)
    try:
        if os.path.exists(file_path) and os.path.isfile(file_path):
            return f"File exists: {shown}"
        return f"File does not exist: {shown}"
    except Exception as e:
        return f"Error checking file: {str(e)}"


@refusing_tool
def grep(pattern: str, path: str = ".") -> str:
    """
    Search workspace files for a regular expression and return the matching lines
    as 'file:line: text'. Use this when you already know the exact text you are
    looking for (a function name, an import, a config key). Use 'search_codebase'
    instead for conceptual questions where you do not know the wording. Hidden
    files and directories (names starting with '.') and common build directories
    are skipped; 'path' is workspace-relative and may be a single file or a
    directory.
    """
    if not pattern or not pattern.strip():
        return "No pattern provided"
    pattern = pattern.strip()
    path = _resolve(path or ".", "path")
    shown = _shown(path)

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
                        matches.append(f"{_shown(file_path)}:{lineno}: {line.rstrip()}")
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
        return f"Path not found: {shown}"

    if not matches:
        return f"No matches for {pattern!r} in {shown} ({scanned} files searched)"

    result = "\n".join(matches)
    if truncated:
        result += (
            f"\n... stopped after {_GREP_MAX_MATCHES} matches "
            "(narrow the pattern or pass a more specific path)"
        )
    return result
