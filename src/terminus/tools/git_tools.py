"""Git as a versioning primitive: durable versions of the workspace state.

The filesystem tools give the agent durable *state*. These give it durable
*versions* of that state, so recovery and experimentation are mechanisms rather
than something the model has to improvise with backup files or remembered
instructions:

    Workspace
       |
       +-- filesystem state          (filesystem_tools)
       +-- Git history               (this module)
              |
              +-- checkpoints        git_commit
              +-- inspection         git_log, git_status, git_diff
              +-- rollback           git_checkout
              +-- isolation          git_branch

Three properties hold for everything in this module.

**One primitive.** Every call goes through :func:`_run_git`, which shells out to
the ``git`` executable with an argument *array*. There is no GitPython, no shell
interpreter, and no code path that builds a command string, so a commit message
or branch name containing ``;``, ``|``, ``$(...)`` or a leading dash is inert text
rather than syntax. Nothing here is a generic ``git(command=...)`` escape hatch
either: the model gets six named capabilities and no way to reach an arbitrary
subcommand.

**One workspace.** ``cwd`` is :func:`terminus.workspace.project_root`, computed
inside this module and never taken from an argument, so no Git call can be
pointed at another tree. The environment variables that would let Git ignore
that ``cwd`` (``GIT_DIR``, ``GIT_WORK_TREE``, ``GIT_INDEX_FILE`` and friends) are
stripped for the same reason. The repository is whatever contains the workspace;
a workspace that is not inside a Git work tree fails with one predictable
message rather than running ``git init`` behind the model's back.

**Intentional capabilities.** The surface stops at six tools. ``push``,
``pull``, ``fetch``, ``merge``, ``stash``, ``rebase`` and friends are
deliberately absent, and no tool force-recovers: when checkout cannot proceed
because local changes would be overwritten, that failure is returned. Discarding
the work with ``reset --hard`` or ``clean`` is a separate decision, and it is
not this module's to make.

Shared state, honestly
----------------------
Git state is repository-global. Every agent, /plan worker and later session
pointed at this workspace shares **one** working tree and **one** HEAD, so a
branch is *not* isolation between agents that share a checkout - checking one
out changes the files under all of them, and committing records whatever the
whole tree currently holds, including another agent's uncommitted work.

:func:`terminus.coordination.project_write_guard` serialises the three mutating
tools against other writers inside one Terminus process. It does not give an
agent its own branch, and it does not coordinate across processes. Real per-agent
isolation would need worktrees, which is a different primitive and not a
requirement yet; the honest statement today is the one above.

Two result conventions, following ``tools/filesystem_tools.py``:

* an outcome the model must react to - a clean tree, a refused checkout, a
  nothing-to-commit failure - is **returned as text**;
* an unreachable repository or a malformed argument is likewise returned, because
  both are states the model can read and correct rather than rejected calls.

Every tool pays one extra ``git rev-parse`` to establish that the workspace is a
work tree. That costs a process spawn per call and buys a single uniform failure
mode instead of five tools that report an empty clean tree because their output
was unparseable.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from terminus.coordination import project_write_guard
from terminus.observability.logging import get_logger
from terminus.permissions import Operation, redact_secrets, sanitized_env
from terminus.tools import refusing_tool
from terminus.workspace import project_root

logger = get_logger(__name__)

# Runtime-owned limits. Not model-controllable, and not configurable through the
# tool interface: a model cannot ask for a longer timeout the way it cannot ask
# for a longer path.
_GIT_TIMEOUT_SECONDS = 10
"""Seconds any single git invocation may take before it is killed.

Long enough for a status or a commit in an ordinary repository, short enough that
a wedged index lock, a credential prompt or a huge tree cannot stall a turn.
Every call here is a local, non-interactive command with no reason to take
longer, and the only thing that can make it slow is Git itself.
"""

_MAX_DIFF_CHARS = 20_000
_MAX_LOG_ENTRIES = 50
_DEFAULT_LOG_LIMIT = 10
_MAX_STATUS_ENTRIES = 200

#: Environment variables that would make Git ignore the ``cwd`` this module sets,
#: and therefore reach a repository, index or object store outside the workspace.
_GIT_ENV_STRIPPED = (
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_COMMON_DIR",
    "GIT_INDEX_FILE",
    "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_CEILING_DIRECTORIES",
    "GIT_NAMESPACE",
)

_SEP = "\x1f"
"""ASCII unit separator: the delimiter for machine-readable git output.

A field separator the model cannot type and a commit message is very unlikely to
contain, so parsing never has to guess where a subject ends. Used only where git
expands it - ``--pretty=format:`` does, ``--format=`` on ``git branch`` does not.
"""

_REF_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/~^+@-]{0,254}$")
"""What a branch name or commit-ish is allowed to look like.

Deliberately narrower than Git's own rules, because the job here is to reject
argument injection rather than to re-implement ``check-ref-format``: the first
character must be alphanumeric, so a value can never be read as an option (a
leading dash turns ``git branch -x`` into an unknown switch rather than a branch
called ``-x``), and nothing outside the ref alphabet survives - no whitespace,
quotes, backslashes, glob characters, semicolons or shell metacharacters.

``~`` and ``^`` are included so commit-ish values like ``HEAD~2`` stay
expressible. Git decides whether a given value is a branch or a commit, and this
module asks Git before acting on it.
"""


@dataclass(frozen=True)
class _GitResult:
    """One completed git invocation.

    A frozen record rather than a ``CompletedProcess`` because every interesting
    outcome has to survive being turned into model-facing text: a timeout, a
    missing executable, and a non-zero exit are three different reports.
    """

    args: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False
    unavailable: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out and not self.unavailable

    def failure(self, outcome: str) -> str:
        """Explain a failed call in terms of what the model was trying to do.

        Named by subcommand only. Quoting the full argument list would echo a
        commit message back inside an error - the one argument here that is model
        text rather than a fixed word.
        """
        verb = self.args[0] if self.args else "command"
        if self.timed_out:
            return (
                f"Could not {outcome}: git {verb} timed out after "
                f"{_GIT_TIMEOUT_SECONDS}s and was terminated. Nothing was changed."
            )
        if self.unavailable:
            return (
                f"Could not {outcome}: the git executable could not be run "
                f"({self.stderr.strip()}). Git tools need git on PATH. "
                "Nothing was changed."
            )
        detail = self.stderr.strip() or self.stdout.strip() or "git reported no reason"
        return (
            f"Could not {outcome}: git {verb} failed with exit code "
            f"{self.returncode}. Nothing was changed.\n{detail}"
        )


def _git_env() -> dict[str, str]:
    """The environment a git call runs with.

    Two deliberate differences from the process environment. Credentials are
    dropped, matching every other subprocess in the project, so a token cannot
    leak into a commit or into git's own output. And the ``GIT_*`` variables that
    would relocate the repository, index or object store are dropped, so ``cwd``
    stays the single thing that decides which repository this is.
    """
    return {
        key: value
        for key, value in sanitized_env().items()
        if key not in _GIT_ENV_STRIPPED
    }


def _run_git(*args: str, cwd: Path | None = None) -> _GitResult:
    """Run one git command in the workspace and capture both streams.

    The single place this module talks to Git. Arguments are passed as a list, so
    there is no quoting, no interpolation and nothing for a shell to interpret;
    ``shell=False`` is stated rather than relied on as a default so the guarantee
    is visible at the call. Exit codes are not raised - a non-zero exit is a
    result the caller reports, because "checkout would overwrite local changes"
    is information, not a crash.
    """
    argv = ["git", *args]
    workspace = Path(cwd) if cwd is not None else project_root()
    logger.info("git %s (cwd=%s)", " ".join(args), workspace)
    try:
        completed = subprocess.run(
            argv,
            cwd=str(workspace),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_GIT_TIMEOUT_SECONDS,
            check=False,
            shell=False,
            env=_git_env(),
        )
    except subprocess.TimeoutExpired as exc:
        return _GitResult(
            args, -1, _partial(exc.stdout), _partial(exc.stderr), timed_out=True
        )
    except (OSError, ValueError) as exc:
        return _GitResult(args, -1, "", f"{type(exc).__name__}: {exc}", unavailable=True)
    return _GitResult(
        args,
        completed.returncode,
        redact_secrets(completed.stdout or ""),
        redact_secrets(completed.stderr or ""),
    )


def _partial(stream) -> str:
    """Whatever a timed-out process managed to emit before it was killed."""
    if not stream:
        return ""
    if isinstance(stream, bytes):
        return stream.decode("utf-8", "replace")
    return str(stream)


def _repository_refusal(workspace: Path) -> str | None:
    """Why Git cannot be used in this workspace, or None when it can.

    One probe, called at the top of every tool, so all six fail the same way
    instead of some reporting an empty clean tree because their output was
    unparseable. Terminus does not run ``git init``: a workspace that is not a
    repository is a fact about the project, and turning it into one is the user's
    decision, not a side effect of asking what changed.
    """
    probe = _run_git("rev-parse", "--is-inside-work-tree", cwd=workspace)
    if probe.ok and probe.stdout.strip().lower() == "true":
        return None
    detail = probe.stderr.strip() or probe.stdout.strip() or "git reported no work tree"
    return (
        f"This workspace is not a Git repository ({detail}), so there is no "
        "version history to read or change. Nothing was changed. Git tools act "
        "on the workspace root only and do not create a repository; use the "
        "filesystem tools instead."
    )


def _clamp(text: str, limit: int) -> str:
    """Bound a report, saying plainly that it was bounded."""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n[truncated at {limit} characters; the change is larger]"


def _ref_problem(ref, argument: str, purpose: str) -> str | None:
    """Why *ref* cannot be used as a git argument, or None when it can."""
    raw = "" if ref is None else str(ref)
    if not raw.strip():
        return (
            f"No {argument} was given, so there is nothing to {purpose}. "
            "Nothing was changed."
        )
    if raw != raw.strip():
        return (
            f"The {argument} {raw!r} has leading or trailing whitespace, which a "
            "git argument cannot have. Nothing was changed."
        )
    if _REF_PATTERN.match(raw) is None:
        return (
            f"{raw!r} is not a usable git {argument}. A {argument} must start with "
            "a letter or digit and may contain only letters, digits and the "
            f"characters . _ / - ~ ^ @. Nothing was changed."
        )
    if ".." in raw or "@{" in raw:
        return (
            f"{raw!r} is not a usable git {argument}: '..' and '@{{' are not "
            "allowed. Nothing was changed."
        )
    if raw.endswith(("/", ".", ".lock")) or raw.count("@") > 1:
        return (
            f"{raw!r} is not a usable git {argument}: it ends in a way git forbids, "
            "or names a reflog entry rather than a ref. Nothing was changed."
        )
    return None


# git_status

def _parse_porcelain(text: str) -> tuple[str, list[str], list[str], list[str]]:
    """Split ``git status --porcelain`` into (branch header, staged, unstaged, untracked).

    The porcelain format is two status characters, a space, then the path - which
    is the whole reason it is used here instead of prose. ``X`` is the index,
    ``Y`` is the working tree, and ``??`` marks a path Git has never seen.
    """
    header = ""
    staged: list[str] = []
    unstaged: list[str] = []
    untracked: list[str] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        if line.startswith("## "):
            header = line[3:].strip()
            continue
        index_state, worktree_state = line[0], line[1]
        path = line[3:] if len(line) > 3 else ""
        if index_state == "?" and worktree_state == "?":
            untracked.append(path)
            continue
        if index_state not in (" ", "?"):
            staged.append(f"{index_state} {path}")
        if worktree_state not in (" ", "?"):
            unstaged.append(f"{worktree_state} {path}")
    return header, staged, unstaged, untracked


def _describe_branch(header: str) -> str:
    """Render the porcelain ``##`` header as a line about the branch."""
    if not header:
        return "branch: unknown"
    if "..." in header:
        branch, _, upstream = header.partition("...")
        return f"branch: {branch.strip()}  (upstream {upstream.strip().strip('[]')})"
    if header.startswith("HEAD (no branch)"):
        return "branch: none - HEAD is detached, not on a branch"
    if header.startswith("No commits yet on "):
        return f"branch: {header[len('No commits yet on '):].strip()} (no commits yet)"
    return f"branch: {header}"


@refusing_tool
def git_status() -> str:
    """
    Report the repository state: current branch, staged changes, unstaged
    changes, untracked files, and whether the workspace is clean.

    Read-only. Use it to find out where you are before committing, and to confirm
    what a change actually touched.

    Does not accept a path, a ref or a mode. It describes the whole workspace,
    which is what makes it a checkpoint check: git_status, then git_diff, then
    git_commit is the sequence for saving work.

    If the workspace is not a Git repository, this says so and changes nothing.
    """
    workspace = project_root()
    refusal = _repository_refusal(workspace)
    if refusal:
        return refusal

    result = _run_git("status", "--porcelain=v1", "--branch", cwd=workspace)
    if not result.ok:
        return result.failure("read repository status")

    header, staged, unstaged, untracked = _parse_porcelain(result.stdout)
    total = len(staged) + len(unstaged) + len(untracked)
    lines = [_describe_branch(header)]
    if total == 0:
        lines.append("state: clean - no staged, unstaged or untracked changes")
        return "\n".join(lines)

    def section(label: str, entries: list[str]) -> list[str]:
        if not entries:
            return []
        shown = entries[:_MAX_STATUS_ENTRIES]
        out = [f"{label} ({len(entries)}):"]
        out.extend(f"  {entry}" for entry in shown)
        if len(entries) > len(shown):
            out.append(f"  ... and {len(entries) - len(shown)} more")
        return out

    lines.extend(section("staged", staged))
    lines.extend(section("modified, not staged", unstaged))
    lines.extend(section("untracked", untracked))
    lines.append(
        f"state: dirty - {total} path(s) with changes; "
        "git_diff shows what they are."
    )
    return "\n".join(lines)


# git_diff

@refusing_tool
def git_diff(staged: bool = False) -> str:
    """
    Show the actual text of the workspace changes.

    'staged=False' (the default) shows changes in the working tree that are not
    staged. 'staged=True' shows what is staged and would go into the next commit.

    This is the only switch: there are no path filters, formats or diff modes.
    Untracked files never appear in a diff because Git has no baseline for them -
    git_status is what lists them.

    Output is bounded, and a truncation marker says so.
    """
    workspace = project_root()
    refusal = _repository_refusal(workspace)
    if refusal:
        return refusal

    args = ["diff", "--no-color", "--no-ext-diff"]
    if staged:
        args.append("--cached")
    result = _run_git(*args, cwd=workspace)
    if not result.ok:
        return result.failure("read the diff")

    if not result.stdout.strip():
        which = "staged changes" if staged else "unstaged changes"
        return f"No {which}. (Untracked files are not part of any diff; see git_status.)"

    what = "staged" if staged else "working-tree"
    return f"{what} changes:\n{_clamp(result.stdout.rstrip(), _MAX_DIFF_CHARS)}"


# git_commit

@refusing_tool
def git_commit(message: str) -> str:
    """
    Save the current workspace state as a recoverable checkpoint.

    'message' is required and becomes the commit message; write it for whoever
    reads this history later, including you.

    Staging behaviour, stated plainly because it is not what you might assume:
    this stages everything under the workspace first - new files, changes and
    deletions, honouring .gitignore - and then commits that. You cannot stage a
    subset through this tool; to commit only some paths, read git_diff, commit
    anyway, and be explicit in the message about what is in it.

    It never pushes. Nothing leaves this machine.

    Fails, changing nothing, when there is nothing to commit or when Git rejects
    the commit (an unset author identity is the usual reason); the reason is
    returned rather than swallowed.
    """
    if not message or not str(message).strip():
        return (
            "No commit message was given. A checkpoint with no message is "
            "unreadable later, so nothing was staged or committed. "
            "Pass a message describing the change."
        )
    text = str(message).strip()

    workspace = project_root()
    refusal = _repository_refusal(workspace)
    if refusal:
        return refusal

    with project_write_guard(Operation.WRITE, target="git commit") as grant:
        blocked = grant.refused or grant.deferred
        if blocked:
            return blocked

        # '--' followed by '.' scopes staging to the workspace directory, so a
        # workspace nested inside a larger repository cannot sweep in changes
        # from outside the tree this agent is allowed to touch. Without the
        # pathspec, 'git add -A' stages the whole repository.
        staged = _run_git("add", "-A", "--", ".", cwd=workspace)
        if not staged.ok:
            return staged.failure("stage changes for a commit")

        committed = _run_git("commit", "-m", text, cwd=workspace)
        if not committed.ok:
            return committed.failure("create a commit")

        head = _run_git("rev-parse", "HEAD", "--abbrev-ref", "HEAD", cwd=workspace)
        if not head.ok:
            return committed.failure("create a commit")

    summary = [line.strip() for line in committed.stdout.splitlines() if line.strip()]
    identifier, _, branch = head.stdout.partition("\n")
    lines = [
        f"Committed {identifier.strip()}",
        f"branch: {branch.strip()}",
        f"message: {text}",
    ]
    changed = next((line for line in summary if "changed" in line), "")
    if changed:
        lines.append(f"changed: {changed}")
    lines.append(
        "This is a local checkpoint only; nothing was pushed. "
        "git_log reads it back, git_checkout returns to it."
    )
    return "\n".join(lines)


# git_log

@refusing_tool
def git_log(limit: int = _DEFAULT_LOG_LIMIT) -> str:
    """
    List previous checkpoints: commit hash, author, date, message, and any
    branch or tag pointing at each commit.

    'limit' is how many commits to return, 1 to 50, default 10. It is always
    bounded - this never dumps a repository's whole history into your context.

    Read-only. Use it to find a known-good state before recovering to it. A
    repository with no commits yet reports that rather than an error.
    """
    if limit < 1 or limit > _MAX_LOG_ENTRIES:
        return (
            f"A log limit of {limit} is out of range. Pass between 1 and "
            f"{_MAX_LOG_ENTRIES}. Nothing was changed."
        )

    workspace = project_root()
    refusal = _repository_refusal(workspace)
    if refusal:
        return refusal

    result = _run_git(
        "log",
        f"--max-count={limit}",
        "--no-color",
        "--date=short",
        f"--pretty=format:%h{_SEP}%an{_SEP}%ad{_SEP}%d{_SEP}%s",
        cwd=workspace,
    )
    if not result.ok:
        if "does not have any commits yet" in result.stderr:
            return (
                "This repository has no commits yet, so there is no history to "
                "read. git_commit creates the first checkpoint."
            )
        return result.failure("read commit history")

    lines = [f"Last {min(limit, _MAX_LOG_ENTRIES)} commit(s), newest first:"]
    for entry in result.stdout.splitlines():
        if not entry.strip():
            continue
        parts = [part.strip() for part in entry.split(_SEP)]
        if len(parts) < 5:
            continue
        short, author, date, refs, subject = parts[:5]
        lines.append(f"{short}  {date}  {author}  {subject}")
        if refs:
            lines.append(f"        refs: {refs.lstrip(' ,')}")
    if len(lines) == 1:
        return "This repository has no commits yet."
    return "\n".join(lines)


# git_branch

@refusing_tool
def git_branch(name: str | None = None) -> str:
    """
    List branches, or create one.

    With no 'name': list every local branch, marking the current one, with its
    commit hash. Read-only.

    With a 'name': create that branch at the current commit. It does NOT switch to
    it - call git_checkout for that - and it does not push, delete or merge
    anything.

    Creating a branch only makes commits on it separate from the branch you
    started on. It does not isolate them from another agent working in the same
    workspace: they share one working tree and one HEAD.
    """
    workspace = project_root()
    refusal = _repository_refusal(workspace)
    if refusal:
        return refusal

    if name is None or not str(name).strip():
        return _list_branches(workspace)

    problem = _ref_problem(name, "branch name", "create")
    if problem:
        return problem
    branch = str(name).strip()

    with project_write_guard(Operation.WRITE, target=f"git branch {branch}") as grant:
        blocked = grant.refused or grant.deferred
        if blocked:
            return blocked
        created = _run_git("branch", "--", branch, cwd=workspace)
        if not created.ok:
            return created.failure(f"create branch {branch!r}")
        head = _run_git("rev-parse", "--abbrev-ref", "HEAD", cwd=workspace)

    current = head.stdout.strip() if head.ok else "unknown"
    return (
        f"Created branch {branch!r} at the current commit. "
        f"Still on {current!r}: creating a branch does not switch to it. "
        f'Call git_checkout("{branch}") to work on it. '
        "Nothing was merged, pushed or deleted."
    )


def _list_branches(workspace: Path) -> str:
    """Every local branch, with its hash, and which one is checked out."""
    # for-each-ref over refs/heads lists real branches only: 'git branch' also
    # emits a synthetic '(HEAD detached at abc1234)' line that is not a branch.
    # Tab is the separator because this git version does not expand %x1f in
    # --format, and none of these fields can contain a tab.
    result = _run_git(
        "for-each-ref",
        "--format=%(refname:short)%09%(objectname:short)%09%(HEAD)",
        "refs/heads/",
        cwd=workspace,
    )
    if not result.ok:
        return result.failure("list branches")

    branches: list[tuple[str, str]] = []
    current = ""
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        name, short_hash, marker = parts[0].strip(), parts[1].strip(), parts[2].strip()
        if marker == "*":
            current = name
        branches.append((name, short_hash))

    if not branches:
        detached = _run_git("rev-parse", "--short", "HEAD", cwd=workspace)
        at = f" at {detached.stdout.strip()}" if detached.ok else ""
        return (
            f"No branches yet (HEAD is detached{at}). git_commit creates the first one."
        )

    lines = [f"{len(branches)} local branch(es):"]
    lines.extend(
        f"{'* ' if name == current else '  '}{name}  {short_hash}"
        for name, short_hash in branches
    )
    lines.append(f"Currently on {current!r}." if current else "HEAD is detached: not on any branch.")
    return "\n".join(lines)


# git_checkout

def _resolve_ref(workspace: Path, ref: str) -> tuple[str, str]:
    """Classify *ref* as ('branch', name) or ('commit', sha), or ('unknown', reason).

    Asking Git before acting is not ceremony. ``git checkout <something>`` falls
    back to treating its argument as a *path* when it is not a ref, so passing an
    unverified string would let a branch switch silently become a file restore.
    Verification removes that ambiguity: a branch is switched by name, and a
    commit is only ever reached as a detached HEAD on a full hash.
    """
    branch = _run_git("show-ref", "--verify", "--quiet", f"refs/heads/{ref}", cwd=workspace)
    if branch.ok:
        return "branch", ref

    commit = _run_git(
        "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}", cwd=workspace
    )
    if commit.ok and commit.stdout.strip():
        return "commit", commit.stdout.strip()

    return "unknown", f"{ref!r} is neither a local branch nor a commit in this repository."


def _available_branches(workspace: Path) -> str:
    """Local branch names, for a refusal that has to tell the model what there is."""
    listing = _run_git(
        "for-each-ref", "--format=%(refname:short)", "refs/heads/", cwd=workspace
    )
    if not listing.ok:
        return "none listed"
    names = [line.strip() for line in listing.stdout.splitlines() if line.strip()]
    return ", ".join(names) or "none"


@refusing_tool
def git_checkout(ref: str) -> str:
    """
    Move the workspace to a branch, or to a commit.

    Exactly two outcomes, and which one happens is stated in the result:

    * 'ref' names a local branch -> you are now ON that branch, still attached.
    * 'ref' names a commit (a hash, 'HEAD', 'HEAD~2') -> HEAD becomes DETACHED at
      that commit. You are not on any branch, new commits there belong to no
      branch, and git_branch will not move you back. Recover by calling
      git_checkout with a branch name.

    It never checks out a file path, and it refuses rather than guessing when a
    value is neither a branch nor a commit.

    Local changes that checkout would overwrite make this fail, and the failure is
    returned with nothing changed. It will not discard your work: no hard reset,
    no clean, no stash. Commit, or revert the change yourself, then call it again.

    This rewrites files in the shared workspace, so anyone else working in it -
    another agent, a running task - sees the change too.
    """
    problem = _ref_problem(ref, "ref", "check out")
    if problem:
        return problem
    target = str(ref).strip()

    workspace = project_root()
    refusal = _repository_refusal(workspace)
    if refusal:
        return refusal

    with project_write_guard(Operation.WRITE, target=f"git checkout {target}") as grant:
        blocked = grant.refused or grant.deferred
        if blocked:
            return blocked

        kind, resolved = _resolve_ref(workspace, target)
        if kind == "unknown":
            return (
                f"{resolved} Nothing was changed. Local branches are: "
                f"{_available_branches(workspace)}. Use git_log to find a commit hash."
            )

        if kind == "branch":
            # No '--' here on purpose: '--' would make git read the argument as a
            # path to restore from the index. The name is verified to exist as
            # refs/heads/<name> and cannot start with '-', so git can only read it
            # as a branch.
            moved = _run_git("checkout", resolved, cwd=workspace)
            if not moved.ok:
                return moved.failure(f"switch to branch {resolved!r}")
            return "\n".join([
                f"Now on branch {resolved!r}. Workspace files now match it.",
                "Run git_status to see the resulting state.",
            ])

        # A commit is only ever reached detached and only ever by full hash, so
        # this argument cannot be read as a path.
        moved = _run_git("checkout", "--detach", resolved, cwd=workspace)
        if not moved.ok:
            return moved.failure(f"check out commit {resolved[:12]!r}")
        return "\n".join([
            f"HEAD is now DETACHED at {resolved[:12]} - you are not on a branch.",
            "Workspace files now match that commit.",
            "Commits made here belong to no branch. To get back to a branch, "
            "call git_checkout with its name.",
        ])
