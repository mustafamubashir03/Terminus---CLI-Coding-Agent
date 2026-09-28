"""Runtime permission model for shell command execution.

Architectural rule enforced here: **the model never authorises itself.**

    model proposes a command
        -> classify_command() assigns a level from runtime tables
        -> PermissionPolicy.decide() applies the runtime policy
        -> an approver (human, in interactive use) may grant one command
        -> only then does the shell tool execute anything

The model cannot pass a permission, a level, an override, or a timeout. It only
supplies the command text and an optional working directory, both of which are
still validated by the runtime.

Classification is a best-effort allow/deny table, not a sandbox. It can be
bypassed (a python one-liner can do anything), and it is documented as such in
the tool docstring. The tables below are plain data so they can be extended
without touching the shell tool or this logic.
"""

from __future__ import annotations

import os
import re
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Iterable


class PermissionLevel(str, Enum):
    """How much a command is trusted to do."""

    READ_ONLY = "read_only"
    WRITE = "write"
    DESTRUCTIVE = "destructive"

    @property
    def rank(self) -> int:
        return {"read_only": 0, "write": 1, "destructive": 2}[self.value]


class Operation(str, Enum):
    """What kind of thing the model is asking to do.

    This is the vocabulary the permission policy reasons about. It is
    deliberately independent of shell syntax: "write this file" is expressible
    without pretending to be a command line, and "run this command" is just
    another operation whose *level* still comes from the shell classifier.
    """

    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    EXECUTE = "execute"

    @property
    def label(self) -> str:
        return {
            "read": "read",
            "write": "write",
            "delete": "delete",
            "execute": "execute",
        }[self.value]


# ---------------------------------------------------------------------------
# Classification tables (extend these; do not edit the logic below)
# ---------------------------------------------------------------------------

# Exact executable names that only observe state.
READ_ONLY_COMMANDS = frozenset({
    "pwd", "ls", "dir", "echo", "printf", "cat", "type", "head", "tail", "wc",
    "which", "where", "whoami", "hostname", "uname", "date", "env", "printenv",
    "stat", "file", "du", "df", "basename", "dirname", "realpath", "readlink",
    "grep", "egrep", "fgrep", "rg", "ag", "ack", "find", "fd", "tree",
    "diff", "cmp", "sort", "uniq", "cut", "tr", "nl", "jq", "yq", "less",
    "more", "column", "od", "xxd", "hexdump", "true", "false", "man", "help",
    "whereis", "getent", "ps", "wmic", "systeminfo", "ver", "uname",
})

# Commands that are only read-only in certain sub-argument shapes, matched as
# regexes against the whole segment. First match wins.
READ_ONLY_PATTERNS: tuple[tuple[re.Pattern, PermissionLevel], ...] = (
    # version probes
    (re.compile(r"^\s*(python[\d.]*|py)\s+(-V|--version)\s*$"), PermissionLevel.READ_ONLY),
    (re.compile(r"^\s*(python[\d.]*|py)\s+-c\s+.*--version\s*$"), PermissionLevel.READ_ONLY),
    (re.compile(r"^\s*node\s+(--version|-v)\s*$"), PermissionLevel.READ_ONLY),
    (re.compile(r"^\s*(npm|pnpm|yarn|bun)\s+(--version|-v)\s*$"), PermissionLevel.READ_ONLY),
    (re.compile(r"^\s*(pip[\d.]*|uv|poetry|conda)\s+(--version|-V)\s*$"), PermissionLevel.READ_ONLY),
    (re.compile(r"^\s*(cargo|rustc|go|java|javac|ruby|gem|dotnet|git)\s+(--version|-v)\s*$"),
     PermissionLevel.READ_ONLY),
    # read-only test introspection
    (re.compile(r"^\s*(pytest|py\.test)\b.*--collect-only"), PermissionLevel.READ_ONLY),
    (re.compile(r"^\s*(pytest|py\.test)\b.*--list-tests"), PermissionLevel.READ_ONLY),
    (re.compile(r"^\s*python[\d.]*\s+-m\s+(pytest|py\.test)\b.*--collect-only"),
     PermissionLevel.READ_ONLY),
    (re.compile(r"^\s*python[\d.]*\s+-m\s+(pytest|py\.test)\b.*--list-tests"),
     PermissionLevel.READ_ONLY),
    (re.compile(r"^\s*(npm|pnpm|yarn)\s+(ls|list|ls-remote|view|outdated|audit)\b"),
     PermissionLevel.READ_ONLY),
    (re.compile(r"^\s*uv\s+(pip\s+list|tree)\b"), PermissionLevel.READ_ONLY),
    (re.compile(r"^\s*(pip[\d.]*)\s+(list|show|freeze|config)\b"), PermissionLevel.READ_ONLY),
)

# git subcommands that only read.
GIT_READ_ONLY_SUBCOMMANDS = frozenset({
    "status", "diff", "log", "show", "blame", "reflog", "describe", "rev-parse",
    "ls-files", "ls-tree", "ls-remote", "cat-file", "shortlog", "whatchanged",
    "annotate", "grep", "remote", "config", "branch", "tag", "stash",
})
# git subcommands that destroy local or remote state.
GIT_DESTRUCTIVE_SUBCOMMANDS = frozenset({
    "clean", "reset", "rebase", "filter-branch", "filter-repo", "gc", "prune",
    "push", "reflog",
})
# git subcommands that modify the working tree or history.
GIT_WRITE_SUBCOMMANDS = frozenset({
    "add", "commit", "checkout", "switch", "restore", "merge", "cherry-pick",
    "revert", "am", "apply", "mv", "rm", "worktree", "submodule", "notes",
    "update-index", "stash",
})

# Substrings that mark a command as destructive regardless of the executable.
DESTRUCTIVE_PATTERNS: tuple[re.Pattern, ...] = (
    re.compile(r"(?<![\w./\\-])rm(\.exe)?\s+"),                        # any rm
    re.compile(r"\brmdir\b"), re.compile(r"\bshred\b"), re.compile(r"\bdel\b"),
    re.compile(r"\bformat\s+[a-zA-Z]:"), re.compile(r"\bmkfs\b"),
    re.compile(r"\bdd\s+if="), re.compile(r"\bfdisk\b"), re.compile(r"\bdiskpart\b"),
    re.compile(r"\bshutdown\b"), re.compile(r"\breboot\b"), re.compile(r"\bhalt\b"),
    re.compile(r"\binit\s+/dev/"), re.compile(r"\bkill(all)?\b"), re.compile(r"\btaskkill\b"),
    re.compile(r"\bStop-Process\b"), re.compile(r"\bRemove-Item\b.*-Recurse"),
    re.compile(r"\bDROP\s+(TABLE|DATABASE)\b", re.I),
    re.compile(r"\bTRUNCATE\s+TABLE\b", re.I),
    re.compile(r"\bgit\s+push\b.*(--force|-f)\b"),
    re.compile(r"\bgit\s+clean\b.*-[a-zA-Z]*[dfx]"),
    re.compile(r"\bgit\s+reset\b.*--hard\b"),
    re.compile(r":\(\)\s*\{.*\}\s*;\s*:"),                          # fork bomb
    re.compile(r"\bcurl\b.*\|\s*(ba)?sh\b"), re.compile(r"\bwget\b.*\|\s*(ba)?sh\b"),
    re.compile(r"\bchmod\s+(-R\s+)?777\b"), re.compile(r"\bchown\s+-R\b"),
    re.compile(r">\s*/dev/[sh]d[a-z]"), re.compile(r"\bnpm\s+publish\b"),
    re.compile(r"\btwine\s+upload\b"), re.compile(r"\bdocker\s+(rm|rmi|system\s+prune)\b"),
    re.compile(r"\bkubectl\s+delete\b"), re.compile(r"\bterraform\s+(apply|destroy)\b"),
)

# Substrings that mean "modifies project or system state" (checked only when the
# command is not already read-only).
WRITE_PATTERNS: tuple[re.Pattern, ...] = (
    re.compile(r"\b(mkdir|rmdir|touch|cp|mv|ln|tee|truncate|install)\b"),
    re.compile(r"\bsed\b.*-i"), re.compile(r"\bpatch\b"),
    re.compile(r"\b(pip[\d.]*|pip3)\s+install\b"), re.compile(r"\buv\s+(add|sync|pip\s+install)\b"),
    re.compile(r"\bpoetry\s+(add|install|lock|update)\b"), re.compile(r"\bconda\s+install\b"),
    re.compile(r"\b(npm|pnpm|yarn|bun)\s+(install|add|i|update|remove|uninstall|ci)\b"),
    re.compile(r"\b(pytest|py\.test|tox|nox|unittest)\b"),
    re.compile(r"\b(pyright|mypy|ruff|flake8|black|isort|pylint|eslint|prettier|tsc|biome)\b"),
    re.compile(r"\b(cargo|go|gradle|gradlew|maven|mvn|make|cmake|ninja)\s+(build|test|run|check|install|compile|vet)\b"),
    re.compile(r"\btsc\b"), re.compile(r"\bwebpack\b"), re.compile(r"\bvite\b"),
    re.compile(r"\bgit\b"), re.compile(r"\bpython[\d.]*\s+\S+\.py\b"),
    re.compile(r"\bnode\b.*\.(js|mjs|cjs|ts)\b"), re.compile(r"\bdocker\s+(build|run|compose)\b"),
    re.compile(r"\b(dotenv|export)\b"), re.compile(r"\breg\s+(add|delete|import)\b"),
)

# Command chaining: a chain is only as safe as its most dangerous segment.
_CHAIN_SPLIT = re.compile(r"(?:&&|\|\||;|\||&|\n)")

_ENV_ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")

# Windows executable suffixes, stripped so a table entry like "python" also
# matches "python.exe".
_EXE_SUFFIX = re.compile(r"\.(exe|cmd|bat|com)$", re.I)

# Prefix tokens that wrap the real executable rather than being it.
_WRAPPERS = frozenset({"sudo", "command", "env", "nohup", "time", "exec"})


def _split_segments(command: str) -> list[str]:
    """Split a command line into segments on shell chaining operators."""
    parts = [p.strip() for p in _CHAIN_SPLIT.split(command)]
    return [p for p in parts if p]


def _executable_index(tokens: list[str]) -> int | None:
    """Index of the token naming the real executable, or None."""
    for i, token in enumerate(tokens):
        if _ENV_ASSIGNMENT.match(token):
            continue
        if token.lower() in _WRAPPERS:
            # `env` alone is a read-only command; `env FOO=1 python ...` is not.
            if not any(
                not t.startswith("-") and not _ENV_ASSIGNMENT.match(t)
                for t in tokens[i + 1:]
            ):
                return i
            continue
        if token.startswith("-"):
            continue
        return i
    return None


def _base_executable(segment: str) -> str:
    """Executable name of a segment: lowercased, without path or file suffix."""
    tokens = segment.split()
    index = _executable_index(tokens)
    if index is None:
        return ""
    name = tokens[index].strip("\"'").replace("\\", "/").rsplit("/", 1)[-1]
    return _EXE_SUFFIX.sub("", name).lower()


def _normalized(segment: str, executable: str) -> str:
    """Rewrite a segment so its executable appears as a bare command name.

    ``"C:\\\\venv\\\\Scripts\\\\python.exe" -m pytest --collect-only`` becomes
    ``python -m pytest --collect-only``, so the anchored READ_ONLY_PATTERNS match
    regardless of quoting, path, or Windows file suffix.
    """
    if not executable:
        return segment
    tokens = segment.split()
    index = _executable_index(tokens)
    if index is not None:
        tokens[index] = executable
    return " ".join(tokens)

# Credential-shaped environment variables that must not reach a subprocess.
_SECRET_NAME = re.compile(
    r"(?i)(api[_-]?key|secret|token|password|passwd|credential|private[_-]?key|auth)"
)
# Redaction patterns for anything a command might print.
_SECRET_VALUE_PATTERNS: tuple[re.Pattern, ...] = (
    re.compile(r"(?i)authorization\s*[:=]\s*bearer\s+\S+"),
    re.compile(r"(?i)\b(?:api[_-]?key|access[_-]?token|secret|password)\b\s*[:=]\s*\S+"),
    re.compile(r"\bsk-[A-Za-z0-9_\-]{12,}\b"),
    re.compile(r"\bfc-[A-Za-z0-9_\-]{8,}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{16,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{12,}\b"),
)


def redact_secrets(text: str) -> str:
    """Mask credential-shaped substrings in text bound for the model."""
    out = str(text)
    for name, value in os.environ.items():
        if _SECRET_NAME.search(name) and len(value) >= 8 and value in out:
            out = out.replace(value, "[REDACTED]")
    for pattern in _SECRET_VALUE_PATTERNS:
        out = pattern.sub("[REDACTED]", out)
    return out


def sanitized_env() -> dict[str, str]:
    """A copy of the environment with credential-shaped variables removed."""
    return {k: v for k, v in os.environ.items() if not _SECRET_NAME.search(k)}


def _git_level(segment: str) -> PermissionLevel | None:
    """Classify a `git ...` segment, or None if it is not a git command."""
    tokens = segment.split()
    git_at = None
    for i, token in enumerate(tokens):
        if token.replace("\\", "/").rsplit("/", 1)[-1].lower() == "git":
            git_at = i
            break
    if git_at is None:
        return None
    for token in tokens[git_at + 1:]:
        if _ENV_ASSIGNMENT.match(token) or token.startswith("-"):
            continue
        sub = token.lower()
        if sub in GIT_DESTRUCTIVE_SUBCOMMANDS:
            if sub in {"reset", "clean", "push"} and not re.search(
                r"--hard|--force|-f\b|-[a-zA-Z]*[dfx]", segment
            ):
                return PermissionLevel.WRITE
            return PermissionLevel.DESTRUCTIVE
        if sub in GIT_WRITE_SUBCOMMANDS:
            return PermissionLevel.WRITE
        if sub in GIT_READ_ONLY_SUBCOMMANDS:
            # `git branch -d` / `git tag -d` / `git stash drop` mutate
            if re.search(r"\s-(d|D|delete|m|move)\b", segment):
                return PermissionLevel.WRITE
            return PermissionLevel.READ_ONLY
        return PermissionLevel.WRITE
    return PermissionLevel.READ_ONLY


def _classify_segment(segment: str) -> PermissionLevel:
    executable = _base_executable(segment)
    normalized = _normalized(segment, executable)

    # git first: it has the most precise subcommand tables, and the generic
    # patterns below would otherwise misread `git rm` as a bare `rm`.
    git = _git_level(segment)
    if git is not None:
        return git

    for pattern in DESTRUCTIVE_PATTERNS:
        if pattern.search(normalized):
            return PermissionLevel.DESTRUCTIVE

    for pattern, level in READ_ONLY_PATTERNS:
        if pattern.match(normalized):
            return level

    if executable in READ_ONLY_COMMANDS:
        return PermissionLevel.READ_ONLY

    for pattern in WRITE_PATTERNS:
        if pattern.search(normalized):
            return PermissionLevel.WRITE

    # Unknown executable: require approval rather than assuming it is safe.
    return PermissionLevel.WRITE


def classify_command(command: str) -> PermissionLevel:
    """Return the highest permission level required by *command*.

    A chained command inherits the level of its most dangerous segment, so
    ``ls && rm -rf /`` is DESTRUCTIVE even though it starts with ``ls``.

    DESTRUCTIVE_PATTERNS are also matched against the un-split command, because
    some of them only exist *because* of a chain operator
    (``curl ... | sh``). Splitting first would hide the very thing being
    matched.
    """
    raw = (command or "").strip()
    if not raw:
        return PermissionLevel.WRITE
    for pattern in DESTRUCTIVE_PATTERNS:
        if pattern.search(raw):
            return PermissionLevel.DESTRUCTIVE

    segments = _split_segments(raw)
    if not segments:
        return PermissionLevel.WRITE
    level = PermissionLevel.READ_ONLY
    for segment in segments:
        level = max(level, _classify_segment(segment), key=lambda lvl: lvl.rank)
        if level is PermissionLevel.DESTRUCTIVE:
            break
    return level


@dataclass(frozen=True)
class PermissionDecision:
    """Outcome of applying the runtime policy to a proposed operation."""

    allowed: bool
    level: PermissionLevel
    reason: str
    requires_approval: bool
    description: str = ""
    context: str | None = None

    def refusal_message(self) -> str:
        """A model-visible refusal that says what happened and what to do next.

        Distinguishes the three refusal causes the model must react to
        differently: the permission simply is not available here, a human
        declined, or approval is structurally impossible. Deliberately does not
        leak the policy's internal configuration.
        """
        what = f" {self.description}" if self.description else ""
        where = f" (during {self.context})" if self.context else ""
        if "rejected by user" in self.reason:
            return (
                f"Refused:{what} - the user declined to allow this "
                f"{self.level.value} operation{where}. Nothing was changed. "
                "Do not retry it; ask the user how to proceed."
            )
        if self.requires_approval or "requires" in self.reason:
            return (
                f"Refused:{what} - this {self.level.value} operation needs human "
                f"approval, and no approver is available{where}. "
                "Nothing was changed. Report this to the user instead of retrying."
            )
        return (
            f"Refused:{what} - {self.level.value} permission is not available{where}. "
            "Nothing was changed. "
            "Do not attempt to work around this; tell the user what you wanted to do."
        )


Approver = Callable[[str, str, PermissionLevel, str], bool]
"""(description, context, level, reason) -> bool

Called only by the runtime, only when a policy requires human approval, and
only for one concrete operation at a time. The model never supplies or
reaches this callable.
"""


class PermissionPolicy:
    """Runtime policy for every operation that can change the workspace.

    The decision logic is level-based and knows nothing about shell syntax:

    ``auto_approve``  levels that never need a human
    ``deny_levels``   levels that are refused outright
    ``approver``      consulted for anything in between

    When no approver is configured, anything not in ``auto_approve`` is
    **denied**, so a non-interactive context can never silently approve
    anything. The default policy is fail-closed: read-only only.
    """

    def __init__(
        self,
        auto_approve: Iterable[PermissionLevel] = (PermissionLevel.READ_ONLY,),
        approver: Approver | None = None,
        deny_levels: Iterable[PermissionLevel] = (PermissionLevel.DESTRUCTIVE,),
    ) -> None:
        self.auto_approve = frozenset(auto_approve)
        self.approver = approver
        self.deny_levels = frozenset(deny_levels)

    # -- core: level in, decision out. No knowledge of commands or tools. ----

    def decide(
        self,
        level: PermissionLevel,
        description: str = "",
        context: str | None = None,
    ) -> PermissionDecision:
        """Apply the policy to a level without consulting anyone."""
        if level in self.auto_approve:
            return PermissionDecision(
                True, level, f"{level.value} is pre-authorised by the active policy",
                False, description, context,
            )
        if level in self.deny_levels:
            if self.approver is None:
                return PermissionDecision(
                    False, level,
                    f"{level.value} operations are not permitted in this context",
                    False, description, context,
                )
            return PermissionDecision(
                False, level,
                f"{level.value} operation requires explicit human approval",
                True, description, context,
            )
        if self.approver is None:
            return PermissionDecision(
                False, level,
                f"{level.value} operation requires approval and no approver is configured",
                False, description, context,
            )
        return PermissionDecision(
            False, level, f"{level.value} operation requires approval",
            True, description, context,
        )

    def authorize(
        self,
        level: PermissionLevel,
        description: str = "",
        context: str | None = None,
    ) -> PermissionDecision:
        """Decide, consulting the approver only if the policy requires one."""
        decision = self.decide(level, description, context)
        if decision.allowed or not decision.requires_approval or self.approver is None:
            return decision
        approved = bool(
            self.approver(
                description or level.value,
                context or os.getcwd(),
                decision.level,
                decision.reason,
            )
        )
        return PermissionDecision(
            approved, decision.level,
            "approved by user" if approved else "rejected by user",
            False, description, context,
        )

    # -- shell convenience: classify a command, then use the core above. ---

    def decide_command(self, command: str, working_directory: str | None = None
                       ) -> PermissionDecision:
        return self.decide(classify_command(command), command, working_directory)

    def authorize_command(self, command: str, working_directory: str | None = None
                          ) -> PermissionDecision:
        return self.authorize(classify_command(command), command, working_directory)


# ---------------------------------------------------------------------------
# The one process-wide policy. Every mutating tool goes through this.
# ---------------------------------------------------------------------------

def classify_operation(operation: Operation, command: str | None = None) -> PermissionLevel:
    """Level required by an operation.

    EXECUTE defers to the shell classifier, so command-level knowledge stays in
    one place; the other levels are inherent to the operation itself.
    """
    if operation is Operation.READ:
        return PermissionLevel.READ_ONLY
    if operation is Operation.WRITE:
        return PermissionLevel.WRITE
    if operation is Operation.DELETE:
        return PermissionLevel.DESTRUCTIVE
    if operation is Operation.EXECUTE:
        return classify_command(command or "")
    return PermissionLevel.WRITE


def describe_operation(operation: Operation, target: str = "",
                       command: str | None = None) -> str:
    """A short human-meaningful label, used in prompts and refusals."""
    if operation is Operation.EXECUTE:
        return f"run_command: {command or ''}".rstrip()
    if target:
        return f"{operation.label}: {target}"
    return operation.label


_policy: ContextVar[PermissionPolicy] = ContextVar(
    "terminus_permission_policy", default=PermissionPolicy()
)
"""The policy in force for the *current execution*.

A ContextVar, not a module global, because an execution is not a process: the
CLI's /ask loop, a /plan task worker and (later) a spawned child agent each run
as their own asyncio Task, and asyncio copies the context per Task. A policy
installed inside one of them therefore cannot be observed or changed by a
sibling, and a worker cannot widen /ask's permissions or vice versa.

The default is fail-closed - read-only, no approver - so code that runs without
entering a scope denies rather than allows.
"""


def get_permission_policy() -> PermissionPolicy:
    """The policy in force for the current execution."""
    return _policy.get()


def set_permission_policy(policy: PermissionPolicy):
    """Install *policy* for the current execution.

    Returns the ContextVar token so a caller can restore the previous policy.
    Prefer :func:`permission_scope`, which does that automatically.
    """
    return _policy.set(policy)


@contextmanager
def permission_scope(policy: PermissionPolicy):
    """Run a block under *policy*, then restore the previous one.

    This is the only supported way to authorise an execution. Because the token
    is reset in a ``finally``, the policy cannot outlive the block even if the
    body raises.
    """
    token = _policy.set(policy)
    try:
        yield policy
    finally:
        _policy.reset(token)


def authorize_operation(
    operation: Operation,
    target: str = "",
    command: str | None = None,
    context: str | None = None,
) -> PermissionDecision:
    """The single authorization entry point for every mutating tool.

    ``write_file``, ``edit_file`` and ``run_command`` all come through here, so
    there is exactly one place where a mutation can be permitted or refused.
    The level is inherent to the operation; the decision belongs to the policy
    of the execution that is running, never to the caller.
    """
    level = classify_operation(operation, command)
    description = describe_operation(operation, target, command)
    return _policy.get().authorize(level, description, context)

