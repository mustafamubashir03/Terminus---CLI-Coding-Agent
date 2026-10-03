# Agent Memory

This file is the agent's durable memory across sessions.
The harness loads it at the start of every session.
Update it whenever you learn something useful.

## Conventions

## Decisions

- The six Git tools in `tools/git_tools.py` are the only deliberate Git surface.
  `push`/`pull`/`fetch`/`merge`/`stash`/`rebase` are intentionally absent until a
  product requirement forces one. Do not add them to "fix" something.
- Only `/ask` gets the three Git *writers* (`git_commit`, `git_checkout`,
  `git_branch`). `/plan` workers and every child role get the three readers only,
  because neither surface has a permission approver. Keep it that way: a child
  agent sharing the parent's checkout must not be able to move its branch.
- The workspace has exactly one identity, `terminus.workspace.project_root()`.
  Filesystem tools and Git tools both resolve through it. There is no second
  workspace concept.
- **An enabled Sandbox never falls back to the host.** `cli.start_sandbox()`
  raises when the container will not start, and `terminus_cli_run` ends the
  session rather than continuing without isolation. The alternative - log a
  warning, leave the ContextVar unset, let the tools' host path take over -
  produces a session that *believes* it is sandboxed while every command runs on
  the host. Turning `sandbox.enabled` off is the supported way to run without
  Docker; that is a choice, not a failure mode.
- **One container per session, injected through a `ContextVar`.** The tools read
  `current_sandbox()` and never construct a `Sandbox`, because constructing one
  per command is a container per command. `tests/test_sandbox.py` asserts both
  halves, plus that a `/plan` worker in a child task inherits the container.

## Gotchas

- **A Docker bind mount silently binds the wrong tree when the source path is
  relative.** Docker resolves a volume source against the *daemon's* working
  directory, not the client's. The container then starts, `/workspace` exists,
  every command succeeds, and none of them touch the host workspace. So
  `Sandbox.__init__` calls `.resolve()`, and the startup check writes a token file
  on the host and reads that exact path back out of the container. Checking
  `test -d /workspace` is worthless here: the image creates `/workspace` at build
  time, so it passes against a container that has the host workspace nowhere in
  it. This actually happened and shipped no commands to the host.
- **Container paths must be built with `PurePosixPath`, not `Path`.**
  `Path("/workspace") / "sub"` is `\workspace\sub` on Windows, which is not a path
  inside the container.
- **`tempfile.mkdtemp` makes an owner-only ACL on Windows that Docker cannot write
  through.** The sandbox integration tests therefore use a directory inside the
  checkout (`.sandbox-integration/`, gitignored) rather than `tmp_path`. A
  sandbox test failing on a temp directory is usually this, not the sandbox.
- **The sandbox image tag is pinned and mirrored in two places.**
  `sandbox.SANDBOX_IMAGE` and `CONFIG["sandbox"]["image"]` must be equal, because
  `start_sandbox` reads the config key and a stale default silently overrides the
  pinned tag with `latest`. `tests/test_sandbox.py` asserts they agree.
- **The sandbox image carries a toolchain, not this repository's dependencies.**
  It ships pytest deliberately - an agent's first move is usually running tests,
  and making that cost a `pip install` per container spends a turn on it - but
  langgraph and the rest are the project's business and are installed per session
  in the disposable container. So "the project's own suite runs in the bare image"
  is false; the integration test runs a project's tests instead.

- **Comments in this repo are documentation, not noise.** They record *why* a
  non-obvious choice was made, plus scope limits and past defects. Specific ones
  that must not be deleted: `permissions.py` "not a sandbox"; `coordination.py`
  "process-local ... two processes are NOT covered"; `git_tools.py` the note that
  `--` is deliberately omitted from `git checkout` (with it, git reads the
  argument as a *path* and restores a file instead of switching branches);
  `config.py` "first file that exists wins ... not a merge". If you think a
  comment is redundant, check whether deleting it makes a *claim* appear that the
  code does not support.
- **`config.yaml` in the repo root leaks into the test suite.** `config.py`
  reads `Path.cwd() / "config.yaml"`, so several `tests/test_user_config_ux.py`
  cases pass or fail depending on whichever provider the developer has configured.
  Changing the provider there breaks ~23 tests. Use `~/.terminus/config.yaml` or
  `terminus config set llm.provider <name> --scope global` instead.
- **Config precedence is cwd > global > package.** A project `config.yaml` always
  beats `~/.terminus/config.yaml`, so a global setting appears to be ignored when
  you run inside a repo that ships one.
- **`git add -A` is repository-wide**, not cwd-limited (verified on git 2.51). Use
  `git add -A -- .` to scope staging to the workspace directory.
- **`git checkout <x>` silently treats a non-ref as a path.** Always resolve the
  ref first (`show-ref --verify refs/heads/<name>`, then
  `rev-parse --verify <ref>^{commit}`) before moving HEAD.
- **Ollama works without a new dependency** because it serves an
  OpenAI-compatible API at `/v1`. It is declared in `llm/providers.py` with
  `key_required=False` and `base_url_env="OLLAMA_BASE_URL"`.
- **Other agents are editing this repo concurrently.** During one session both
  newly created files were silently deleted by an external process, and
  `agent/orchestrator.py` was rewritten wholesale between two reads. Re-read a
  file immediately before editing it, and re-run the suite after every batch
  rather than assuming an earlier green run still holds.
- **Attribute failures before fixing them.** `tests/test_observation_enforcement.py`
  (and its module `agent/observation.py`) were untracked new work that failed
  independently of anything else in the repo.

## Active Tasks