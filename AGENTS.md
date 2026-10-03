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

## Gotchas

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