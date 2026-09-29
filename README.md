# Terminus

A terminal coding agent. Terminus answers questions about a codebase, makes
controlled changes to project files, and can plan and execute multi-step work.

It runs against a local vector index, so it works with no accounts and no API
keys for anything except the language model.

## Install

```bash
pipx install terminus --python python3.12
```

Then run it from any project directory:

```bash
cd /path/to/your/project
terminus
```

Requires Python 3.12.

## Use

```bash
terminus                                    # interactive session
terminus agent -p "where is the retry logic?"   # one prompt, then exit
```

Both reach the same agent. The interactive form is better for a conversation;
`-p` is what you want in a script, a hook, or a CI step.

Inside the session, a bare message is treated as a question:

```
Query >> how does the auth callback work?
Query >> /plan add retry handling to the fetcher
Query >> /help
```

`/help` lists every command. It is generated from the same table the dispatcher
uses, so it cannot fall out of date.

## The agent

| Layer | What it is |
|---|---|
| **Session state** | LangGraph's `AsyncSqliteSaver`, under `.terminus/memory/` |
| **Semantic retrieval** | Qdrant or Chroma, over the project's own source |
| **Structure** | tree-sitter chunking; relationships are not yet extracted |

These are deliberately separate. A failed vector store degrades retrieval, not the
conversation: the agent falls back to `grep` and `read_file` rather than ending
the turn.

### Tools

`read_file`, `write_file`, `edit_file`, `list_directory`, `file_exists`, `grep`,
`search_codebase`, `web_search`, `web_fetch`, `run_command`, `load_skill`,
`project_status`, `spawn_agent`.

The model never authorises itself. A command is classified at runtime
(read-only / write / destructive) and the active policy decides; a destructive
command needs a human. In a pipe or a CI job there is nobody to ask, so anything
beyond read-only is refused.

### Delegation

`spawn_agent` hands one self-contained task to a bounded child with its own
context, its own role-restricted tools and a deadline. Children cannot nest
deeper than one level, and the total per turn is capped. A child that finishes
is a claim, not a conclusion - the parent still verifies.

Roles: `researcher`, `debugger`, `implementer`, `tester`, `reviewer`.

## Vector retrieval

Two backends, both first-class, both local by default.

```yaml
# config.yaml - Chroma, the default: no credentials, no server
vector_store:
  provider: chromadb
rag:
  mode: semantic
```

```yaml
# Qdrant, running locally in-process
vector_store:
  provider: qdrant
  retrieval_mode: hybrid      # hybrid adds BM25; Qdrant only
qdrant:
  mode: local
  path: .terminus/qdrant
  collection_name: terminus_hybrid
```

For a hosted cluster, set `qdrant.mode: cloud` and put `QDRANT_API_KEY` and
`CLUSTER_ENDPOINT` in `.env`. An existing cloud configuration keeps working: if
`qdrant.mode` is unset and `CLUSTER_ENDPOINT` is present, cloud is used.

The configured backend is the backend. If it cannot be reached, Terminus says
so and stops, rather than quietly switching to something else. There is an
opt-in compatibility switch, `vector_store.fallback_to_chroma`, and when it is
on the substitution is reported rather than hidden.

```bash
terminus index status      # configured backend, resolved location, reachability
terminus models list       # what is configured, and what each provider offers
```

## Configuration

`config.yaml` in the working directory, or the packaged defaults. Secrets live in
`.env`, never in configuration.

```bash
terminus config list
terminus config get llm.provider
terminus config set vector_store.provider chroma
```

Provider credentials go in the environment, or through the CLI:

```bash
terminus providers login --provider groq     # prompts, does not echo
terminus providers status
```

`terminus --log-level DEBUG` shows the routing decision and the resolved vector
backend.

## Where state lives

Everything is under `.terminus/` in the project, and all of it is derived and
rebuildable:

```
.terminus/
  memory/     conversation checkpoints and the current session id
  tasks/      planned projects and task state
  index/      what has been indexed, for incremental refresh
  chromadb/   local Chroma store
  qdrant/     local Qdrant store
```

`.terminus/` is gitignored. Delete it to reset; the next run rebuilds the index.

## Development

```bash
uv sync
uv run pytest
uv run ruff check .
```

Requires Python 3.12. `benchmarks/delegation_benchmark.py` runs the A/B that
established when delegation pays for itself.
