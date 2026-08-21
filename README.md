# Terminus

Terminus is a CLI tool that lets you ask natural-language questions about a codebase. It ingests a repository, builds a semantic vector index over the source code, and answers questions through an LLM agent equipped with a code-search tool.

## Status

- Semantic indexing and retrieval pipeline: **implemented**
- Agent orchestration (LangChain `create_agent` + tool-calling loop): **implemented**
- Multi-provider LLM support (Fireworks, OpenAI, Cerebras, Anthropic): **implemented**
- Hybrid retrieval (semantic + exact/lexical lookup): **planned**
- Build-artifact exclusion from indexing (`.egg-info`, `SOURCES.txt`, etc.): **planned**

## Features

- **Codebase ingestion** — walks a repository, splits source code into ast nodes and embeds them into a persistent Chroma vector store.
- **Semantic search tool** — a `search_codebase` tool backed by the vector index, exposed to the agent via `create_retriever_tool` / a custom retrieval wrapper.
- **Conversational CLI** — a REPL (`/ask`, `/clear`, `/help`, `/show_semantic_index`, `/exit`) for querying the indexed codebase.
- **Lazy agent construction with caching** — the agent is built once, on first query, and reused for the session (`handle_query` in `orchestrator.py`).
- **Configurable LLM backend** — provider and model are chosen via `config.yaml` and resolved at runtime by `llm/factory.py`, with support for:
  - Fireworks (`FIREWORKS_API_KEY`)
  - OpenAI (`OPENAI_API_KEY`)
  - Cerebras (`CEREBRAS_API_KEY`)
  - Anthropic (`ANTHROPIC_API_KEY`)
- **Call-budget middleware** — `ModelCallLimitMiddleware` and `ToolCallLimitMiddleware` cap how many model calls and `search_codebase` calls a single query can consume, so a query fails fast instead of looping indefinitely.

## Architecture

```
terminus/
├── cli.py                     # REPL entry point (/ask, /clear, /help, /show_semantic_index)
├── config.py                  # loads config.yaml into CONFIG
├── agent/
│   ├── factory.py             # builds the LangChain agent (LLM + tools + system prompt + middleware)
│   ├── orchestrator.py        # handle_query(): lazy agent build + cache + invoke
│   └── tools.py                # search_codebase tool: formats retrieved chunks for the agent
├── context/
│   ├── indexers/
│   │   └── semantic_chroma.py # builds/updates the Chroma index from source files
│   └── retrievers/
│       └── semantic_chroma.py # retrieve(): embeds a query and returns top-k Chroma matches
├── llm/
│   └── factory.py             # get_llm(): resolves provider/model from CONFIG
└── observability/
    └── logging.py              # structured logging
```

**Request flow:**

1. User types `/ask <question>` in the CLI.
2. `cli.py` strips the command prefix and calls `handle_query(question)`.
3. `orchestrator.py` builds the agent on first call (cached thereafter) and invokes it with the question, under a `recursion_limit`.
4. The agent (LangChain `create_agent`) decides whether to call `search_codebase`.
5. `search_codebase` embeds the query, retrieves the top-k nearest chunks from Chroma, and returns them formatted with file path, line range, type, and name.
6. The agent reads the results and produces a final answer, or calls the tool again — bounded by `ToolCallLimitMiddleware` and `ModelCallLimitMiddleware`.
7. The CLI prints the response.

## Configuration

LLM provider and model are set in `config.yaml`:

```yaml
llm:
  provider: fireworks   # fireworks | openai | cerebras | anthropic
  model: kimi-k3
```

Set the matching API key as an environment variable (e.g. `FIREWORKS_API_KEY`) via a `.env` file or your shell.

The embedder is fixed to `sentence-transformers/all-MiniLM-L6-v2` via HuggingFace.

## Usage

```bash
terminus
```

Commands inside the REPL:

| Command | Description |
|---|---|
| `/ask <question>` | Ask a question about the indexed codebase |
| `/show_semantic_index` | Show stats about the current semantic index |
| `/clear` | Clear the screen |
| `/help` | Show available commands |
| `/exit`, `/quit` | Exit the CLI |

## Evaluation Notes: Semantic Retrieval + Model Behavior

The semantic indexing and retrieval pipeline has been implemented and evaluated against multiple LLM backends before moving on to a hybrid retrieval architecture. The findings below summarize that evaluation.

### Retrieval quality

Retrieval performs well for **behavioral / descriptive queries** ("how does the agent get built?", "how is the LLM provider selected?", "how does the CLI handle the /ask command?") — these consistently return the correct source chunk in the top results, regardless of which LLM is used.

Retrieval performs poorly for **filename-style queries** ("orchestrator.py", "where is orchestrator.py?", "what does orchestrator.py do?"). In these cases, packaging/build metadata files (`SOURCES.txt`, `entry_points.txt`, `top_level.txt`, `PKG-INFO` under `.egg-info/`) tend to win the top-k slots over the actual source file, because they contain the literal filename as text while the source file itself does not. This is a known indexing gap, not a fundamental limitation of semantic search — excluding build artifacts from the index is expected to resolve it, and is planned as part of the hybrid retrieval work (adding an exact/lexical filename-match path alongside semantic search).

### Model comparison: gpt-oss-120b vs. kimi-k3

With an identical embedder, identical index, identical system prompt, and identical middleware limits, the two models diverged sharply in tool-use discipline:

> What was observed while testing is that models like gpt-oss-120b take more tool calls and model calls for a semantic search, whereas a model like kimi-k3 takes fewer model calls and tool calls for the same semantic search, using the same embedding model.

To test whether this was simply a matter of budget, the model-call and tool-call limits were deliberately increased to see whether gpt-oss-120b could perform closer to kimi-k3 given more room:

> The budget increase didn't help; it just delayed the same failure by exactly the amount added. With the old limit of 2 tool calls, gpt-oss-120b failed at 3. With the limit raised to 4, it failed at 5. That's not "it needed more room"; it's that the model will always use every call available, regardless of the quality of what it already has. A model that is actually judging sufficiency would stop earlier on at least some queries once it hit a good chunk. gpt-oss-120b never did; a 0% self-stop rate was observed across every multi-call case tested, on two separate call budgets.

By contrast:

> Compared directly against kimi-k3 on the identical prompt, identical limits, and identical index: kimi-k3 stopped at 2 of 4 available calls on the hardest query tested ("what does orchestrator.py do?") and returned a structured, honest answer distinguishing what it could confirm from what it could not. gpt-oss-120b, on an easier query, used its entire call budget and returned nothing but a limit-exceeded error.

An additional observation from earlier, unbounded runs:

> Without middleware limits in place, models like gpt-oss-120b were observed making a much larger number of tool calls and model calls for a single semantic search. With middleware limits applied, this behavior is instead forced to fail cleanly within a bounded range rather than looping indefinitely.

Some queries still fail on **both** models under semantic-only retrieval (primarily the filename-style queries described above). This is expected to be fixed by the planned hybrid retrieval architecture, but the semantic-only baseline needed to be properly evaluated on both models first, which this round of testing accomplished.

### Conclusion

> It has been concluded that increasing the model-call and tool-call limits still does not produce as good a result as simply using a better-behaved model. Increasing the budget only delays failure; it does not change whether the model is judging retrieved context as sufficient.

Both Fireworks and Cerebras are used as LLM providers in this project — not because either is required architecturally, but specifically to test multiple models against multiple embedding setups and identify which combinations perform best. The overall architecture and design choices aim to get reliable results even with the lowest-cost/lowest-capability models feasible; where that isn't achievable (as with gpt-oss-120b's tool-use discipline in this evaluation), a stronger model such as kimi-k3 is used instead.

## Roadmap

- [ ] Exclude `.egg-info/`, `SOURCES.txt`, `PKG-INFO`, `entry_points.txt`, `top_level.txt`, `dependency_links.txt`, `requires.txt` from indexing.
- [ ] Add an exact/lexical filename-match path as a second retrieval tool, to complement semantic search for "where is file X" style queries.
- [ ] Implement hybrid retrieval (semantic + lexical) and re-evaluate against both gpt-oss-120b and kimi-k3.
- [ ] Re-run the full query test suite after the indexing fix to confirm filename-style queries resolve correctly.