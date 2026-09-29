"""Standalone Fireworks latency/timeout reproduction tool (NOT wired into the app).

Diagnostic aid for the production symptom: an LLM call hangs for ~30s to 10+
minutes, then fails with httpx.ReadTimeout/httpcore.ReadTimeout, surfaced only
as "Query failed:" in the CLI.

Run from the repo root (it reads FIREWORKS_API_KEY from ./.env):

    .venv\\Scripts\\python.exe scripts\\fireworks_repro.py

It performs three real requests against the Fireworks API and prints latency
and error type for each:

  1. small prompt, raw SDK, 30s timeout, no retries
  2. realistic app payload (system prompt + ~1k-token skill markdown + 26 MCP
     tool schemas), raw SDK, 30s timeout, no retries
  3. same realistic payload through langchain_fireworks.ChatFireworks using the
     app's configured per-attempt timeout (llm.request_timeout_seconds) and
     retries (llm.max_retries)

The point is to tell "provider-side latency" apart from "payload-size-driven".
Bounded timeout + retry in-app is the correct mitigation: a provider stall now
fails (or succeeds on retry) within a predictable window instead of blocking.
"""
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path.cwd() / ".env")

MODEL = os.environ.get("FW_REPRO_MODEL", "accounts/fireworks/models/kimi-k3")

_SKILL_MARKDOWN = """# Expert Debugger Skill

You are an expert Python debugger. Follow these steps strictly:

1. **Reproduce first.** Before touching anything, produce a minimal repro. A bug
   that cannot be reproduced cannot be fixed. Write the repro as a test.
2. **Read the full error.** Parse the traceback top to bottom. The cause is
   almost never on the final line. Identify the frame where the fault originates.
3. **Form a hypothesis with the stack.** Never guess. Each hypothesis must name
   the mechanism: attribute error, type coercion, ordering, mutation, timing,
   env drift, or race.
4. **Test the hypothesis in isolation.** A one-liner in a scratch file is worth
   50 print statements. If the one-liner contradicts the hypothesis, drop it.
5. **Fix at the root, not the symptom.** The fix must make the failure class
   impossible, not just this occurrence pass.
6. **Regression test.** Add the repro as a permanent test. A bug without a
   regression test will come back within two releases.
7. **Verify the fix in the real environment.** Run the real scenario end to
   end, not only the isolated test.

Troubleshooting matrix:
- `NoneType has no attribute` -> the object came from an api boundary; find the
  boundary and enforce a schema there.
- `KeyError` with a dict you "just set" -> you set a different dict (copy,
  shadowing, or a new object built from the same data).
- Works locally, fails in CI -> dependency drift, platform path separators, or
  a different working directory. Pin the environment, then bisect.
- Intermittent -> threading/the event loop. Look for non-thread-safe state,
  blocking calls on async paths, or unlocked shared mutables.
- Slow not wrong -> N+1 queries, accidental O(n^2), or synchronous I/O in a
  hot loop. Profile before optimizing.

Always end with a summary of root cause, the fix, and the regression test.
"""


def make_tool_schemas(n: int = 26) -> list[dict]:
    """Approximate the ~26 MCP tool schemas sent on implement/test type tasks."""
    return [
        {
            "type": "function",
            "function": {
                "name": f"mcp_tool_{i:02d}",
                "description": (
                    f"An MCP-provided tool {i}. Performs an external operation "
                    "related to the project workspace."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "target path"},
                        "options": {
                            "type": "object",
                            "description": "additional options for the operation",
                        },
                    },
                    "required": ["path"],
                },
            },
        }
        for i in range(n)
    ]


def make_large_payload() -> dict:
    system = (
        "You are tasked with executing a software engineering subtask.\n"
        "Task Type: implement\nDescription: Create the core module with tests.\n\n"
        "======SKILLS======\n\nAvailable skills:\nSkill: expert-python-debugger\n"
        "Description: Structured bug-hunting methodology\n\nWhen the user's "
        "request matches a skill, call load_skill(name) to load the skill.\n"
        f"\n{_SKILL_MARKDOWN}"
    )
    tools = make_tool_schemas(26)
    return {
        "system": system,
        "tools": tools,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Implement the module now. The project directory may be empty. "
                    "Create all output files from scratch."
                )
            }
        ],
        "payload_chars": len(system) + sum(len(str(t)) for t in tools),
    }


def run_raw_sdk(api_key: str, label: str, payload: dict, timeout: float) -> None:
    """Bypass LangChain: raw fireworks SDK with a short explicit timeout."""
    from fireworks.client import Fireworks

    fw = Fireworks(api_key=api_key, timeout=timeout)
    body = {
        "messages": payload.get("messages") or [{"role": "user", "content": "Say hi in one word."}]
    }
    if payload.get("system"):
        body["messages"] = [{"role": "system", "content": payload["system"]}, *body["messages"]]
    if payload.get("tools"):
        body["tools"] = payload["tools"]

    start = time.monotonic()
    try:
        resp = fw.chat.completions.create(model=MODEL, **body)
        dt = time.monotonic() - start
        content = (
            resp.choices[0].message.content
            if getattr(resp, "choices", None)
            else ""
        )
        print(f"[{label}] OK in {dt:.1f}s | {len(content or '')} chars | timeout={timeout}s", flush=True)
    except Exception as e:
        dt = time.monotonic() - start
        print(f"[{label}] FAIL in {dt:.1f}s | {type(e).__name__}: {e}", flush=True)


def run_chatfireworks(api_key: str, payload: dict, timeout: float, max_retries: int) -> None:
    """Through langchain_fireworks using the app's configured knobs."""
    from langchain_fireworks import ChatFireworks

    llm = ChatFireworks(
        model=MODEL,
        api_key=api_key,
        temperature=0,
        timeout=timeout,
        max_retries=max_retries,
    )
    body = [{"role": "system", "content": payload["system"]}, *payload["messages"]]
    start = time.monotonic()
    try:
        resp = llm.invoke(body)
        dt = time.monotonic() - start
        content = getattr(resp, "content", "") or ""
        print(
            f"[ChatFireworks] OK in {dt:.1f}s | {len(content)} chars | "
            f"timeout={timeout}s max_retries={max_retries}",
            flush=True,
        )
    except Exception as e:
        dt = time.monotonic() - start
        print(f"[ChatFireworks] FAIL in {dt:.1f}s | {type(e).__name__}: {e}", flush=True)


def main() -> None:
    api_key = os.environ.get("FIREWORKS_API_KEY")
    if not api_key:
        print("FIREWORKS_API_KEY not set")
        return 1

    print(f"Model: {MODEL}")
    run_raw_sdk(api_key, "raw-small", {"messages": [{"role": "user", "content": "Say hi in one word."}]}, 30.0)
    big = make_large_payload()
    print(f"approx payload: {big['payload_chars']} chars (sys + 26 tool schemas)")
    run_raw_sdk(api_key, "raw-large", big, 30.0)
    run_chatfireworks(api_key, big, 120.0, 2)
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
