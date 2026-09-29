from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from dotenv import load_dotenv

from terminus.config import CONFIG
from terminus.tasks.errors import classify_failure


def _openrouter_diagnostic() -> dict[str, object]:
    from openai import OpenAI

    llm = CONFIG.get("llm", {})
    model = str(llm.get("model", ""))
    base_url = "https://openrouter.ai/api/v1"
    endpoint = f"{base_url}/chat/completions"
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        return {
            "provider": "openrouter",
            "model": model,
            "endpoint": endpoint,
            "request": "failed",
            "category": "authentication",
            "retryable": False,
            "detail": "OPENROUTER_API_KEY is not set",
        }
    client = OpenAI(
        api_key=key,
        base_url=base_url,
        timeout=float(llm.get("request_timeout_seconds", 120)),
        max_retries=0,
    )
    extra_body = {"include_reasoning": True}
    if llm.get("reasoning"):
        extra_body["reasoning"] = llm["reasoning"]
    started = time.perf_counter()
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply with OK."}],
            max_tokens=64,
            stream=True,
            stream_options={"include_usage": True},
            extra_body=extra_body,
        )
        chunks = list(stream)
        returned_model = next(
            (
                chunk.model
                for chunk in chunks
                if getattr(chunk, "model", None)
            ),
            model,
        )
        return {
            "provider": "openrouter",
            "model": model,
            "returned_model": returned_model,
            "endpoint": endpoint,
            "request": "succeeded",
            "http_status": 200,
            "category": "ok",
            "retryable": False,
            "elapsed_ms": round((time.perf_counter() - started) * 1000),
            "stream_chunks": len(chunks),
        }
    except Exception as exc:
        failure = classify_failure(exc, provider="openrouter", model=model)
        return {
            "provider": "openrouter",
            "model": model,
            "endpoint": endpoint,
            "request": "failed",
            "http_status": failure.status_code,
            "category": failure.category,
            "retryable": failure.retryable,
            "error_type": failure.error_type,
            "provider_code": failure.provider_code,
            "upstream_provider": failure.upstream_provider,
            "retry_after": failure.retry_after,
            "elapsed_ms": round((time.perf_counter() - started) * 1000),
        }
    finally:
        client.close()


def _gemini_diagnostic() -> dict[str, object]:
    import os

    from google import genai
    from google.genai import types

    llm = CONFIG.get("llm", {})
    fallback_list = llm.get("fallbacks") or ([llm["fallback"]] if llm.get("fallback") else [])
    fallback = fallback_list[0] if fallback_list else {}
    model = str(fallback.get("model") or llm.get("gemini_model") or "")
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        return {
            "provider": "google_genai",
            "model": model,
            "endpoint": endpoint,
            "request": "failed",
            "category": "authentication",
            "retryable": False,
            "detail": "GEMINI_API_KEY or GOOGLE_API_KEY is not set",
        }
    client = genai.Client(
        api_key=key,
        http_options=types.HttpOptions(
            retry_options=types.HttpRetryOptions(attempts=1)
        ),
    )
    started = time.perf_counter()
    try:
        response = client.models.generate_content(
            model=model,
            contents="Reply with OK.",
        )
        return {
            "provider": "google_genai",
            "model": model,
            "returned_model": getattr(response, "model_version", model),
            "endpoint": endpoint,
            "request": "succeeded",
            "http_status": 200,
            "category": "ok",
            "retryable": False,
            "elapsed_ms": round((time.perf_counter() - started) * 1000),
        }
    except Exception as exc:
        failure = classify_failure(exc, provider="google_genai", model=model)
        return {
            "provider": "google_genai",
            "model": model,
            "endpoint": endpoint,
            "request": "failed",
            "http_status": failure.status_code,
            "category": failure.category,
            "retryable": failure.retryable,
            "error_type": failure.error_type,
            "provider_code": failure.provider_code,
            "upstream_provider": failure.upstream_provider,
            "retry_after": failure.retry_after,
            "elapsed_ms": round((time.perf_counter() - started) * 1000),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("provider", choices=("openrouter", "google_genai"))
    args = parser.parse_args()
    load_dotenv(Path.cwd() / ".env")
    result = _openrouter_diagnostic() if args.provider == "openrouter" else _gemini_diagnostic()
    for key, value in result.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
