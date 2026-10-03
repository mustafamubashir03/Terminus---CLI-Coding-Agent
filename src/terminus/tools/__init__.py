"""How a Terminus tool is built.

One helper, shared by every tool that can refuse a call. It exists because
LangChain already decides what a *raised* tool error means and how the model
sees it - and because the default does not do it.

``BaseTool.handle_tool_error`` is ``False`` for a ``@tool``-decorated function.
With it off, a raised exception leaves ``BaseTool.run`` as an exception, and
LangGraph's ``ToolNode`` - whose default ``handle_tool_errors`` delegates to
``_default_handle_tool_errors``, which re-raises anything that is not an
argument-validation error - lets it out of the graph node entirely. A model that
tries one out-of-workspace path would therefore end the run rather than be told
why.

Turning the flag on uses the framework's own machinery, unmodified: the
exception becomes the tool's result with ``status="error"``, the loop continues,
and the model reads the reason. The alternative - returning the refusal as an
ordinary string - produces a *successful* ToolMessage, which is how a hard
boundary ends up looking like a result the model should try something else for.
See ``tools/filesystem_tools.py`` for which cases raise and which return.
"""

from __future__ import annotations

from typing import Any, Callable

from langchain.tools import tool as _langchain_tool

__all__ = ["refusing_tool"]


def _refusal_message(exc: Exception) -> str:
    """The text the model reads when a tool refused a call.

    ``str(exc)`` and nothing else: the framework's default template wraps the
    exception in ``repr()``, which would show the model ``ToolException('...')``
    - the type name is noise, and any hint about how refusals are spelled out in
    the system prompt would no longer match.
    """
    return str(exc)


def refusing_tool(func: Callable[..., Any] | None = None, *, name: str | None = None) -> Any:
    """Build a ``@tool`` from *func* whose refusals are observable, not fatal.

    Identical to ``@tool`` in every respect that a provider sees - same name,
    same inferred JSON schema, same description - plus one behavioural setting.

    Usable bare (``@refusing_tool``) or with an explicit provider-facing
    *name*, which ``@tool("some_name")`` also allows.
    """
    if func is None:
        def decorate(target: Callable[..., Any]) -> Any:
            return refusing_tool(target, name=name)

        return decorate
    built = _langchain_tool(name)(func) if name else _langchain_tool(func)
    built.handle_tool_error = _refusal_message
    return built
