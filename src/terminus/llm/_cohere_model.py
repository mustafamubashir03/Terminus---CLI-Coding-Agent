"""The Cohere chat client, in its own module so its SDK import is optional.

Same reasoning as :mod:`terminus.llm._openrouter_model`: subclassing
``ChatCohere`` forces the ``langchain_cohere`` import to happen at module level,
so isolating the subclass is what lets :mod:`terminus.llm.factory` defer it to
the Cohere route instead of charging every deployment for it.
"""

from __future__ import annotations

from langchain_cohere import ChatCohere


class CohereChatModel(ChatCohere):
    """Cohere, with its tool-choice vocabulary mapped onto OpenAI's.

    Cohere rejects ``"any"``/``"auto"`` and wants ``"REQUIRED"``. Without this the
    agent's tool loop fails at the first call on a Cohere route.
    """

    def bind_tools(self, tools, **kwargs):
        if kwargs.get("tool_choice") in ("any", "auto"):
            kwargs["tool_choice"] = "REQUIRED"
        return super().bind_tools(tools, **kwargs)
