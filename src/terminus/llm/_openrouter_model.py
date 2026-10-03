"""The OpenRouter chat client, in its own module so its SDK import is optional.

This class subclasses ``ChatOpenAI``, so the ``langchain_openai`` import cannot
live inside a function - a base class has to exist before the module that uses it
is defined. Keeping the subclass here instead of in :mod:`terminus.llm.factory`
means ``factory`` can import it *inside* the builder, so a Cohere or Google
deployment never pays the OpenAI SDK's import cost.

Importing ``langchain_openai`` costs seconds; that is worth paying only on the
route that actually uses it.
"""

from __future__ import annotations

from typing import Any

from langchain_openai import ChatOpenAI


class OpenRouterChatModel(ChatOpenAI):
    """OpenRouter, preserving ``reasoning_details`` across turns.

    OpenRouter returns a provider-specific ``reasoning_details`` block on a
    response and expects it echoed back on the next request for the same
    conversation. LangChain's ``ChatOpenAI`` drops it, so a multi-turn exchange
    loses the model's own reasoning between turns. Both directions are handled
    here: read on the way out, re-attached on the way in.
    """

    @staticmethod
    def _reasoning_details(response: Any, index: int) -> Any:
        if isinstance(response, dict):
            choices = response.get("choices") or []
        else:
            choices = getattr(response, "choices", None) or []
        if index >= len(choices):
            return None
        choice = choices[index]
        message = choice.get("message") if isinstance(choice, dict) else getattr(choice, "message", None)
        if isinstance(message, dict):
            return message.get("reasoning_details")
        return getattr(message, "reasoning_details", None)

    def _create_chat_result(self, response: Any, generation_info: dict | None = None):
        result = super()._create_chat_result(response, generation_info)
        for index, generation in enumerate(result.generations):
            details = self._reasoning_details(response, index)
            if details is None:
                continue
            metadata = dict(getattr(generation.message, "response_metadata", None) or {})
            metadata["reasoning_details"] = details
            generation.message.response_metadata = metadata
            additional = dict(getattr(generation.message, "additional_kwargs", None) or {})
            additional["reasoning_details"] = details
            generation.message.additional_kwargs = additional
        return result

    def _get_request_payload(self, input_: Any, *, stop: list[str] | None = None, **kwargs: Any) -> dict:
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        messages = self._convert_input(input_).to_messages()
        wire_messages = payload.get("messages") or []
        for index, message in enumerate(messages):
            if index >= len(wire_messages) or getattr(message, "type", "") != "ai":
                continue
            details = (getattr(message, "response_metadata", None) or {}).get(
                "reasoning_details"
            ) or (getattr(message, "additional_kwargs", None) or {}).get(
                "reasoning_details"
            )
            if details is not None:
                wire_messages[index]["reasoning_details"] = details
        return payload
