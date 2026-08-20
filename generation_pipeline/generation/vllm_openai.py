"""ChatOpenAI compatibility wrapper that preserves vLLM reasoning output."""

from typing import Any

from langchain_openai import ChatOpenAI


def _get_field(value: Any, field: str) -> Any:
    if isinstance(value, dict):
        return value.get(field)
    return getattr(value, field, None)


class VLLMChatOpenAI(ChatOpenAI):
    """Preserve vLLM's non-standard reasoning response field."""

    def _create_chat_result(self, response, generation_info=None):
        raw_choices = _get_field(response, "choices") or []

        result = super()._create_chat_result(
            response,
            generation_info=generation_info,
        )

        for generation, raw_choice in zip(
            result.generations,
            raw_choices,
        ):
            raw_message = _get_field(raw_choice, "message") or {}

            reasoning = _get_field(raw_message, "reasoning")
            if reasoning is None:
                reasoning = _get_field(
                    raw_message,
                    "reasoning_content",
                )

            if reasoning is not None:
                generation.message.additional_kwargs[
                    "reasoning_content"
                ] = reasoning

        return result