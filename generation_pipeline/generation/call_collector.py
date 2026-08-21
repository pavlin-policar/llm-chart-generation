import re

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import message_to_dict


def json_safe(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if hasattr(value, "model_dump"):
        return json_safe(value.model_dump(mode="json"))
    return str(value)


def sanitize_llm_input(value):
    if isinstance(value, str) and value.startswith("data:image/") and ";base64," in value:
        return "[image inserted]"
    if isinstance(value, dict):
        return {key: sanitize_llm_input(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_llm_input(item) for item in value]
    return value


def get_reasoning(message):
    reasoning = message.additional_kwargs.get("reasoning_content")
    if reasoning is not None:
        return json_safe(reasoning)

    if isinstance(message.content, str):
        match = re.search(r"<think>(.*?)</think>", message.content, re.DOTALL)
        if match:
            return match.group(1).strip()

    return None


class LLMCallCollector(BaseCallbackHandler):
    def __init__(self):
        self.calls = None
        self.pending = {}

    def start(self, initial_calls=None):
        self.calls = list(initial_calls or [])
        self.pending = {}

    def stop(self):
        calls = self.calls or []
        self.calls = None
        self.pending = {}
        return calls

    def on_chat_model_start(self, serialized, messages, *, run_id, metadata=None, **kwargs):
        if self.calls is None:
            return

        call = {
            "metadata": json_safe(metadata or {}),
            "input": [[sanitize_llm_input(json_safe(message_to_dict(message))) for message in batch] for batch in messages],
            "output": None,
        }
        self.calls.append(call)
        self.pending[str(run_id)] = call

    def on_llm_end(self, response, *, run_id, **kwargs):
        call = self.pending.pop(str(run_id), None)
        if call is None:
            return

        generation = response.generations[0][0]
        if hasattr(generation, "message"):
            message = generation.message
            call["output"] = json_safe(message_to_dict(message))["data"]
            call["output"]["reasoning"] = get_reasoning(message)
        else:
            call["output"] = {
                "content": json_safe(generation),
                "reasoning": None,
            }

    def on_llm_error(self, error, *, run_id, **kwargs):
        call = self.pending.pop(str(run_id), None)
        if call is not None:
            call["error"] = f"{type(error).__name__}: {error}"
