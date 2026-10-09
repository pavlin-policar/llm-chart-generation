"""Shared question validity indicators for the local and web viewers."""
from __future__ import annotations


def question_validity_labels(question: dict) -> tuple[str, str]:
    """Keep visual and data judgments separate, including missing judgments."""
    visual = question.get("vlisual_valid", question.get("visual_valid", question.get("valid")))
    data = question.get("data_valid")

    def label(name: str, value: object) -> str:
        if value is True:
            return f"🟢 {name}: valid"
        if value is False:
            return f"🔴 {name}: invalid"
        return f"⚪ {name}: not evaluated"

    return label("Visual", visual), label("Data", data)
