"""Summarize observed corrections of structured chart-feedback errors."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path


def _iteration_number(image: dict, fallback: int) -> int:
    stem = Path(str(image.get("path") or "")).stem
    if "_it" in stem:
        suffix = stem.rsplit("_it", 1)[1]
        if suffix.isdigit():
            return int(suffix)
    return fallback


def _ordered_images(record: dict) -> list[tuple[int, dict]]:
    """Use each saved iteration once, in image iteration order."""
    seen = set()
    images = []
    for position, image in enumerate(record.get("images") or []):
        if not isinstance(image, dict):
            continue
        path = image.get("path")
        if path and path in seen:
            continue
        if path:
            seen.add(path)
        images.append((_iteration_number(image, position), position, image))
    images.sort(key=lambda item: (item[0], item[1]))
    return [(number, image) for number, _, image in images]


def summarize_error_corrections(records: list[dict]) -> dict[str, dict]:
    """Count error-type episodes that clear in a later structured evaluation.

    An episode starts when a type first appears and continues while it is
    reported. A later evaluated image without that type closes the episode.
    Reappearance starts another episode. Missing structured feedback breaks
    observation and cannot establish a correction.
    """
    totals = defaultdict(lambda: {
        "episodes": 0,
        "corrected": 0,
        "revision_sum": 0,
        "by_revisions": defaultdict(int),
    })
    for record in records:
        active: dict[str, int] = {}
        for iteration, image in _ordered_images(record):
            errors = image.get("errors")
            if not isinstance(errors, list):
                active.clear()
                continue
            current = {
                error["type"].strip()
                for error in errors
                if isinstance(error, dict) and isinstance(error.get("type"), str)
                and error["type"].strip() and error["type"].strip() != "none"
            }
            for error_type in set(active) - current:
                revisions = iteration - active.pop(error_type)
                if revisions > 0:
                    totals[error_type]["corrected"] += 1
                    totals[error_type]["revision_sum"] += revisions
                    totals[error_type]["by_revisions"][revisions] += 1
            for error_type in current - active.keys():
                active[error_type] = iteration
                totals[error_type]["episodes"] += 1
    return {
        error_type: {
            **values,
            "by_revisions": dict(values["by_revisions"]),
        }
        for error_type, values in totals.items()
    }


def correction_rows(metrics: list[dict]) -> list[dict]:
    """Pool correction episodes across generation datasets for display."""
    pooled = defaultdict(lambda: {
        "episodes": 0,
        "corrected": 0,
        "revision_sum": 0,
        "by_revisions": defaultdict(int),
    })
    for metric in metrics:
        for error_type, values in (metric.get("feedback_corrections") or {}).items():
            target = pooled[error_type]
            for key in ("episodes", "corrected", "revision_sum"):
                target[key] += int(values.get(key, 0))
            for revisions, count in (values.get("by_revisions") or {}).items():
                target["by_revisions"][int(revisions)] += int(count)

    revision_counts = sorted({
        revisions
        for values in pooled.values()
        for revisions in values["by_revisions"]
    })
    rows = []
    for error_type, values in pooled.items():
        episodes = values["episodes"]
        corrected = values["corrected"]
        row = {
            "Error type": error_type,
            "Episodes": episodes,
            "Corrected": corrected,
            "Corrected share": f"{corrected / episodes:.1%}" if episodes else "-",
        }
        for revisions in revision_counts:
            row[f"{revisions} revision{'s' if revisions != 1 else ''}"] = (
                values["by_revisions"][revisions]
            )
        row["Mean revisions"] = (
            round(values["revision_sum"] / corrected, 2) if corrected else None
        )
        rows.append(row)
    return sorted(rows, key=lambda row: (-row["Corrected"], row["Error type"]))
