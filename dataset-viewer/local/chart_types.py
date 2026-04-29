"""Chart-type normalization helpers for evaluation and Stan scaffolding."""

from __future__ import annotations

import re


QUESTION_TYPE_ORDER = [
    "metadata",
    "value extraction",
    "reasoning",
    "comparison",
    "trends",
]


def _normalize(text: str) -> str:
    text = text.strip().lower()
    text = text.replace("_", " ")
    text = text.replace("-", " ")
    text = re.sub(r"[^a-z0-9\s/]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def canonicalize_chart_type(raw_chart_type: str) -> str:
    """Collapse noisy chart labels into a manageable family taxonomy."""
    chart_type = _normalize(raw_chart_type)

    if not chart_type:
        return "Unknown"

    if "confusion" in chart_type:
        return "Confusion Matrix"
    if "heatmap" in chart_type or "heat map" in chart_type or "correlation" in chart_type:
        return "Heatmap"
    if "contingency heatmap" in chart_type or "categorical heatmap" in chart_type:
        return "Heatmap"
    if "error bar" in chart_type or "errorbar" in chart_type:
        return "Error Bar"
    if "facet" in chart_type or "subplot" in chart_type or "multi panel" in chart_type:
        return "Faceted Plot"
    if "hexbin" in chart_type:
        return "Hexbin"
    if "contour" in chart_type or "quiver" in chart_type:
        return "Contour / Field"
    if "3d surface" in chart_type:
        return "3D Surface"
    if "3d scatter" in chart_type or "scatter 3d" in chart_type:
        return "3D Scatter"
    if "scatter matrix" in chart_type or "pair plot" in chart_type or "pairplot" in chart_type:
        return "Scatter Matrix"
    if "pair grid" in chart_type or "pairwise" in chart_type:
        return "Scatter Matrix"
    if "parallel coordinate" in chart_type:
        return "Parallel Coordinates"
    if "pca" in chart_type or "projection" in chart_type or "biplot" in chart_type:
        return "Projection Plot"
    if "radar" in chart_type:
        return "Radar Chart"
    if "bubble" in chart_type:
        return "Bubble Chart"
    if "pie" in chart_type or "donut" in chart_type:
        return "Pie / Donut"
    if "treemap" in chart_type or "sunburst" in chart_type or "mosaic" in chart_type or "upset" in chart_type:
        return "Set / Composition"
    if "histogram" in chart_type or chart_type == "hist" or "2d histogram" in chart_type:
        return "Histogram"
    if "ecdf" in chart_type or "cumulative distribution" in chart_type or "cdf" in chart_type:
        return "Distribution Summary"
    if "qq plot" in chart_type or "q q plot" in chart_type:
        return "Distribution Summary"
    if "density" in chart_type or chart_type == "kde" or "kernel density" in chart_type:
        return "Density Plot"
    if "kde plot" in chart_type or "kde plot" == chart_type or "kde" in chart_type:
        return "Density Plot"
    if "ridge" in chart_type or "ridgeline" in chart_type:
        return "Density Plot"
    if "violin" in chart_type or "boxen" in chart_type or "boxplot" in chart_type:
        return "Box / Violin"
    if chart_type == "box" or "box plot" in chart_type or "box and whisker" in chart_type:
        return "Box / Violin"
    if "raincloud" in chart_type:
        return "Box / Violin"
    if "swarm" in chart_type or "strip" in chart_type or "jitter" in chart_type:
        return "Categorical Scatter"
    if "dot plot" in chart_type or chart_type == "dot":
        return "Categorical Scatter"
    if chart_type == "point" or "regression" in chart_type or "residual" in chart_type:
        return "Scatter Plot"
    if "scatter" in chart_type or "jointplot" in chart_type or "scatter map" in chart_type:
        return "Scatter Plot"
    if chart_type == "pair":
        return "Scatter Matrix"
    if "image plot" in chart_type:
        return "Heatmap"
    if "sequence logo" in chart_type or "sequence_logo" in chart_type:
        return "Sequence Logo"
    if "line" in chart_type or "time series" in chart_type or "trend line" in chart_type:
        return "Line / Area"
    if "step" in chart_type or "area" in chart_type or "profile line" in chart_type:
        return "Line / Area"
    if "bar" in chart_type or "count plot" in chart_type or "lollipop" in chart_type:
        return "Bar Chart"

    return raw_chart_type.strip()
