#!/usr/bin/env python3
"""
Re-render a stored chart from dataset metadata with paper-friendly sizing.

The dataset metadata keeps the final plotting code, chart description, and the
OpenML dataset id, but it does not keep the fully renamed dataframe schema that
the plotting code expects. This utility reconstructs the source dataframe,
infers a best-effort raw->renamed column mapping, and then re-executes the
stored plotting code with figure-size overrides.

Typical usage:

    python generation-pipeline/render_chart_from_metadata.py \
        --graph-id 7865168a-0ff4-4551-ba42-7426f8bb3eaf \
        --fig-width 4.0 \
        --fig-height 3.0 \
        --output paper/figures/example_chart.png

If automatic column reconstruction is ambiguous, provide explicit overrides:

    python generation-pipeline/render_chart_from_metadata.py \
        --graph-id 7865168a-0ff4-4551-ba42-7426f8bb3eaf \
        --column-map mcg=McGeoch_Signal_Score \
        --column-map gvh=von_Heijne_Signal_Score \
        --column-map class=Protein_Localization_Site
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable


DEFAULT_METADATA_PATH = Path(__file__).resolve().parents[1] / "dataset" / "metadata.jsonl"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "paper" / "figures"
STOPWORDS = {
    "a",
    "an",
    "and",
    "attribute",
    "attributes",
    "by",
    "class",
    "data",
    "dataset",
    "feature",
    "features",
    "for",
    "from",
    "in",
    "is",
    "nominal",
    "numeric",
    "of",
    "on",
    "or",
    "target",
    "the",
    "to",
    "used",
    "value",
    "values",
    "variable",
    "variables",
}


@dataclass
class ColumnMatch:
    raw: str
    renamed: str
    score: float
    evidence: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Re-render a chart from dataset/metadata.jsonl by reconstructing the "
            "underlying dataframe and re-running the stored plotting code."
        )
    )
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--graph-id", help="Graph UUID from dataset/metadata.jsonl")
    selector.add_argument("--index", type=int, help="0-based line index in metadata.jsonl")

    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=DEFAULT_METADATA_PATH,
        help=f"Path to metadata JSONL (default: {DEFAULT_METADATA_PATH})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output image path. Defaults to paper/figures/<graph-id>_paper.png",
    )
    parser.add_argument(
        "--dataset-file",
        type=Path,
        help=(
            "Optional local dataset file to use instead of fetching from OpenML. "
            "Supported: .csv, .tsv, .json, .jsonl, .parquet, .pkl/.pickle."
        ),
    )
    parser.add_argument(
        "--column-map",
        action="append",
        default=[],
        metavar="RAW=RENAMED",
        help=(
            "Explicit raw->renamed column override. Repeat as needed, e.g. "
            "--column-map class=Protein_Localization_Site"
        ),
    )
    parser.add_argument(
        "--code-source",
        choices=("latest-image", "graph"),
        default="latest-image",
        help="Use the latest image code revision or graph.code (default: latest-image).",
    )
    parser.add_argument("--fig-width", type=float, default=4.0, help="Output figure width in inches.")
    parser.add_argument("--fig-height", type=float, default=3.0, help="Output figure height in inches.")
    parser.add_argument("--dpi", type=int, default=300, help="Output DPI (default: 300).")
    parser.add_argument(
        "--font-scale",
        type=float,
        default=1.0,
        help="Scale explicit fontsize=... literals in the stored code (default: 1.0).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the inferred mapping and planned output path without rendering.",
    )
    return parser.parse_args()


def parse_column_overrides(items: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --column-map value {item!r}; expected RAW=RENAMED.")
        raw, renamed = item.split("=", 1)
        raw = raw.strip()
        renamed = renamed.strip()
        if not raw or not renamed:
            raise ValueError(f"Invalid --column-map value {item!r}; expected RAW=RENAMED.")
        mapping[raw] = renamed
    return mapping


def load_metadata_entry(metadata_path: Path, graph_id: str | None, index: int | None) -> tuple[int, dict]:
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    with metadata_path.open("r", encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            entry = json.loads(line)
            if graph_id is not None and entry.get("id") == graph_id:
                return row_index, entry
            if index is not None and row_index == index:
                return row_index, entry

    selector = f"graph id {graph_id}" if graph_id is not None else f"index {index}"
    raise ValueError(f"Could not find metadata entry for {selector}.")


def choose_code(entry: dict, code_source: str) -> str:
    if code_source == "graph":
        return entry["graph"]["code"]

    images = entry.get("images") or []
    if images:
        return images[-1]["code"]
    return entry["graph"]["code"]


def infer_output_path(entry: dict, requested_output: Path | None) -> Path:
    if requested_output is not None:
        return requested_output
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return DEFAULT_OUTPUT_DIR / f"{entry['id']}_paper.png"


def read_local_dataset(dataset_file: Path):
    import pandas as pd

    suffix = dataset_file.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(dataset_file)
    if suffix == ".tsv":
        return pd.read_csv(dataset_file, sep="\t")
    if suffix == ".json":
        return pd.read_json(dataset_file)
    if suffix == ".jsonl":
        return pd.read_json(dataset_file, lines=True)
    if suffix == ".parquet":
        return pd.read_parquet(dataset_file)
    if suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(dataset_file)
    raise ValueError(f"Unsupported --dataset-file format: {dataset_file.suffix}")


def load_openml_dataset(dataset_id: int):
    try:
        import openml  # type: ignore

        ds = openml.datasets.get_dataset(int(dataset_id))
        X, y, _, _ = ds.get_data(dataset_format="dataframe")
        df = X.copy()
        target_name = ds.default_target_attribute or "target"
        if y is not None:
            if hasattr(y, "name") and getattr(y, "name", None):
                target_name = y.name
            df[target_name] = y
        return df, target_name
    except ImportError:
        pass

    try:
        from sklearn.datasets import fetch_openml  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Neither `openml` nor `scikit-learn` is installed, so the source dataset "
            "cannot be reconstructed automatically. Install one of them, or pass "
            "`--dataset-file` with a local dataframe export."
        ) from exc

    try:
        bunch = fetch_openml(data_id=int(dataset_id), as_frame=True, parser="auto")
    except TypeError:
        bunch = fetch_openml(data_id=int(dataset_id), as_frame=True)
    if getattr(bunch, "frame", None) is not None:
        df = bunch.frame.copy()
    else:
        df = bunch.data.copy()
        if bunch.target is not None:
            target_name = bunch.target.name or "target"
            df[target_name] = bunch.target
    target_name = getattr(getattr(bunch, "target", None), "name", None) or "target"
    return df, target_name


def normalize_text(text: str) -> str:
    text = text.replace("_", " ")
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = text.lower().replace("'s", "")
    text = text.replace("-", " ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def meaningful_tokens(text: str) -> list[str]:
    return [token for token in normalize_text(text).split() if token and token not in STOPWORDS]


def similarity_score(target: str, candidate: str, raw_name: str) -> float:
    target_norm = normalize_text(target)
    cand_norm = normalize_text(candidate)
    if not target_norm or not cand_norm:
        return 0.0

    target_tokens = set(meaningful_tokens(target))
    cand_tokens = set(meaningful_tokens(candidate))
    overlap = 0.0
    if target_tokens and cand_tokens:
        overlap = len(target_tokens & cand_tokens) / max(len(target_tokens), len(cand_tokens))

    seq = SequenceMatcher(None, target_norm, cand_norm).ratio()
    contains = 0.85 if target_norm in cand_norm or cand_norm in target_norm else 0.0

    raw_digits = "".join(ch for ch in raw_name if ch.isdigit())
    target_digits = "".join(ch for ch in target if ch.isdigit())
    digit_bonus = 0.0
    if raw_digits and target_digits and raw_digits == target_digits:
        digit_bonus = 0.2

    raw_tokens = set(meaningful_tokens(raw_name))
    exact_token_bonus = 0.0
    if raw_tokens and target_tokens and raw_tokens == target_tokens:
        exact_token_bonus = 0.1

    return max(overlap, seq, contains) + digit_bonus + exact_token_bonus


def extract_description_aliases(description: str, target_column: str | None) -> dict[str, list[str]]:
    aliases: dict[str, list[str]] = {}

    for short_name, long_name in re.findall(r"\b([A-Za-z][A-Za-z0-9_]*)\s*\(([^()]+)\)", description):
        aliases.setdefault(short_name, []).append(long_name)

    target_matches = re.findall(
        r"target attribute[^.]*?representing\s+([^.,;]+)",
        description,
        flags=re.IGNORECASE,
    )
    if target_column and target_matches:
        aliases.setdefault(target_column, []).extend(target_matches)

    return aliases


def collect_needed_columns(entry: dict, code: str) -> list[str]:
    needed = []

    structured = entry.get("graph", {}).get("structured_data") or {}
    for col in structured.get("features_expected") or []:
        if isinstance(col, str):
            needed.append(col)

    string_refs = re.findall(r"""df\s*\[\s*['"]([^'"]+)['"]\s*\]""", code)
    needed.extend(string_refs)

    seen = set()
    ordered = []
    for col in needed:
        if col not in seen:
            seen.add(col)
            ordered.append(col)
    return ordered


def infer_column_mapping(
    raw_columns: Iterable[str],
    needed_columns: list[str],
    dataset_description: str,
    target_column: str | None,
    overrides: dict[str, str],
) -> tuple[list[ColumnMatch], dict[str, list[tuple[str, float, str]]]]:
    raw_columns = list(raw_columns)
    aliases = extract_description_aliases(dataset_description, target_column)

    reverse_overrides = {renamed: raw for raw, renamed in overrides.items()}
    used_raw = set()
    matches: list[ColumnMatch] = []
    suggestions: dict[str, list[tuple[str, float, str]]] = {}

    for needed in needed_columns:
        if needed in raw_columns:
            used_raw.add(needed)
            matches.append(ColumnMatch(raw=needed, renamed=needed, score=10.0, evidence="exact"))
            continue

        if needed in reverse_overrides:
            raw = reverse_overrides[needed]
            if raw not in raw_columns:
                raise ValueError(
                    f"Manual override requested {raw}={needed}, but raw column {raw!r} is not present."
                )
            used_raw.add(raw)
            matches.append(ColumnMatch(raw=raw, renamed=needed, score=10.0, evidence="manual"))
            continue

        scored: list[tuple[float, str, str]] = []
        for raw in raw_columns:
            evidence_candidates = [raw]
            evidence_candidates.extend(aliases.get(raw, []))

            best_score = -1.0
            best_evidence = raw
            for candidate in evidence_candidates:
                score = similarity_score(needed, candidate, raw)
                if score > best_score:
                    best_score = score
                    best_evidence = candidate
            scored.append((best_score, raw, best_evidence))

        scored.sort(key=lambda item: item[0], reverse=True)
        suggestions[needed] = [(raw, score, evidence) for score, raw, evidence in scored[:5]]

        chosen_score, chosen_raw, chosen_evidence = scored[0]
        if chosen_score < 0.40:
            continue
        if chosen_raw in used_raw:
            for alt_score, alt_raw, alt_evidence in scored[1:]:
                if alt_raw not in used_raw and alt_score >= 0.40:
                    chosen_score, chosen_raw, chosen_evidence = alt_score, alt_raw, alt_evidence
                    break
            else:
                continue

        used_raw.add(chosen_raw)
        matches.append(
            ColumnMatch(
                raw=chosen_raw,
                renamed=needed,
                score=chosen_score,
                evidence=chosen_evidence,
            )
        )

    return matches, suggestions


def build_render_dataframe(df, matches: list[ColumnMatch]):
    render_df = df.copy()
    for match in matches:
        if match.renamed not in render_df.columns and match.raw in render_df.columns:
            render_df[match.renamed] = render_df[match.raw]
    return render_df


def validate_mapping(needed_columns: list[str], matches: list[ColumnMatch], suggestions: dict[str, list[tuple[str, float, str]]]):
    resolved = {match.renamed for match in matches}
    missing = [col for col in needed_columns if col not in resolved]
    if not missing:
        return

    lines = ["Could not infer all renamed columns required by the stored plotting code."]
    for col in missing:
        lines.append(f"  - Missing: {col}")
        for raw, score, evidence in suggestions.get(col, [])[:3]:
            lines.append(f"      suggestion: {raw}  score={score:.2f}  evidence={evidence!r}")
    lines.append("Add explicit overrides with --column-map RAW=RENAMED.")
    raise RuntimeError("\n".join(lines))


def patch_fontsizes(code: str, font_scale: float) -> str:
    if font_scale == 1.0:
        return code

    def repl(match: re.Match[str]) -> str:
        value = float(match.group(1))
        scaled = value * font_scale
        if scaled.is_integer():
            value_text = str(int(scaled))
        else:
            value_text = f"{scaled:.2f}".rstrip("0").rstrip(".")
        return f"fontsize={value_text}"

    return re.sub(r"fontsize\s*=\s*([0-9]+(?:\.[0-9]+)?)", repl, code)


@contextmanager
def matplotlib_overrides(fig_width: float, fig_height: float, dpi: int):
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    original_subplots = plt.subplots
    original_figure = plt.figure
    original_savefig = plt.savefig
    original_figure_savefig = Figure.savefig

    def patched_subplots(*args, **kwargs):
        kwargs["figsize"] = (fig_width, fig_height)
        return original_subplots(*args, **kwargs)

    def patched_figure(*args, **kwargs):
        kwargs["figsize"] = (fig_width, fig_height)
        return original_figure(*args, **kwargs)

    def patched_savefig(*args, **kwargs):
        kwargs.setdefault("dpi", dpi)
        kwargs.setdefault("bbox_inches", "tight")
        return original_savefig(*args, **kwargs)

    def patched_figure_savefig(self, *args, **kwargs):
        kwargs.setdefault("dpi", dpi)
        kwargs.setdefault("bbox_inches", "tight")
        return original_figure_savefig(self, *args, **kwargs)

    plt.subplots = patched_subplots
    plt.figure = patched_figure
    plt.savefig = patched_savefig
    Figure.savefig = patched_figure_savefig
    try:
        yield
    finally:
        plt.subplots = original_subplots
        plt.figure = original_figure
        plt.savefig = original_savefig
        Figure.savefig = original_figure_savefig


def render_chart(
    entry: dict,
    code: str,
    render_df,
    output_path: Path,
    fig_width: float,
    fig_height: float,
    dpi: int,
    font_scale: float,
):
    import matplotlib
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)

    selected_plot = {
        "type": entry["graph"]["type"],
        "features": (entry["graph"].get("structured_data") or {}).get("features_expected", []),
        "style": entry["graph"].get("style", "default"),
    }

    patched_code = patch_fontsizes(code, font_scale)

    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    plt.style.use("default")
    if selected_plot["style"] in plt.style.available:
        plt.style.use(selected_plot["style"])

    with matplotlib_overrides(fig_width, fig_height, dpi):
        exec_ns = {
            "df": render_df,
            "selected_plot": selected_plot,
            "graph_file_path": str(output_path),
            "__builtins__": __builtins__,
        }
        exec(patched_code, exec_ns, exec_ns)

    return patched_code


def print_mapping(matches: list[ColumnMatch]) -> None:
    print("Resolved column mapping:")
    for match in matches:
        score = "manual" if match.evidence == "manual" else f"{match.score:.2f}"
        print(f"  {match.raw} -> {match.renamed}  ({score}; evidence={match.evidence})")


def main() -> int:
    args = parse_args()
    overrides = parse_column_overrides(args.column_map)
    row_index, entry = load_metadata_entry(args.metadata_path, args.graph_id, args.index)
    code = choose_code(entry, args.code_source)
    output_path = infer_output_path(entry, args.output)

    if args.dataset_file is not None:
        df = read_local_dataset(args.dataset_file)
        target_column = None
    else:
        df, target_column = load_openml_dataset(int(entry["dataset"]["id"]))

    needed_columns = collect_needed_columns(entry, code)
    matches, suggestions = infer_column_mapping(
        raw_columns=list(df.columns),
        needed_columns=needed_columns,
        dataset_description=entry["dataset"]["description"],
        target_column=target_column,
        overrides=overrides,
    )
    validate_mapping(needed_columns, matches, suggestions)
    render_df = build_render_dataframe(df, matches)

    print(f"Metadata row: {row_index}")
    print(f"Graph id: {entry['id']}")
    print(f"Dataset id: {entry['dataset']['id']}")
    print(f"Graph type: {entry['graph']['type']}")
    print_mapping(matches)
    print(f"Output: {output_path}")

    if args.dry_run:
        return 0

    try:
        render_chart(
            entry=entry,
            code=code,
            render_df=render_df,
            output_path=output_path,
            fig_width=args.fig_width,
            fig_height=args.fig_height,
            dpi=args.dpi,
            font_scale=args.font_scale,
        )
    except KeyError as exc:
        missing = exc.args[0]
        raise RuntimeError(
            f"The plotting code still requested an unresolved column {missing!r}. "
            "Add an explicit --column-map RAW=RENAMED override for that field."
        ) from exc

    print("Render complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
