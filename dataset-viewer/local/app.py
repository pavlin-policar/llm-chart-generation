"""Streamlit viewer for the generated chart dataset.

Run:
    streamlit run local/app.py

Set DATA_DIR to point at the repository directory containing dataset/:
    DATA_DIR=/path/to/data streamlit run local/app.py
"""
from __future__ import annotations

import base64
import html
import json
import os
from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path
from urllib.parse import urlencode

import streamlit as st
from PIL import Image

from chart_types import canonicalize_chart_type as canonicalize
from indexer import build_index, read_records

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_DEFAULT_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ["DATA_DIR"]) if "DATA_DIR" in os.environ else _DEFAULT_ROOT
DATASETS_DIR = ROOT / "dataset"
_DEFAULT_RESULTS_DIR = ROOT / "results"
if not _DEFAULT_RESULTS_DIR.exists() and (ROOT / "evaluation").exists():
    _DEFAULT_RESULTS_DIR = ROOT / "evaluation"
RESULTS_DIR = Path(os.environ["RESULTS_DIR"]) if "RESULTS_DIR" in os.environ else _DEFAULT_RESULTS_DIR
CACHE_DIR = Path(__file__).resolve().parent / ".cache"

THUMBS_PER_PAGE = 24
GRID_COLS = 4

st.set_page_config(page_title="Chart Dataset Viewer", layout="wide")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _iter_sort_key(img: dict) -> int:
    stem = Path(img.get("path", "")).stem
    if "_it" in stem:
        try:
            return int(stem.rsplit("_it", 1)[1])
        except ValueError:
            pass
    return 0


_REFINEMENT_MAX_ROUNDS = 3


def _quality(rec: dict) -> str:
    """'bad' if the chart reached _it3+ AND that final image still has feedback."""
    images = [
        img for img in rec.get("images", [])
        if isinstance(img, dict) and isinstance(img.get("path"), str)
    ]
    if not images:
        return "good"
    last = max(images, key=_iter_sort_key)
    if _iter_sort_key(last) < _REFINEMENT_MAX_ROUNDS:
        return "good"
    fb = last.get("feedback") or ""
    if isinstance(fb, list):
        fb = " ".join(str(x).strip() for x in fb if str(x).strip())
    return "bad" if str(fb).strip() else "good"

def chart_accepted(rec: dict) -> bool:
    """Return the chart-level acceptance flag, with an iteration fallback."""
    accepted = rec.get("accepted")
    if isinstance(accepted, bool):
        return accepted
    images = ordered_iterations(rec.get("images", []) or [])
    return bool(images and images[-1].get("accept"))



def discover_dataset_dirs() -> dict[str, Path]:
    """Return folder-name -> dataset directory for every usable dataset."""
    dataset_dirs: dict[str, Path] = {}

    # Backward compatibility for the former flat dataset/ layout.
    if (DATASETS_DIR / "metadata.jsonl").is_file():
        dataset_dirs[DATASETS_DIR.name] = DATASETS_DIR

    if DATASETS_DIR.is_dir():
        for child in sorted(DATASETS_DIR.iterdir(), key=lambda path: path.name.lower()):
            if child.is_dir() and (child / "metadata.jsonl").is_file():
                dataset_dirs[child.name] = child
    return dataset_dirs


def dataset_dir(dataset_name: str) -> Path:
    try:
        return discover_dataset_dirs()[dataset_name]
    except KeyError as exc:
        raise FileNotFoundError(f"Unknown generation dataset: {dataset_name}") from exc

@st.cache_data(show_spinner=False)
def load_error_counts(dataset_name: str) -> dict[str, int]:
    """Load generation-stage error counts from JSON or JSONL error logs."""
    counts: Counter = Counter()
    source_dir = dataset_dir(dataset_name)
    error_file = next(
        (
            source_dir / name
            for name in ("error.jsonl", "errors.jsonl", "error.json", "errors.json")
            if (source_dir / name).is_file()
        ),
        None,
    )
    if error_file is None:
        return {}

    rows: list[dict] = []
    try:
        if error_file.suffix == ".jsonl":
            with error_file.open(encoding="utf-8") as f:
                for line in f:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(row, dict):
                        rows.append(row)
        else:
            payload = json.loads(error_file.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                rows = [row for row in payload if isinstance(row, dict)]
            elif isinstance(payload, dict):
                nested = payload.get("errors")
                rows = (
                    [row for row in nested if isinstance(row, dict)]
                    if isinstance(nested, list)
                    else [payload]
                )
    except (OSError, json.JSONDecodeError):
        return {}

    for row in rows:
        stage = row.get("stage")
        if stage:
            counts[str(stage)] += 1
    return dict(counts)



@st.cache_resource(show_spinner="Loading metadata...")
def load_metadata(dataset_name: str) -> list[dict]:
    records: list[dict] = []
    metadata_file = dataset_dir(dataset_name) / "metadata.jsonl"
    with metadata_file.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            rec["_canonical_type"] = canonicalize(rec.get("graph", {}).get("type", ""))
            rec["_quality"] = _quality(rec)
            rec["_generation_dataset"] = dataset_name
            rec["_accepted"] = chart_accepted(rec)
            records.append(rec)
    return records


@st.cache_resource(show_spinner="Indexing model results…")
def load_result_indexes() -> dict[str, dict[str, list[tuple[int, int]]]]:
    out: dict[str, dict[str, list[tuple[int, int]]]] = {}
    for jl in sorted(RESULTS_DIR.glob("*.jsonl")):
        out[jl.stem] = build_index(jl, CACHE_DIR)
    return out


@st.cache_resource(show_spinner="Computing per-chart answer stats…")
def compute_per_chart_stats() -> dict[str, dict[str, int]]:
    """Return {chart_id: {correct: N, incorrect: N}} summed across all models.

    Results are persisted to .cache/per_chart_stats.pkl and only recomputed
    when the result files change.
    """
    import pickle
    from collections import defaultdict

    cache_file = CACHE_DIR / "per_chart_stats.pkl"
    result_files = sorted(RESULTS_DIR.glob("*.jsonl"))

    fingerprint = tuple(
        (f.name, f.stat().st_size, int(f.stat().st_mtime)) for f in result_files
    )

    if cache_file.exists():
        try:
            with cache_file.open("rb") as f:
                cached = pickle.load(f)
            if cached.get("fingerprint") == fingerprint:
                return cached["stats"]
        except (pickle.PickleError, EOFError, KeyError):
            pass

    totals: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "incorrect": 0})
    for jl in result_files:
        idx = build_index(jl, CACHE_DIR)
        for gid, locs in idx.items():
            for r in read_records(jl, locs):
                if r.get("correct"):
                    totals[gid]["correct"] += 1
                else:
                    totals[gid]["incorrect"] += 1

    stats = dict(totals)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with cache_file.open("wb") as f:
        pickle.dump({"fingerprint": fingerprint, "stats": stats}, f)

    return stats


def resolve_image(path_str: str, dataset_name: str) -> Path | None:
    """Resolve a metadata image path against its generation dataset folder."""
    if not path_str:
        return None
    p = Path(path_str)
    if p.is_absolute() and p.exists():
        return p
    source_dir = dataset_dir(dataset_name)
    cand = source_dir / p
    if cand.exists():
        return cand
    cand = source_dir / "images" / p.name
    return cand if cand.exists() else None


@st.cache_data(show_spinner=False, max_entries=2048)
def thumbnail_data_uri(path: str, width: int = 360) -> str:
    """Return a base64 data URI for the given image, resized to width px."""
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            ratio = width / im.width if im.width > width else 1.0
            if ratio < 1.0:
                im = im.resize((width, int(im.height * ratio)), Image.LANCZOS)
            buf = BytesIO()
            im.save(buf, format="JPEG", quality=82)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return ""


def ordered_iterations(images: list[dict]) -> list[dict]:
    """Deduplicate and sort by 'it{n}' suffix in filename."""
    seen: dict[str, dict] = {}
    for img in images:
        path = img.get("path", "")
        if not path or path in seen:
            continue
        seen[path] = img
    items = list(seen.values())

    def key(img: dict) -> int:
        stem = Path(img.get("path", "")).stem
        if "_it" in stem:
            try:
                return int(stem.rsplit("_it", 1)[1])
            except ValueError:
                pass
        return 0

    return sorted(items, key=key)


def compute_generation_metrics(dataset_name: str, records: list[dict]) -> dict:
    """Compute cumulative acceptance and generation error rates for one folder."""
    chart_count = len(records)
    image_lists = [
        [img for img in (rec.get("images") or []) if isinstance(img, dict)]
        for rec in records
    ]
    iteration_output_count = sum(len(images) for images in image_lists)
    regeneration_attempt_count = max(iteration_output_count - chart_count, 0)
    max_iteration_count = max((len(images) for images in image_lists), default=0)
    accepted_iteration_outputs = 0
    accepted_chart_count = 0
    for rec, images in zip(records, image_lists):
        first_acceptance = next(
            (index for index, image in enumerate(images) if bool(image.get("accept"))),
            None,
        )
        if first_acceptance is None and rec.get("accepted") is True and images:
            first_acceptance = len(images) - 1
        if first_acceptance is not None:
            accepted_chart_count += 1
            accepted_iteration_outputs += first_acceptance + 1


    acceptance_by_iteration = []
    for iteration in range(max_iteration_count):
        accepted_count = 0
        for rec, images in zip(records, image_lists):
            accepted = any(bool(img.get("accept")) for img in images[: iteration + 1])
            # Backward compatibility when only the chart-level final flag exists.
            if (
                not accepted
                and rec.get("accepted") is True
                and images
                and iteration >= len(images) - 1
            ):
                accepted = True
            accepted_count += int(accepted)
        acceptance_by_iteration.append({
            "iteration": iteration,
            "accepted_count": accepted_count,
            "denominator": chart_count,
            "rate": (accepted_count / chart_count) if chart_count else None,
        })

    error_counts = load_error_counts(dataset_name)
    execution_errors = int(error_counts.get("code_execution", 0))
    regeneration_errors = int(error_counts.get("code_regeneration", 0))
    return {
        "name": dataset_name,
        "chart_count": chart_count,
        "iteration_output_count": iteration_output_count,
        "regeneration_attempt_count": regeneration_attempt_count,
        "accepted_chart_count": accepted_chart_count,
        "accepted_iteration_output_count": accepted_iteration_outputs,
        "iterations_per_accepted_chart": (
            accepted_iteration_outputs / accepted_chart_count
            if accepted_chart_count
            else None
        ),
        "acceptance_by_iteration": acceptance_by_iteration,
        "error_rates": {
            "code_execution": {
                "count": execution_errors,
                "denominator": chart_count,
                "rate": (execution_errors / chart_count) if chart_count else None,
            },
            "code_regeneration": {
                "count": regeneration_errors,
                "denominator": regeneration_attempt_count,
                "rate": (
                    regeneration_errors / regeneration_attempt_count
                    if regeneration_attempt_count
                    else None
                ),
            },
        },
    }


def _metric_rate_label(metric: dict | None) -> str:
    if not metric:
        return "-"
    count = int(metric.get("count", metric.get("accepted_count", 0)) or 0)
    denominator = int(metric.get("denominator", 0) or 0)
    rate = metric.get("rate")
    if not denominator or rate is None:
        return f"N/A ({count:,}/{denominator:,})"
    return f"{float(rate):.1%} ({count:,}/{denominator:,})"


def render_generation_metrics(metrics: list[dict], standalone: bool = False) -> None:
    """Render exact per-iteration acceptance and stage error rates."""
    if not metrics:
        return
    iterations = sorted({
        int(point["iteration"])
        for metric in metrics
        for point in metric.get("acceptance_by_iteration", [])
    })

    acceptance_rows = []
    error_rows = []
    for metric in metrics:
        points = {
            int(point["iteration"]): point
            for point in metric.get("acceptance_by_iteration", [])
        }
        average_iterations = metric.get("iterations_per_accepted_chart")
        accepted_count = int(metric.get("accepted_chart_count", 0))
        accepted_outputs = int(metric.get("accepted_iteration_output_count", 0))
        iterations_label = (
            f"{float(average_iterations):.2f} ({accepted_outputs:,}/{accepted_count:,})"
            if average_iterations is not None else "-"
        )
        acceptance_row = {
            "Generation dataset": metric["name"],
            "Charts": int(metric.get("chart_count", 0)),
            "Iterations / accepted chart": iterations_label,
        }
        for iteration in iterations:
            acceptance_row[f"Accept @ it{iteration}"] = _metric_rate_label(
                points.get(iteration)
            )
        acceptance_rows.append(acceptance_row)

        error_rates = metric.get("error_rates") or {}
        error_rows.append({
            "Generation dataset": metric["name"],
            "Iteration outputs": int(metric.get("iteration_output_count", 0)),
            "Regeneration attempts": int(metric.get("regeneration_attempt_count", 0)),
            "Code execution": _metric_rate_label(error_rates.get("code_execution")),
            "Code regeneration": _metric_rate_label(error_rates.get("code_regeneration")),
        })

    if standalone:
        st.title("Dataset generation statistics")
        metrics_container = st.container()
    else:
        metrics_container = st.expander("Dataset generation metrics", expanded=True)

    with metrics_container:
        st.markdown("**Cumulative acceptance by iteration**")
        st.caption(
            "Once a chart is accepted, it remains accepted for every later iteration. "
            "Each rate is accepted charts divided by all generated charts. "
            "Iterations per accepted chart counts outputs through first acceptance; "
            "the initial it0 output counts as one."
        )
        st.dataframe(acceptance_rows, use_container_width=True, hide_index=True)

        st.markdown("**Generation error rates**")
        st.caption(
            "Code execution uses all generated charts as its denominator. "
            "Code regeneration uses iteration outputs minus generated charts."
        )
        st.dataframe(error_rows, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
def render_page_navigation() -> str:
    """Render sidebar navigation and return the active top-level page."""
    page = st.query_params.get("view") or "charts"
    if page not in {"charts", "stats"}:
        page = "charts"

    labels = {"charts": "Charts", "stats": "Statistics"}
    state_key = "page_navigation"
    if st.session_state.get(state_key) != labels[page]:
        st.session_state[state_key] = labels[page]

    def change_page() -> None:
        target = "stats" if st.session_state[state_key] == "Statistics" else "charts"
        st.session_state["selected_id"] = None
        st.query_params.clear()
        if target == "stats":
            st.query_params["view"] = "stats"
        else:
            generation = (
                st.session_state.get("generation_dataset")
                or st.session_state.get("generation_filter")
            )
            if generation:
                st.query_params["generation"] = generation

    st.sidebar.markdown("**Pages**")
    st.sidebar.segmented_control(
        "Pages",
        ["Charts", "Statistics"],
        key=state_key,
        on_change=change_page,
        label_visibility="collapsed",
        width="stretch",
    )
    st.sidebar.markdown("---")
    return page


def _generation_dataset_changed() -> None:
    """Clear dataset-specific state and immediately persist the new folder."""
    generation_dataset = st.session_state["generation_dataset"]
    for key in (
        "type_filter",
        "dataset_filter",
        "quality_filter",
        "quality_filter_display",
        "search",
        "acceptance_filter",
        "page",
        "sort_by",
        "sort_asc",
        "selected_id",
        "_last_filter",
    ):
        st.session_state.pop(key, None)
    st.query_params.clear()
    st.query_params["generation"] = generation_dataset


def render_generation_dataset_selector(dataset_names: list[str]) -> str:
    requested = st.query_params.get("generation") or ""
    if "generation_dataset" not in st.session_state:
        st.session_state["generation_dataset"] = (
            requested if requested in dataset_names else dataset_names[0]
        )

    st.sidebar.header("Generation dataset")
    selected = st.sidebar.selectbox(
        "Dataset folder",
        dataset_names,
        key="generation_dataset",
        on_change=_generation_dataset_changed,
    )
    st.sidebar.caption(f"dataset/{selected}")
    st.sidebar.markdown("---")
    return selected


def dataset_short_label(description: str) -> str:
    """Best-effort short label from a dataset description."""
    if not description:
        return "(no description)"
    first = description.strip().split(".")[0].strip()
    first = first.replace("\n", " ")
    if len(first) > 70:
        first = first[:67] + "…"
    return first or "(no description)"


def _init_state(records: list[dict]) -> None:
    """Initialize session state, restoring filters from URL when present.

    URL is the source of truth because clicking a card link causes a full
    navigation — which would otherwise wipe the filter selectbox state.
    """
    qp = st.query_params

    # Build option lists once, so we can map canonical URL values back to the
    # display strings used as selectbox values.
    type_counts = Counter(r["_canonical_type"] for r in records)
    ds_counts: Counter = Counter()
    ds_labels: dict[str, str] = {}
    for r in records:
        ds = r.get("dataset", {}) or {}
        did = str(ds.get("id", "?"))
        ds_counts[did] += 1
        if did not in ds_labels:
            ds_labels[did] = dataset_short_label(ds.get("description", ""))

    # Type filter (URL stores canonical name)
    qp_type = qp.get("type") or ""
    if "type_filter" not in st.session_state:
        if qp_type and qp_type in type_counts:
            st.session_state["type_filter"] = f"{qp_type} ({type_counts[qp_type]})"
        else:
            st.session_state["type_filter"] = "(all)"

    # Dataset filter (URL stores id)
    qp_ds = qp.get("dataset") or ""
    if "dataset_filter" not in st.session_state:
        if qp_ds and qp_ds in ds_counts:
            st.session_state["dataset_filter"] = (
                f"{ds_labels[qp_ds]}  ·  id={qp_ds} ({ds_counts[qp_ds]})"
            )
        else:
            st.session_state["dataset_filter"] = "(all)"

    # Quality filter
    if "quality_filter" not in st.session_state:
        qp_quality = qp.get("quality") or ""
        st.session_state["quality_filter"] = qp_quality if qp_quality in ("good", "bad") else "(all)"


    # Acceptance filter
    if "acceptance_filter" not in st.session_state:
        qp_acceptance = qp.get("accepted") or ""
        st.session_state["acceptance_filter"] = (
            qp_acceptance if qp_acceptance in ("accepted", "rejected") else "(all)"
        )
    # Search (text input, key='search')
    if "search" not in st.session_state:
        st.session_state["search"] = qp.get("search") or ""

    # Page
    if "page" not in st.session_state:
        try:
            st.session_state["page"] = max(0, int(qp.get("page") or 0))
        except ValueError:
            st.session_state["page"] = 0

    # Sort
    if "sort_by" not in st.session_state:
        st.session_state["sort_by"] = qp.get("sort") or "Default"
    if "sort_asc" not in st.session_state:
        st.session_state["sort_asc"] = (qp.get("asc") or "1") == "1"

    # Selection
    if "selected_id" not in st.session_state:
        st.session_state["selected_id"] = qp.get("open") or None


def current_filter_qp(
    sel_type: str,
    sel_dataset: str,
    search: str,
    page: int,
    sort_by: str,
    sort_asc: bool,
    quality: str = "(all)",
    generation_dataset: str = "",
    acceptance: str = "(all)",
) -> dict[str, str]:
    """Build the filter/sort portion of the URL query params."""
    out: dict[str, str] = {}
    if generation_dataset:
        out["generation"] = generation_dataset
    if sel_type != "(all)":
        out["type"] = sel_type
    if sel_dataset != "(all)":
        out["dataset"] = sel_dataset
    if search:
        out["search"] = search
    if quality != "(all)":
        out["quality"] = quality
    if acceptance != "(all)":
        out["accepted"] = acceptance
    if page:
        out["page"] = str(page)
    if sort_by != "Default":
        out["sort"] = sort_by
    if not sort_asc:
        out["asc"] = "0"
    return out


def sync_url(filter_qp: dict[str, str], selected_id: str | None) -> None:
    desired = dict(filter_qp)
    if selected_id:
        desired["open"] = selected_id
    current = dict(st.query_params)
    if current != desired:
        st.query_params.clear()
        for k, v in desired.items():
            st.query_params[k] = v


def clear_selection() -> None:
    st.session_state["selected_id"] = None
    if "open" in st.query_params:
        del st.query_params["open"]


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------
SORT_OPTIONS = ["Default", "Incorrect answers"]


def render_sidebar(records: list[dict]) -> tuple[str, str, str, str, bool, str, str]:
    type_counts = Counter(r["_canonical_type"] for r in records)
    types = ["(all)"] + [f"{t} ({n})" for t, n in sorted(type_counts.items(), key=lambda kv: (-kv[1], kv[0]))]

    ds_counts: Counter = Counter()
    ds_labels: dict[str, str] = {}
    for r in records:
        ds = r.get("dataset", {}) or {}
        did = str(ds.get("id", "?"))
        ds_counts[did] += 1
        if did not in ds_labels:
            ds_labels[did] = dataset_short_label(ds.get("description", ""))
    datasets = ["(all)"] + [
        f"{ds_labels[did]}  ·  id={did} ({n})"
        for did, n in sorted(ds_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    ]

    st.sidebar.header("Filters")
    sel_display = st.sidebar.selectbox(
        "Chart type",
        types,
        index=types.index(st.session_state["type_filter"]) if st.session_state["type_filter"] in types else 0,
        key="type_filter",
    )
    selected_type = "(all)" if sel_display == "(all)" else sel_display.rsplit(" (", 1)[0]

    ds_display = st.sidebar.selectbox(
        "Dataset",
        datasets,
        index=datasets.index(st.session_state["dataset_filter"]) if st.session_state["dataset_filter"] in datasets else 0,
        key="dataset_filter",
    )
    if ds_display == "(all)":
        selected_dataset = "(all)"
    else:
        # Parse back the id from "label  ·  id=X (N)"
        try:
            selected_dataset = ds_display.split("id=", 1)[1].split(" (", 1)[0]
        except IndexError:
            selected_dataset = "(all)"

    search = st.sidebar.text_input("Search (dataset / description / ID)", key="search").strip().lower()

    good_count = sum(1 for r in records if r["_quality"] == "good")
    bad_count = sum(1 for r in records if r["_quality"] == "bad")
    quality_opts = ["(all)", f"Good ({good_count})", f"Bad ({bad_count})"]
    quality_keys = {"(all)": "(all)", f"Good ({good_count})": "good", f"Bad ({bad_count})": "bad"}
    quality_display_map = {"(all)": "(all)", "good": f"Good ({good_count})", "bad": f"Bad ({bad_count})"}
    q_state = st.session_state.get("quality_filter", "(all)")
    q_display_default = quality_display_map.get(q_state, "(all)")
    q_display = st.sidebar.selectbox(
        "Plot quality",
        quality_opts,
        index=quality_opts.index(q_display_default) if q_display_default in quality_opts else 0,
        key="quality_filter_display",
    )
    quality = quality_keys.get(q_display, "(all)")
    st.session_state["quality_filter"] = quality

    accepted_count = sum(1 for r in records if r["_accepted"])
    rejected_count = len(records) - accepted_count
    acceptance_labels = {
        "(all)": f"All ({len(records)})",
        "accepted": f"Accepted ({accepted_count})",
        "rejected": f"Not accepted ({rejected_count})",
    }
    acceptance = st.sidebar.selectbox(
        "Acceptance",
        ["(all)", "accepted", "rejected"],
        format_func=lambda value: acceptance_labels[value],
        key="acceptance_filter",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Sort**")
    sort_asc = st.session_state["sort_asc"]
    _sc1, _sc2 = st.sidebar.columns([4, 1], vertical_alignment="bottom")
    with _sc1:
        sort_by = st.selectbox(
            "Sort by",
            SORT_OPTIONS,
            index=SORT_OPTIONS.index(st.session_state["sort_by"]) if st.session_state["sort_by"] in SORT_OPTIONS else 0,
            key="sort_by",
        )
    with _sc2:
        if st.button("↑" if sort_asc else "↓", use_container_width=True):
            st.session_state["sort_asc"] = not sort_asc
            st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.caption(
        f"{len(records)} total charts · {len(type_counts)} canonical types · {len(ds_counts)} datasets"
    )
    return (
        selected_type,
        selected_dataset,
        search,
        sort_by,
        st.session_state["sort_asc"],
        quality,
        acceptance,
    )


def sort_records(
    records: list[dict],
    sort_by: str,
    ascending: bool,
    stats: dict[str, dict[str, int]],
) -> list[dict]:
    if sort_by == "Incorrect answers":
        return sorted(
            records,
            key=lambda r: stats.get(r["id"], {}).get("incorrect", 0),
            reverse=not ascending,
        )
    return records  # Default: preserve metadata.jsonl order


def filter_records(
    records: list[dict],
    sel_type: str,
    sel_dataset: str,
    search: str,
    quality: str = "(all)",
    acceptance: str = "(all)",
) -> list[dict]:
    out = records
    if sel_type != "(all)":
        out = [r for r in out if r["_canonical_type"] == sel_type]
    if sel_dataset != "(all)":
        out = [r for r in out if str((r.get("dataset") or {}).get("id", "?")) == sel_dataset]
    if search:
        def match(r: dict) -> bool:
            g = r.get("graph", {})
            blob = " ".join([
                str(r.get("id", "")),
                str(r.get("dataset", {}).get("description", "")),
                str(g.get("short_description", "")),
                str(g.get("type", "")),
            ]).lower()
            return search in blob
        out = [r for r in out if match(r)]
    if quality != "(all)":
        out = [r for r in out if r["_quality"] == quality]
    if acceptance == "accepted":
        out = [r for r in out if r["_accepted"]]
    elif acceptance == "rejected":
        out = [r for r in out if not r["_accepted"]]
    return out


def _render_pagination(page: int, pages: int, total: int, position: str) -> None:
    c1, c2, c3 = st.columns([1, 2, 1])
    with c1:
        if st.button("← Prev", key=f"prev_{position}", disabled=page == 0, use_container_width=True):
            st.session_state["page"] = max(0, page - 1)
            st.rerun()
    with c2:
        st.markdown(
            f"<div style='text-align:center;padding-top:6px'>Page {page + 1} / {pages} · {total} charts</div>",
            unsafe_allow_html=True,
        )
    with c3:
        if st.button("Next →", key=f"next_{position}", disabled=page >= pages - 1, use_container_width=True):
            st.session_state["page"] = min(pages - 1, page + 1)
            st.rerun()


def render_grid(records: list[dict], filter_qp: dict[str, str]) -> None:
    st.subheader("Charts")
    total = len(records)
    if total == 0:
        st.info("No charts match the current filters.")
        return

    pages = max(1, (total + THUMBS_PER_PAGE - 1) // THUMBS_PER_PAGE)
    page = min(st.session_state["page"], pages - 1)

    _render_pagination(page, pages, total, "top")

    start = page * THUMBS_PER_PAGE
    subset = records[start:start + THUMBS_PER_PAGE]

    st.markdown(_GRID_CSS, unsafe_allow_html=True)

    cards: list[str] = []
    for rec in subset:
        iters = ordered_iterations(rec.get("images", []))
        thumb_path = resolve_image(iters[-1]["path"], rec["_generation_dataset"]) if iters else None
        data_uri = thumbnail_data_uri(str(thumb_path)) if thumb_path else ""
        label = rec["_canonical_type"]
        short_id = rec["id"][:8]
        status_class = "accepted" if rec["_accepted"] else "rejected"
        status_label = "Accepted" if rec["_accepted"] else "Not accepted"
        img_html = (
            f'<img src="{data_uri}" alt="{label}" />'
            if data_uri
            else '<div class="chart-card__noimg">no image</div>'
        )
        link_qp = {**filter_qp, "open": rec["id"]}
        href = "?" + urlencode(link_qp)
        cards.append(
            f'<a class="chart-card" href="{href}" target="_self">'
            f'  <div class="chart-card__imgwrap">{img_html}</div>'
            f'  <div class="chart-card__label"><b>{label}</b><br/>'
            f'    <span class="chart-card__id">{short_id}…</span>'
            f'    <span class="chart-card__status chart-card__status--{status_class}" title="{status_label}"></span>'
            f'  </div>'
            f'</a>'
        )
    st.markdown(f'<div class="chart-grid">{"".join(cards)}</div>', unsafe_allow_html=True)

    st.markdown("<div style='margin-top:16px'/>", unsafe_allow_html=True)
    _render_pagination(page, pages, total, "bottom")


_GRID_CSS = """
<style>
.chart-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 14px;
  margin-top: 8px;
}
.chart-card {
  display: flex;
  flex-direction: column;
  border: 1px solid #e4e4e7;
  border-radius: 8px;
  overflow: hidden;
  text-decoration: none !important;
  color: inherit !important;
  background: #fff;
  transition: border-color 120ms ease, box-shadow 120ms ease, transform 120ms ease;
}
.chart-card:hover {
  border-color: #3b82f6;
  box-shadow: 0 2px 8px rgba(59, 130, 246, 0.18);
  transform: translateY(-1px);
}
.chart-card__imgwrap {
  aspect-ratio: 4 / 3;
  display: flex;
  align-items: center;
  justify-content: center;
  background: #fafafa;
  overflow: hidden;
}
.chart-card__imgwrap img {
  width: 100%;
  height: 100%;
  object-fit: contain;
  display: block;
}
.chart-card__noimg {
  color: #9ca3af;
  font-size: 13px;
}
.chart-card__label {
  padding: 8px 10px;
  position: relative;
  padding-right: 30px;
  font-size: 13px;
  line-height: 1.35;
  border-top: 1px solid #f0f0f0;
}
.chart-card__id {
  color: #6b7280;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 11px;
}
.chart-card__status {
  position: absolute;
  right: 10px;
  bottom: 10px;
  width: 11px;
  height: 11px;
  border: 2px solid #fff;
  border-radius: 50%;
  box-shadow: 0 0 0 1px rgba(0, 0, 0, 0.12);
}
.chart-card__status--accepted {
  background: #16a34a;
}
.chart-card__status--rejected {
  background: #dc2626;
}
@media (max-width: 1100px) {
  .chart-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); }
}
@media (max-width: 800px) {
  .chart-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
</style>
"""


def _llm_call_label(index: int, call: object, has_reasoning: bool = False) -> str:
    """Build a compact selector label from the call's stage, iterations, model, and token usage."""
    parts = [f"Call {index + 1}"]
    if not isinstance(call, dict):
        return parts[0]

    metadata = call.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    stage_name = metadata.get("stage_name")
    if stage_name:
        parts.append(str(stage_name))
    regeneration_iteration = metadata.get("regeneration_iteration")
    if regeneration_iteration is not None:
        parts.append(f"regeneration {regeneration_iteration}")
    error_iteration = metadata.get("error_iteration")
    if error_iteration is not None:
        parts.append(f"error {error_iteration}")

    output = call.get("output")
    if not isinstance(output, dict):
        return " · ".join(parts)

    response_metadata = output.get("response_metadata")
    response_metadata = response_metadata if isinstance(response_metadata, dict) else {}
    model = response_metadata.get("model_name")
    if model:
        parts.append(str(model))

    usage = output.get("usage_metadata")
    usage = usage if isinstance(usage, dict) else {}
    token_usage = response_metadata.get("token_usage")
    token_usage = token_usage if isinstance(token_usage, dict) else {}
    total_tokens = usage.get("total_tokens", token_usage.get("total_tokens"))
    if isinstance(total_tokens, (int, float)):
        parts.append(f"{total_tokens:,.0f} tokens")
    if has_reasoning:
        parts.append("reasoning")
    return " · ".join(parts)


def _llm_input_messages(value: object) -> list[tuple[str, object]]:
    """Flatten common LangChain/OpenAI input envelopes into role/content pairs."""
    messages: list[tuple[str, object]] = []

    def walk(node: object) -> None:
        if isinstance(node, list):
            for child in node:
                walk(child)
            return
        if not isinstance(node, dict):
            return

        data = node.get("data")
        if isinstance(data, dict) and "content" in data:
            role = node.get("type") or data.get("role") or data.get("type") or "message"
            messages.append((str(role), data["content"]))
            return
        if "content" in node and ("role" in node or "type" in node):
            role = node.get("role") or node.get("type") or "message"
            messages.append((str(role), node["content"]))
            return

        for child in node.values():
            walk(child)

    walk(value)
    return messages


def _llm_image_placeholder(block: dict) -> str | None:
    """Return a safe summary for an image block without printing its URL/data."""
    block_type = str(block.get("type", "")).lower()
    source = block.get("source")
    source = source if isinstance(source, dict) else {}
    image_url = block.get("image_url")
    image_url = image_url if isinstance(image_url, dict) else {}
    media_type = (
        block.get("media_type")
        or block.get("mime_type")
        or source.get("media_type")
        or source.get("mime_type")
    )
    looks_like_image = (
        "image" in block_type
        or "image" in block
        or "image_url" in block
        or str(media_type or "").lower().startswith("image/")
    )
    if not looks_like_image:
        return None

    details: list[str] = []
    url = str(image_url.get("url") or block.get("url") or "")
    if not media_type and url.startswith("data:image/"):
        media_type = url[5:].split(";", 1)[0]
    if media_type:
        details.append(str(media_type))

    detail = image_url.get("detail") or block.get("detail")
    if detail:
        details.append(f"detail={detail}")
    width, height = block.get("width"), block.get("height")
    if width and height:
        details.append(f"{width}×{height}")
    if not details:
        details.append("embedded data" if url.startswith("data:") else "URL reference")
    return f"[Image included: {', '.join(details)}]"


def _llm_content_text(value: object) -> str:
    """Convert message content blocks into readable text and image markers."""
    if isinstance(value, str):
        return value
    if value is None:
        return "[No content recorded]"
    if isinstance(value, list):
        parts = [_llm_content_text(part).strip() for part in value]
        return "\n\n".join(part for part in parts if part) or "[No content recorded]"
    if isinstance(value, dict):
        image_placeholder = _llm_image_placeholder(value)
        if image_placeholder:
            return image_placeholder
        for key in ("text", "input_text", "output_text", "content"):
            if key in value:
                return _llm_content_text(value[key])
        keys = ", ".join(str(key) for key in value)
        return f"[Structured content: {keys}]" if keys else "[Empty content object]"
    return str(value)


def _llm_joined_messages(messages: list[tuple[str, object]]) -> str:
    """Join every message into one transcript while retaining its type."""
    return "\n\n".join(
        f"[{role}]\n{_llm_content_text(content)}"
        for role, content in messages
    )


def _llm_reasoning_traces(value: object) -> list[tuple[str, str]]:
    """Find and deduplicate reasoning traces regardless of provider nesting."""
    reasoning_fields = {
        "analysis",
        "reasoning",
        "reasoning_content",
        "reasoning_details",
        "reasoning_trace",
        "thinking",
        "thinking_content",
        "thinking_trace",
    }
    traces: list[tuple[str, str]] = []
    seen: set[str] = set()

    def add(path: str, content: object) -> None:
        text = _llm_content_text(content).strip()
        if not text or text == "[No content recorded]" or text in seen:
            return
        seen.add(text)
        traces.append((path, text))

    def walk(node: object, path: str) -> None:
        if isinstance(node, list):
            for index, child in enumerate(node):
                walk(child, f"{path}[{index}]")
            return
        if not isinstance(node, dict):
            return

        block_type = str(node.get("type", "")).lower()
        if block_type in {"analysis", "reasoning", "thinking"}:
            for key in ("text", "content", "reasoning", "thinking"):
                if key in node:
                    add(path, node[key])
                    break

        for key, child in node.items():
            child_path = f"{path}.{key}" if path else str(key)
            normalized_key = str(key).lower()
            if (
                normalized_key in reasoning_fields
                and isinstance(child, (str, list, dict))
                and child
            ):
                add(child_path, child)
            else:
                walk(child, child_path)

    walk(value, "call")
    return traces


def _render_llm_content(value: object) -> None:
    """Show text with decoded whitespace; fall back to JSON for structured data."""
    if isinstance(value, str):
        st.code(value, language="text", wrap_lines=True)
    elif value is None:
        st.caption("No content recorded.")
    else:
        st.json(value, expanded=False)


def render_llm_calls(rec: dict) -> None:
    """Render readable call contents plus the exact metadata.jsonl payload."""
    llm_calls = rec.get("llm_calls")
    if not isinstance(llm_calls, list):
        llm_calls = [llm_calls] if llm_calls else []

    st.subheader(f"LLM calls ({len(llm_calls)})")
    if not llm_calls:
        st.caption("No LLM calls are recorded for this chart.")
        return

    reasoning_by_call = [_llm_reasoning_traces(call) for call in llm_calls]
    reasoning_call_count = sum(bool(traces) for traces in reasoning_by_call)
    st.caption(
        f"Reasoning traces are recorded for {reasoning_call_count} of "
        f"{len(llm_calls)} calls in this chart."
    )
    selected_index = st.selectbox(
        "LLM call",
        options=range(len(llm_calls)),
        format_func=lambda index: _llm_call_label(
            index,
            llm_calls[index],
            bool(reasoning_by_call[index]),
        ),
        key=f"llm_call_{rec.get('id', 'chart')}",
    )
    selected_call = llm_calls[selected_index]

    if isinstance(selected_call, dict):
        with st.expander("Input", expanded=False):
            messages = _llm_input_messages(selected_call.get("input"))
            if messages:
                _render_llm_content(_llm_joined_messages(messages))
            else:
                _render_llm_content(_llm_content_text(selected_call.get("input")))

        reasoning_traces = reasoning_by_call[selected_index]
        with st.expander("Reasoning trace", expanded=False):
            if reasoning_traces:
                _render_llm_content("\n\n".join(
                    f"[{path}]\n{trace}" for path, trace in reasoning_traces
                ))
            else:
                st.caption("No separate reasoning trace is recorded for this call.")

        with st.expander("Output", expanded=False):
            output = selected_call.get("output")
            if isinstance(output, dict):
                _render_llm_content(_llm_content_text(output.get("content")))
            else:
                _render_llm_content(_llm_content_text(output))
    else:
        _render_llm_content(selected_call)

    with st.expander("View raw JSON for all LLM calls", expanded=False):
        st.caption("Unmodified `llm_calls` payload from this chart's metadata.jsonl record.")
        st.json(llm_calls, expanded=False)


def render_detail(rec: dict, result_indexes: dict[str, dict[str, list[tuple[int, int]]]]) -> None:
    gid = rec["id"]
    chart_block = rec.get("graph", {})
    ds = rec.get("dataset", {})
    status_class = "accepted" if rec["_accepted"] else "rejected"
    status_label = "Accepted" if rec["_accepted"] else "Not accepted"

    st.button("← Back to grid", on_click=clear_selection)
    st.subheader(f"{chart_block.get('type', 'Chart')}  —  {rec['_canonical_type']}")
    _copy_icon = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24"'
        ' fill="none" stroke="currentColor" stroke-width="2"'
        ' stroke-linecap="round" stroke-linejoin="round">'
        '<rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>'
        '<path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>'
        '</svg>'
    )
    _check_icon = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24"'
        ' fill="none" stroke="#16a34a" stroke-width="2.5"'
        ' stroke-linecap="round" stroke-linejoin="round">'
        '<polyline points="20 6 9 17 4 12"></polyline>'
        '</svg>'
    )
    st.components.v1.html(
        f"""
        <style>
          body {{ margin:0; padding:0; background:transparent; }}
          #row {{ display:flex; align-items:center; gap:8px;
                  font-size:13px; color:#6b7280;
                  font-family:system-ui,-apple-system,sans-serif; }}
          #pill {{ display:inline-flex; align-items:center; gap:5px; cursor:pointer;
                   font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:12px;
                   color:#c2410c; padding:2px 7px; background:#fff7ed;
                   border-radius:4px; border:1px solid #fed7aa; user-select:none; }}
          #pill:hover {{ background:#ffedd5; }}
          #icon-wrap {{ position:relative; width:13px; height:13px; }}
          .status-tag {{ display:inline-flex; align-items:center; padding:2px 8px;
                         border-radius:999px; font-size:12px; font-weight:600; }}
          .status-tag--accepted {{ color:#166534; background:#dcfce7;
                                    border:1px solid #86efac; }}
          .status-tag--rejected {{ color:#991b1b; background:#fee2e2;
                                    border:1px solid #fca5a5; }}
          .status-tag__dot {{ margin-right:5px; font-size:9px; }}

          #icon-copy, #icon-check {{
            position:absolute; top:0; left:0;
            display:inline-flex; align-items:center;
            transition: opacity 400ms ease, transform 400ms ease;
          }}
          #icon-check {{ opacity:0; transform:scale(0.5); }}
        </style>
        <div id="row">
          <div id="pill" onclick="doCopy()">
            <span>{gid}</span>
            <span id="icon-wrap">
              <span id="icon-copy">{_copy_icon}</span>
              <span id="icon-check">{_check_icon}</span>
            </span>
          </div>
          <span class="status-tag status-tag--{status_class}"><span class="status-tag__dot">&#9679;</span>{status_label}</span>
          <span>· generation dataset {html.escape(rec["_generation_dataset"])}
          · source dataset #{ds.get("id", "?")}</span>
        </div>
        <script>
          var busy = false;
          function doCopy() {{
            if (busy) return;
            navigator.clipboard.writeText('{gid}').then(function() {{
              busy = true;
              var cp = document.getElementById('icon-copy');
              var ck = document.getElementById('icon-check');
              cp.style.opacity = '0'; cp.style.transform = 'scale(0.5)';
              ck.style.opacity = '1'; ck.style.transform = 'scale(1)';
              setTimeout(function() {{
                ck.style.opacity = '0'; ck.style.transform = 'scale(0.5)';
                cp.style.opacity = '1'; cp.style.transform = 'scale(1)';
                setTimeout(function() {{ busy = false; }}, 400);
              }}, 1500);
            }});
          }}
        </script>
        """,
        height=34,
    )

    iters = ordered_iterations(rec.get("images", []))

    left, right = st.columns([3, 2])

    with left:
        if iters:
            labels = [f"it{i}" for i in range(len(iters))]
            last_idx = len(iters) - 1
            iter_key = f"iter_{gid}"
            if iter_key not in st.session_state:
                st.session_state[iter_key] = last_idx
            chosen = st.radio(
                "Iteration",
                options=list(range(len(iters))),
                format_func=lambda i: labels[i],
                horizontal=True,
                key=iter_key,
            )
            img = iters[chosen]
            resolved = resolve_image(img.get("path", ""), rec["_generation_dataset"])
            if resolved:
                st.image(str(resolved), use_container_width=True)
            else:
                st.warning(f"Image not found: {img.get('path')}")

            fb_raw = img.get("feedback") or ""
            if isinstance(fb_raw, list):
                fb = "\n".join(f"- {str(item).strip()}" for item in fb_raw if str(item).strip())
            else:
                fb = str(fb_raw).strip()
            code = (img.get("code") or "").strip()

            with st.container(border=True):
                st.markdown(f"**Iteration {chosen} feedback**")
                if fb:
                    st.markdown(fb)
                else:
                    st.caption("No feedback recorded for this iteration.")

            if code:
                with st.expander("Iteration code", expanded=False):
                    st.code(code, language="python")
            else:
                st.caption("No code recorded for this iteration.")
        else:
            st.info("No images recorded for this chart.")

    with right:
        short = (chart_block.get("short_description") or "").strip()
        full = (chart_block.get("full_description") or "").strip()
        if short:
            st.markdown("**Summary**")
            st.markdown(short)
        if ds.get("description"):
            with st.expander("Dataset description"):
                st.markdown(ds["description"])
        if full:
            with st.expander("Full chart description"):
                st.markdown(full)
        if chart_block.get("code"):
            with st.expander("Final generation code"):
                st.code(chart_block["code"], language="python")
        sd = chart_block.get("structured_data")
        if sd:
            with st.expander("Structured data (JSON)"):
                st.json(sd, expanded=False)

    st.markdown("---")
    render_llm_calls(rec)
    st.markdown("---")
    render_questions(gid, chart_block.get("questions", []) or [], result_indexes)


def render_questions(
    gid: str,
    questions: list[dict],
    result_indexes: dict[str, dict[str, list[tuple[int, int]]]],
) -> None:
    st.subheader(f"Questions ({len(questions)}) & model answers")
    if not questions:
        st.caption("No questions for this chart.")
        return

    # Collect all per-model records for this chart, once.
    per_model_records: dict[str, list[dict]] = {}
    for model_name, index in result_indexes.items():
        locs = index.get(gid)
        if not locs:
            continue
        jl = RESULTS_DIR / f"{model_name}.jsonl"
        per_model_records[model_name] = read_records(jl, locs)

    # Group per question-text for fast lookup.
    by_question: dict[str, dict[str, dict]] = defaultdict(dict)
    for model_name, recs in per_model_records.items():
        for r in recs:
            qtext = (r.get("question") or {}).get("question", "")
            if qtext:
                by_question[qtext][model_name] = r

    # Overall accuracy chart
    if per_model_records:
        import altair as alt
        import pandas as pd

        acc_rows = []
        for model_name, recs in per_model_records.items():
            total = len(recs)
            correct = sum(1 for r in recs if r.get("correct"))
            acc_rows.append({
                "model": model_name,
                "accuracy": (correct / total) if total else 0.0,
                "correct": correct,
                "total": total,
            })
        df = pd.DataFrame(acc_rows).sort_values("accuracy", ascending=False)

        with st.expander(f"Per-model accuracy on this chart ({len(df)} models)", expanded=True):
            chart = (
                alt.Chart(df)
                .mark_bar()
                .encode(
                    x=alt.X("model:N", sort=df["model"].tolist(), title=None,
                            axis=alt.Axis(labelAngle=-35)),
                    y=alt.Y("accuracy:Q",
                            scale=alt.Scale(domain=[0, 1], clamp=True, nice=False),
                            axis=alt.Axis(format=".0%")),
                    tooltip=["model", alt.Tooltip("accuracy:Q", format=".1%"),
                             "correct", "total"],
                )
                .properties(height=340)
                .configure_view(stroke=None)
            )
            st.altair_chart(chart, use_container_width=True)
    else:
        st.caption("No model result records found for this chart.")

    # Pre-compute per-question accuracy for header coloring.
    def q_accuracy(qtext: str) -> float | None:
        model_hits = by_question.get(qtext)
        if not model_hits:
            return None
        results = list(model_hits.values())
        return sum(1 for r in results if r.get("correct")) / len(results)

    for i, q in enumerate(questions, 1):
        qtext = q.get("question", "")
        qtype = q.get("type", "")
        answer = q.get("answer", "")
        basis = q.get("answer_basis", "")

        frac = q_accuracy(qtext)
        marker_id = f"qacc-{gid[:8]}-{i}"
        if frac is not None:
            hue = int(frac * 120)  # 0 = red, 120 = green
            bg = f"hsl({hue}, 65%, 88%)"
            bg_hover = f"hsl({hue}, 65%, 82%)"
            st.markdown(
                f"<style>"
                f"div:has(#{marker_id}) + div [data-testid='stExpander'] details summary {{"
                f"  background-color: {bg} !important;"
                f"}}"
                f"div:has(#{marker_id}) + div [data-testid='stExpander'] details summary:hover {{"
                f"  background-color: {bg_hover} !important;"
                f"}}"
                f"</style>"
                f'<div id="{marker_id}"></div>',
                unsafe_allow_html=True,
            )

        with st.expander(f"Q{i}. [{qtype}] {qtext}", expanded=False):
            st.markdown(f"**Ground truth:** {answer}")
            if basis:
                st.caption(f"Answer basis: {basis}")

            model_rows = []
            for model_name in sorted(result_indexes.keys()):

                r = by_question.get(qtext, {}).get(model_name)
                if not r:
                    continue
                model_rows.append({
                    "model": model_name,
                    "correct": "✓" if r.get("correct") else "✗",
                    "answer": str(r.get("test_answer", "")),
                })
            if model_rows:
                st.dataframe(model_rows, use_container_width=True, hide_index=True)
            else:
                st.caption("No model answers recorded for this question.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    dataset_dirs = discover_dataset_dirs()
    if not dataset_dirs:
        st.error(
            f"No datasets found under {DATASETS_DIR}. Expected folders containing metadata.jsonl."
        )
        return

    page_view = render_page_navigation()
    if page_view == "stats":
        metrics = [
            compute_generation_metrics(name, load_metadata(name))
            for name in dataset_dirs
        ]
        render_generation_metrics(metrics, standalone=True)
        return

    generation_dataset = render_generation_dataset_selector(list(dataset_dirs))
    records = load_metadata(generation_dataset)
    result_indexes = load_result_indexes()
    chart_stats = compute_per_chart_stats()
    _init_state(records)

    (
        sel_type,
        sel_dataset,
        search,
        sort_by,
        sort_asc,
        quality,
        acceptance,
    ) = render_sidebar(records)

    # Reset page AND selection if filter changed — jump back to the grid.
    filter_key = (generation_dataset, sel_type, sel_dataset, search, sort_by, sort_asc, quality, acceptance)
    last_filter = st.session_state.get("_last_filter")
    if last_filter is not None and last_filter != filter_key:
        st.session_state["page"] = 0
        clear_selection()
    st.session_state["_last_filter"] = filter_key

    filter_qp = current_filter_qp(
        sel_type,
        sel_dataset,
        search,
        st.session_state["page"],
        sort_by,
        sort_asc,
        quality,
        generation_dataset,
        acceptance,
    )
    sync_url(filter_qp, st.session_state["selected_id"])

    if st.session_state["selected_id"]:
        selected = next((r for r in records if r["id"] == st.session_state["selected_id"]), None)
        if selected is None:
            clear_selection()
            st.rerun()
        else:
            render_detail(selected, result_indexes)
            return

    render_generation_metrics([
        compute_generation_metrics(generation_dataset, records)
    ])

    filtered = filter_records(records, sel_type, sel_dataset, search, quality, acceptance)
    filtered = sort_records(filtered, sort_by, sort_asc, chart_stats)
    render_grid(filtered, filter_qp)


if __name__ == "__main__":
    main()
