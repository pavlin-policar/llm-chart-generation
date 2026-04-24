"""Streamlit viewer for the generated chart dataset.

Run:
    streamlit run app.py

Set DATA_DIR to point at the directory containing dataset/ and results/:
    DATA_DIR=/path/to/data streamlit run app.py
"""
from __future__ import annotations

import base64
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
_DEFAULT_ROOT = Path(__file__).resolve().parent.parent
ROOT = Path(os.environ["DATA_DIR"]) if "DATA_DIR" in os.environ else _DEFAULT_ROOT
DATASET_DIR = ROOT / "dataset"
IMAGES_DIR = DATASET_DIR / "images"
METADATA_FILE = DATASET_DIR / "metadata.jsonl"
RESULTS_DIR = Path(os.environ["RESULTS_DIR"]) if "RESULTS_DIR" in os.environ else ROOT / "results"
CACHE_DIR = Path(__file__).resolve().parent / ".cache"

THUMBS_PER_PAGE = 24
GRID_COLS = 4

st.set_page_config(page_title="Chart Dataset Viewer", layout="wide")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading metadata…")
def load_metadata() -> list[dict]:
    records: list[dict] = []
    with METADATA_FILE.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            rec["_canonical_type"] = canonicalize(rec.get("graph", {}).get("type", ""))
            records.append(rec)
    return records


@st.cache_resource(show_spinner="Indexing model results…")
def load_result_indexes() -> dict[str, dict[str, list[tuple[int, int]]]]:
    out: dict[str, dict[str, list[tuple[int, int]]]] = {}
    for jl in sorted(RESULTS_DIR.glob("*.jsonl")):
        out[jl.stem] = build_index(jl, CACHE_DIR)
    return out


@st.cache_resource(show_spinner="Computing per-chart answer stats…")
def compute_per_graph_stats() -> dict[str, dict[str, int]]:
    """Return {graph_id: {correct: N, incorrect: N}} summed across all models.

    Results are persisted to .cache/per_graph_stats.pkl and only recomputed
    when the result files change.
    """
    import pickle
    from collections import defaultdict

    cache_file = CACHE_DIR / "per_graph_stats.pkl"
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


def resolve_image(path_str: str) -> Path | None:
    """Metadata stores paths like 'images/foo.png' — resolve against dataset/."""
    if not path_str:
        return None
    p = Path(path_str)
    if p.is_absolute() and p.exists():
        return p
    cand = DATASET_DIR / p
    if cand.exists():
        return cand
    cand = IMAGES_DIR / p.name
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


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
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
    sel_type: str, sel_dataset: str, search: str, page: int, sort_by: str, sort_asc: bool
) -> dict[str, str]:
    """Build the filter/sort portion of the URL query params."""
    out: dict[str, str] = {}
    if sel_type != "(all)":
        out["type"] = sel_type
    if sel_dataset != "(all)":
        out["dataset"] = sel_dataset
    if search:
        out["search"] = search
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


def render_sidebar(records: list[dict]) -> tuple[str, str, str, str, bool]:
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

    search = st.sidebar.text_input("Search (dataset / description)", key="search").strip().lower()

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
    return selected_type, selected_dataset, search, sort_by, st.session_state["sort_asc"]


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


def filter_records(records: list[dict], sel_type: str, sel_dataset: str, search: str) -> list[dict]:
    out = records
    if sel_type != "(all)":
        out = [r for r in out if r["_canonical_type"] == sel_type]
    if sel_dataset != "(all)":
        out = [r for r in out if str((r.get("dataset") or {}).get("id", "?")) == sel_dataset]
    if search:
        def match(r: dict) -> bool:
            g = r.get("graph", {})
            blob = " ".join([
                str(r.get("dataset", {}).get("description", "")),
                str(g.get("short_description", "")),
                str(g.get("type", "")),
            ]).lower()
            return search in blob
        out = [r for r in out if match(r)]
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
        thumb_path = resolve_image(iters[-1]["path"]) if iters else None
        data_uri = thumbnail_data_uri(str(thumb_path)) if thumb_path else ""
        label = rec["_canonical_type"]
        short_id = rec["id"][:8]
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
  font-size: 13px;
  line-height: 1.35;
  border-top: 1px solid #f0f0f0;
}
.chart-card__id {
  color: #6b7280;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 11px;
}
@media (max-width: 1100px) {
  .chart-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); }
}
@media (max-width: 800px) {
  .chart-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
</style>
"""


def render_detail(rec: dict, result_indexes: dict[str, dict[str, list[tuple[int, int]]]]) -> None:
    gid = rec["id"]
    graph = rec.get("graph", {})
    ds = rec.get("dataset", {})

    st.button("← Back to grid", on_click=clear_selection)
    st.subheader(f"{graph.get('type', 'Chart')}  —  {rec['_canonical_type']}")
    st.caption(f"id `{gid}` · dataset #{ds.get('id', '?')}")

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
            resolved = resolve_image(img.get("path", ""))
            if resolved:
                st.image(str(resolved), use_container_width=True)
            else:
                st.warning(f"Image not found: {img.get('path')}")

            fb_raw = img.get("feedback") or ""
            if isinstance(fb_raw, list):
                fb = "\n".join(f"- {str(item).strip()}" for item in fb_raw if str(item).strip())
            else:
                fb = str(fb_raw).strip()
            if fb:
                with st.expander("Iteration feedback", expanded=False):
                    st.markdown(fb)
            code = (img.get("code") or "").strip()
            if code:
                with st.expander("Iteration code", expanded=False):
                    st.code(code, language="python")
        else:
            st.info("No images recorded for this chart.")

    with right:
        short = (graph.get("short_description") or "").strip()
        full = (graph.get("full_description") or "").strip()
        if short:
            st.markdown("**Summary**")
            st.markdown(short)
        if ds.get("description"):
            with st.expander("Dataset description"):
                st.markdown(ds["description"])
        if full:
            with st.expander("Full chart description"):
                st.markdown(full)
        if graph.get("code"):
            with st.expander("Final generation code"):
                st.code(graph["code"], language="python")
        sd = graph.get("structured_data")
        if sd:
            with st.expander("Structured data (JSON)"):
                st.json(sd, expanded=False)

    st.markdown("---")
    render_questions(gid, graph.get("questions", []) or [], result_indexes)


def render_questions(
    gid: str,
    questions: list[dict],
    result_indexes: dict[str, dict[str, list[tuple[int, int]]]],
) -> None:
    st.subheader(f"Questions ({len(questions)}) & model answers")
    if not questions:
        st.caption("No questions for this chart.")
        return

    # Collect all per-model records for this graph, once.
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
    records = load_metadata()
    result_indexes = load_result_indexes()
    graph_stats = compute_per_graph_stats()
    _init_state(records)

    sel_type, sel_dataset, search, sort_by, sort_asc = render_sidebar(records)

    # Reset page AND selection if filter changed — jump back to the grid.
    filter_key = (sel_type, sel_dataset, search, sort_by, sort_asc)
    last_filter = st.session_state.get("_last_filter")
    if last_filter is not None and last_filter != filter_key:
        st.session_state["page"] = 0
        clear_selection()
    st.session_state["_last_filter"] = filter_key

    filter_qp = current_filter_qp(sel_type, sel_dataset, search, st.session_state["page"], sort_by, sort_asc)
    sync_url(filter_qp, st.session_state["selected_id"])

    if st.session_state["selected_id"]:
        selected = next((r for r in records if r["id"] == st.session_state["selected_id"]), None)
        if selected is None:
            clear_selection()
            st.rerun()
        else:
            render_detail(selected, result_indexes)
            return

    filtered = filter_records(records, sel_type, sel_dataset, search)
    filtered = sort_records(filtered, sort_by, sort_asc, graph_stats)
    render_grid(filtered, filter_qp)


if __name__ == "__main__":
    main()
