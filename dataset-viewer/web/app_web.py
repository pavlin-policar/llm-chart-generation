"""Lightweight Streamlit viewer backed by remote static chart files.

Deploy this file on Streamlit Community Cloud. It loads the small global
manifest and chart index from a public HTTP(S) location, then fetches one
chart's metadata and results only when a chart detail view is opened.
"""
from __future__ import annotations

import gzip
import html
import json
import os
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlencode

import streamlit as st


DEFAULT_MANIFEST_URL = "https://file.biolab.si/llm-chart-generation/manifest.json"
APP_TITLE = "Validation-Driven LLM Workflows for Statistical Chart Generation"
APP_SUBTITLE = (
    "Dataset viewer for generated statistical charts from tabular data, with chart descriptions, "
    "question-answer pairs, and multimodal model responses."
)
THUMBS_PER_PAGE = 24
SORT_OPTIONS = ["Default", "Incorrect answers"]
DETAIL_IMAGE_MAX_HEIGHT_PX = 720

st.set_page_config(page_title=APP_TITLE, layout="wide")


# ---------------------------------------------------------------------------
# Remote data loading
# ---------------------------------------------------------------------------
def manifest_url() -> str:
    url = os.environ.get("REMOTE_MANIFEST_URL") or os.environ.get("MANIFEST_URL")
    try:
        url = st.secrets.get("REMOTE_MANIFEST_URL", url)
    except Exception:
        pass
    return (url or DEFAULT_MANIFEST_URL).strip()


def join_url(base: str, rel: str) -> str:
    return f"{base.rstrip('/')}/{rel.lstrip('/')}"


def fetch_bytes(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "streamlit-chart-viewer/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read()


def fetch_json(url: str) -> dict:
    return json.loads(fetch_bytes(url).decode("utf-8"))


def fetch_jsonl_gz(url: str) -> list[dict]:
    rows = []
    for line in gzip.decompress(fetch_bytes(url)).decode("utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


@st.cache_data(show_spinner="Loading dataset manifest...")
def load_manifest(url: str) -> dict:
    return fetch_json(url)


@st.cache_data(show_spinner="Loading chart index...")
def load_chart_index(manifest: dict) -> list[dict]:
    index_url = manifest.get("chart_index_url")
    if not index_url:
        index_url = join_url(manifest.get("base_url", ""), manifest["chart_index"])
    return fetch_jsonl_gz(index_url)


@st.cache_data(show_spinner="Loading chart details...")
def load_chart_detail(metadata_url: str, results_url: str, detail_url: str) -> dict:
    return {
        "base_url": detail_url,
        "metadata": fetch_json(metadata_url),
        "results": fetch_jsonl_gz(results_url),
    }


# ---------------------------------------------------------------------------
# State and filtering
# ---------------------------------------------------------------------------
def dataset_short_label(description: str) -> str:
    if not description:
        return "(no description)"
    first = description.strip().split(". ", 1)[0].strip().replace("\n", " ")
    return (first[:67] + "...") if len(first) > 70 else (first or "(no description)")


def dataset_maps(manifest: dict) -> tuple[dict[str, dict], dict[str, str]]:
    entries = {str(d["id"]): d for d in manifest.get("datasets", [])}
    labels = {did: dataset_short_label(d.get("description", "")) for did, d in entries.items()}
    return entries, labels


def init_state(rows: list[dict], manifest: dict) -> None:
    qp = st.query_params
    type_counts = Counter(r.get("canonical_type", "") for r in rows)
    ds_entries, ds_labels = dataset_maps(manifest)
    ds_counts = Counter(str(r.get("dataset_id", "?")) for r in rows)

    if "type_filter" not in st.session_state:
        qp_type = qp.get("type") or ""
        st.session_state["type_filter"] = (
            f"{qp_type} ({type_counts[qp_type]})" if qp_type in type_counts else "(all)"
        )
    if "dataset_filter" not in st.session_state:
        qp_ds = qp.get("dataset") or ""
        if qp_ds in ds_entries:
            st.session_state["dataset_filter"] = (
                f"{ds_labels[qp_ds]}  ·  id={qp_ds} ({ds_counts[qp_ds]})"
            )
        else:
            st.session_state["dataset_filter"] = "(all)"
    if "quality_filter" not in st.session_state:
        qp_quality = qp.get("quality") or ""
        st.session_state["quality_filter"] = qp_quality if qp_quality in ("good", "bad", "(all)") else "good"
    if "search" not in st.session_state:
        st.session_state["search"] = qp.get("search") or ""
    if "page" not in st.session_state:
        try:
            st.session_state["page"] = max(0, int(qp.get("page") or 0))
        except ValueError:
            st.session_state["page"] = 0
    if "sort_by" not in st.session_state:
        st.session_state["sort_by"] = qp.get("sort") or "Default"
    if "sort_asc" not in st.session_state:
        st.session_state["sort_asc"] = (qp.get("asc") or "1") == "1"
    st.session_state["selected_id"] = qp.get("open") or None


def current_filter_qp(
    sel_type: str,
    sel_dataset: str,
    search: str,
    page: int,
    sort_by: str,
    sort_asc: bool,
    quality: str,
) -> dict[str, str]:
    out: dict[str, str] = {}
    if sel_type != "(all)":
        out["type"] = sel_type
    if sel_dataset != "(all)":
        out["dataset"] = sel_dataset
    if search:
        out["search"] = search
    if quality != "(all)":
        out["quality"] = quality
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
    if dict(st.query_params) != desired:
        st.query_params.clear()
        for key, value in desired.items():
            st.query_params[key] = value


def clear_selection() -> None:
    st.session_state["selected_id"] = None
    if "open" in st.query_params:
        del st.query_params["open"]


def search_matches(row: dict, search: str) -> bool:
    if not search:
        return True
    blob = " ".join([
        str(row.get("id", "")),
        str(row.get("dataset_id", "")),
        str(row.get("dataset_description", "")),
        str(row.get("short_description", "")),
        str(row.get("graph_type", "")),
        str(row.get("canonical_type", "")),
    ]).lower()
    return search in blob


def rows_matching(
    rows: list[dict],
    sel_type: str = "(all)",
    sel_dataset: str = "(all)",
    search: str = "",
    quality: str = "(all)",
) -> list[dict]:
    out = rows
    if sel_type != "(all)":
        out = [r for r in out if r.get("canonical_type") == sel_type]
    if sel_dataset != "(all)":
        out = [r for r in out if str(r.get("dataset_id", "?")) == sel_dataset]
    if quality != "(all)":
        out = [r for r in out if r.get("quality") == quality]
    if search:
        out = [r for r in out if search_matches(r, search)]
    return out


def selected_dataset_id(display: str) -> str:
    if display == "(all)":
        return "(all)"
    try:
        return display.split("id=", 1)[1].split(" (", 1)[0]
    except IndexError:
        return "(all)"


def reset_filters() -> None:
    st.session_state["quality_filter"] = "good"
    st.session_state.pop("quality_filter_display", None)
    st.session_state["type_filter"] = "(all)"
    st.session_state["dataset_filter"] = "(all)"
    st.session_state["search"] = ""
    st.session_state["page"] = 0
    st.session_state["sort_by"] = "Default"
    st.session_state["sort_asc"] = True
    st.session_state["selected_id"] = None
    st.query_params.clear()


def render_sidebar(rows: list[dict], manifest: dict) -> tuple[str, str, str, str, bool, str]:
    ds_entries, ds_labels = dataset_maps(manifest)
    active_quality = st.session_state.get("quality_filter", "good")
    active_search = st.session_state.get("search", "").strip().lower()
    active_type_display = st.session_state.get("type_filter", "(all)")
    active_type = "(all)" if active_type_display == "(all)" else active_type_display.rsplit(" (", 1)[0]
    active_dataset = selected_dataset_id(st.session_state.get("dataset_filter", "(all)"))

    st.sidebar.header("Filters")

    good_count = len(rows_matching(rows, active_type, active_dataset, active_search, "good"))
    bad_count = len(rows_matching(rows, active_type, active_dataset, active_search, "bad"))
    all_quality_count = len(rows_matching(rows, active_type, active_dataset, active_search, "(all)"))
    quality_opts = [f"Good ({good_count})", f"Bad ({bad_count})", f"All ({all_quality_count})"]
    quality_keys = {
        f"Good ({good_count})": "good",
        f"Bad ({bad_count})": "bad",
        f"All ({all_quality_count})": "(all)",
    }
    quality_display_map = {
        "good": f"Good ({good_count})",
        "bad": f"Bad ({bad_count})",
        "(all)": f"All ({all_quality_count})",
    }
    q_default = quality_display_map.get(active_quality, f"Good ({good_count})")
    q_display = st.sidebar.selectbox(
        "Plot quality",
        quality_opts,
        index=quality_opts.index(q_default) if q_default in quality_opts else 0,
        key="quality_filter_display",
    )
    quality = quality_keys.get(q_display, "good")
    st.session_state["quality_filter"] = quality

    type_count_rows = rows_matching(rows, "(all)", active_dataset, active_search, quality)
    type_counts = Counter(r.get("canonical_type", "") for r in type_count_rows)
    types = ["(all)"] + [
        f"{t} ({n})" for t, n in sorted(type_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    ]
    if active_type != "(all)" and active_type not in type_counts:
        st.session_state["type_filter"] = "(all)"
    elif active_type != "(all)":
        st.session_state["type_filter"] = f"{active_type} ({type_counts[active_type]})"

    sel_display = st.sidebar.selectbox(
        "Chart type",
        types,
        index=types.index(st.session_state["type_filter"]) if st.session_state["type_filter"] in types else 0,
        key="type_filter",
    )
    selected_type = "(all)" if sel_display == "(all)" else sel_display.rsplit(" (", 1)[0]

    ds_count_rows = rows_matching(rows, selected_type, "(all)", active_search, quality)
    ds_counts = Counter(str(r.get("dataset_id", "?")) for r in ds_count_rows)
    datasets = ["(all)"] + [
        f"{ds_labels.get(did, did)}  ·  id={did} ({n})"
        for did, n in sorted(ds_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    ]
    if active_dataset != "(all)" and active_dataset not in ds_counts:
        st.session_state["dataset_filter"] = "(all)"
    elif active_dataset != "(all)":
        st.session_state["dataset_filter"] = (
            f"{ds_labels.get(active_dataset, active_dataset)}  ·  "
            f"id={active_dataset} ({ds_counts[active_dataset]})"
        )

    ds_display = st.sidebar.selectbox(
        "Dataset",
        datasets,
        index=datasets.index(st.session_state["dataset_filter"]) if st.session_state["dataset_filter"] in datasets else 0,
        key="dataset_filter",
    )
    if ds_display == "(all)":
        selected_dataset = "(all)"
    else:
        selected_dataset = selected_dataset_id(ds_display)

    search = st.sidebar.text_input("Search (dataset / description / ID)", key="search").strip().lower()
    st.sidebar.button("Reset filters", use_container_width=True, on_click=reset_filters)

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Sort**")
    sort_asc = st.session_state["sort_asc"]
    sc1, sc2 = st.sidebar.columns([4, 1], vertical_alignment="bottom")
    with sc1:
        sort_by = st.selectbox(
            "Sort by",
            SORT_OPTIONS,
            index=SORT_OPTIONS.index(st.session_state["sort_by"]) if st.session_state["sort_by"] in SORT_OPTIONS else 0,
            key="sort_by",
        )
    with sc2:
        if st.button("↑" if sort_asc else "↓", use_container_width=True):
            st.session_state["sort_asc"] = not sort_asc
            st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.caption(
        f"{len(rows)} total charts · {len(type_counts)} canonical types · "
        f"{len(ds_entries)} datasets · remote mode"
    )
    return selected_type, selected_dataset, search, sort_by, st.session_state["sort_asc"], quality


def filter_records(rows: list[dict], sel_type: str, sel_dataset: str, search: str, quality: str) -> list[dict]:
    return rows_matching(rows, sel_type, sel_dataset, search, quality)


def sort_records(rows: list[dict], sort_by: str, ascending: bool) -> list[dict]:
    if sort_by == "Incorrect answers":
        return sorted(rows, key=lambda r: int(r.get("incorrect") or 0), reverse=not ascending)
    return rows


# ---------------------------------------------------------------------------
# Grid view
# ---------------------------------------------------------------------------
GRID_CSS = """
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
  display: block;
  margin-top: 3px;
  color: #6b7280;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 11px;
  overflow-wrap: anywhere;
  word-break: break-word;
}
@media (max-width: 1100px) {
  .chart-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); }
}
@media (max-width: 800px) {
  .chart-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
</style>
"""


def render_pagination(page: int, pages: int, total: int, position: str) -> None:
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


def render_grid(rows: list[dict], filter_qp: dict[str, str]) -> None:
    st.subheader("Charts")
    total = len(rows)
    if total == 0:
        st.info("No charts match the current filters.")
        return

    pages = max(1, (total + THUMBS_PER_PAGE - 1) // THUMBS_PER_PAGE)
    page = min(st.session_state["page"], pages - 1)
    render_pagination(page, pages, total, "top")

    start = page * THUMBS_PER_PAGE
    subset = rows[start:start + THUMBS_PER_PAGE]
    st.markdown(GRID_CSS, unsafe_allow_html=True)

    cards: list[str] = []
    for rec in subset:
        label = html.escape(str(rec.get("canonical_type", "")))
        chart_id = html.escape(str(rec["id"]))
        thumb = rec.get("thumbnail_url") or ""
        if thumb:
            img_html = f'<img src="{html.escape(thumb)}" alt="{label}" loading="lazy" />'
        else:
            img_html = '<div class="chart-card__noimg">no image</div>'
        link_qp = {**dict(st.query_params), **filter_qp, "open": rec["id"]}
        href = "?" + urlencode(link_qp)
        cards.append(
            f'<a class="chart-card" href="{html.escape(href)}" target="_self">'
            f'  <div class="chart-card__imgwrap">{img_html}</div>'
            f'  <div class="chart-card__label"><b>{label}</b><br/>'
            f'    <span class="chart-card__id">{chart_id}</span>'
            f'  </div>'
            f'</a>'
        )
    st.markdown(f'<div class="chart-grid">{"".join(cards)}</div>', unsafe_allow_html=True)
    st.markdown("<div style='margin-top:16px'/>", unsafe_allow_html=True)
    render_pagination(page, pages, total, "bottom")


# ---------------------------------------------------------------------------
# Detail view
# ---------------------------------------------------------------------------
def image_iter_sort_key(img: dict) -> int:
    stem = Path(img.get("path", "")).stem
    if "_it" in stem:
        try:
            return int(stem.rsplit("_it", 1)[1])
        except ValueError:
            pass
    return 0


def ordered_iterations(images: list[dict]) -> list[dict]:
    seen: dict[str, dict] = {}
    for img in images:
        path = img.get("path", "")
        if path and path not in seen:
            seen[path] = img
    return sorted(seen.values(), key=image_iter_sort_key)


def resolve_detail_image_url(detail_url: str, path_str: str) -> str | None:
    if not path_str:
        return None
    return join_url(detail_url, path_str)


def render_detail_image(image_url: str, alt: str) -> None:
    safe_url = html.escape(image_url, quote=True)
    safe_alt = html.escape(alt, quote=True)
    st.markdown(
        f"""
        <div class="chart-detail-image">
          <img src="{safe_url}" alt="{safe_alt}" />
        </div>
        <style>
          .chart-detail-image {{
            display: flex;
            justify-content: center;
            width: 100%;
            margin-top: 0.25rem;
          }}
          .chart-detail-image img {{
            display: block;
            width: 100%;
            max-height: min(70vh, {DETAIL_IMAGE_MAX_HEIGHT_PX}px);
            object-fit: contain;
          }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def chart_detail_urls(chart_row: dict, manifest: dict) -> tuple[str, str, str]:
    detail_url = chart_row.get("detail_url")
    if not detail_url:
        detail_rel = chart_row.get("detail")
        if not detail_rel:
            detail_rel = f"charts/{chart_row['dataset_id']}/{chart_row['id']}/"
        detail_url = join_url(manifest.get("base_url", ""), detail_rel)
    metadata_url = chart_row.get("metadata_url") or join_url(detail_url, "metadata.json")
    results_url = chart_row.get("results_url") or join_url(detail_url, "results.jsonl.gz")
    return detail_url, metadata_url, results_url


def render_detail(chart_row: dict, manifest: dict) -> None:
    detail_url, metadata_url, results_url = chart_detail_urls(chart_row, manifest)

    detail = load_chart_detail(metadata_url, results_url, detail_url)
    rec = detail["metadata"]

    gid = rec["id"]
    chart_block = rec.get("graph", {}) or {}
    ds = rec.get("dataset", {}) or {}

    st.button("← Back to grid", on_click=clear_selection)
    st.subheader(f"{chart_block.get('type', 'Chart')}  —  {chart_row.get('canonical_type', '')}")
    st.caption(f"`{gid}` · dataset #{ds.get('id', '?')}")

    iters = ordered_iterations(rec.get("images", []) or [])
    left, right = st.columns([3, 2])

    with left:
        if iters:
            labels = [f"it{i}" for i in range(len(iters))]
            iter_key = f"iter_{gid}"
            if iter_key not in st.session_state:
                st.session_state[iter_key] = len(iters) - 1
            chosen = st.radio(
                "Iteration",
                options=list(range(len(iters))),
                format_func=lambda i: labels[i],
                horizontal=True,
                key=iter_key,
            )
            img = iters[chosen]
            image_url = resolve_detail_image_url(detail_url, img.get("path", ""))
            if image_url:
                render_detail_image(image_url, f"{chart_block.get('type', 'Chart')} iteration {chosen}")
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
        if chart_block.get("structured_data"):
            with st.expander("Structured data (JSON)"):
                st.json(chart_block["structured_data"], expanded=False)

    st.markdown("---")
    render_questions(gid, chart_block.get("questions", []) or [], detail["results"], manifest.get("models", []))


def render_questions(gid: str, questions: list[dict], results: list[dict], models: list[str]) -> None:
    st.subheader(f"Questions ({len(questions)}) & model answers")
    if not questions:
        st.caption("No questions for this chart.")
        return

    chart_results = [
        r for r in results
        if str(r.get("chart_id") or r.get("graph_id") or "") == gid
    ]
    per_model_records: dict[str, list[dict]] = defaultdict(list)
    by_question: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in chart_results:
        model = str(r.get("model", "unknown"))
        per_model_records[model].append(r)
        qtext = (r.get("question") or {}).get("question", "")
        if qtext:
            by_question[qtext][model] = r

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
                    x=alt.X("model:N", sort=df["model"].tolist(), title=None, axis=alt.Axis(labelAngle=-35)),
                    y=alt.Y(
                        "accuracy:Q",
                        scale=alt.Scale(domain=[0, 1], clamp=True, nice=False),
                        axis=alt.Axis(format=".0%"),
                    ),
                    tooltip=["model", alt.Tooltip("accuracy:Q", format=".1%"), "correct", "total"],
                )
                .properties(height=340)
                .configure_view(stroke=None)
            )
            st.altair_chart(chart, use_container_width=True)
    else:
        st.caption("No model result records found for this chart.")

    def q_accuracy(qtext: str) -> float | None:
        hits = by_question.get(qtext)
        if not hits:
            return None
        values = list(hits.values())
        return sum(1 for r in values if r.get("correct")) / len(values)

    model_order = models or sorted(per_model_records.keys())
    for i, q in enumerate(questions, 1):
        qtext = q.get("question", "")
        qtype = q.get("type", "")
        answer = q.get("answer", "")
        basis = q.get("answer_basis", "")

        frac = q_accuracy(qtext)
        marker_id = f"qacc-{gid[:8]}-{i}"
        if frac is not None:
            hue = int(frac * 120)
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
            for model_name in model_order:
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
    m_url = manifest_url()
    manifest = load_manifest(m_url)
    rows = load_chart_index(manifest)
    init_state(rows, manifest)

    st.title(APP_TITLE)
    st.caption(APP_SUBTITLE)

    sel_type, sel_dataset, search, sort_by, sort_asc, quality = render_sidebar(rows, manifest)
    filter_key = (sel_type, sel_dataset, search, sort_by, sort_asc, quality)
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
    )
    sync_url(filter_qp, st.session_state["selected_id"])

    if st.session_state["selected_id"]:
        selected = next((r for r in rows if r["id"] == st.session_state["selected_id"]), None)
        if selected is None:
            clear_selection()
            st.rerun()
        render_detail(selected, manifest)
        return

    filtered = filter_records(rows, sel_type, sel_dataset, search, quality)
    filtered = sort_records(filtered, sort_by, sort_asc)
    render_grid(filtered, filter_qp)


if __name__ == "__main__":
    main()
