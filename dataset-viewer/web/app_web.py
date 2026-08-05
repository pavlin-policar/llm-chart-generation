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


def generation_dataset_names(rows: list[dict], manifest: dict) -> list[str]:
    names = [
        str(entry.get("name", ""))
        for entry in manifest.get("generation_datasets", [])
        if entry.get("name")
    ]
    if not names:
        names = sorted({
            str(row.get("generation_dataset", "dataset"))
            for row in rows
            if row.get("generation_dataset", "dataset")
        })
    return sorted(dict.fromkeys(names))


def _metric_rate_label(metric: dict | None) -> str:
    if not metric:
        return "-"
    count = int(metric.get("count", metric.get("accepted_count", 0)) or 0)
    denominator = int(metric.get("denominator", 0) or 0)
    rate = metric.get("rate")
    if not denominator or rate is None:
        return f"N/A ({count:,}/{denominator:,})"
    return f"{float(rate):.1%} ({count:,}/{denominator:,})"


def render_generation_metrics(
    manifest: dict,
    selected_generation: str,
    standalone: bool = False,
) -> None:
    """Render per-folder acceptance progression and generation error rates."""
    metrics = [
        entry
        for entry in manifest.get("generation_datasets", [])
        if selected_generation == "(all)" or entry.get("name") == selected_generation
    ]
    metrics = [
        metric
        for metric in metrics
        if metric.get("acceptance_by_iteration") or metric.get("error_rates")
    ]
    if not metrics:
        st.info(
            "Dataset generation metrics are unavailable in this bundle. "
            "Regenerate it with the current prepare_ftp_bundle.py."
        )
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


def init_state(rows: list[dict], manifest: dict) -> None:
    qp = st.query_params
    type_counts = Counter(r.get("canonical_type", "") for r in rows)
    ds_entries, ds_labels = dataset_maps(manifest)
    ds_counts = Counter(str(r.get("dataset_id", "?")) for r in rows)
    generation_names = set(generation_dataset_names(rows, manifest))

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
    if "generation_filter" not in st.session_state:
        qp_generation = qp.get("generation") or ""
        st.session_state["generation_filter"] = (
            qp_generation if qp_generation in generation_names else "(all)"
        )
    if "acceptance_filter" not in st.session_state:
        qp_acceptance = qp.get("accepted") or ""
        st.session_state["acceptance_filter"] = (
            qp_acceptance if qp_acceptance in ("accepted", "rejected") else "(all)"
        )
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
    generation_dataset: str = "(all)",
    acceptance: str = "(all)",
) -> dict[str, str]:
    out: dict[str, str] = {}
    if sel_type != "(all)":
        out["type"] = sel_type
    if sel_dataset != "(all)":
        out["dataset"] = sel_dataset
    if generation_dataset != "(all)":
        out["generation"] = generation_dataset
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
        str(row.get("generation_dataset", "dataset")),
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
    sel_generation: str = "(all)",
    acceptance: str = "(all)",
) -> list[dict]:
    out = rows
    if sel_type != "(all)":
        out = [r for r in out if r.get("canonical_type") == sel_type]
    if sel_dataset != "(all)":
        out = [r for r in out if str(r.get("dataset_id", "?")) == sel_dataset]
    if sel_generation != "(all)":
        out = [
            r for r in out if str(r.get("generation_dataset", "dataset")) == sel_generation
        ]
    if quality != "(all)":
        out = [r for r in out if r.get("quality") == quality]
    if acceptance == "accepted":
        out = [r for r in out if bool(r.get("accepted"))]
    elif acceptance == "rejected":
        out = [r for r in out if not bool(r.get("accepted"))]
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
    st.session_state["generation_filter"] = "(all)"
    st.session_state["acceptance_filter"] = "(all)"
    st.session_state["sort_by"] = "Default"
    st.session_state["sort_asc"] = True
    st.session_state["selected_id"] = None
    st.query_params.clear()


def render_sidebar(rows: list[dict], manifest: dict) -> tuple[str, str, str, str, str, bool, str, str]:
    ds_entries, ds_labels = dataset_maps(manifest)
    active_quality = st.session_state.get("quality_filter", "good")
    active_search = st.session_state.get("search", "").strip().lower()
    active_type_display = st.session_state.get("type_filter", "(all)")
    active_type = "(all)" if active_type_display == "(all)" else active_type_display.rsplit(" (", 1)[0]
    active_dataset = selected_dataset_id(st.session_state.get("dataset_filter", "(all)"))

    st.sidebar.header("Filters")
    generation_names = generation_dataset_names(rows, manifest)
    generation_counts = Counter(str(r.get("generation_dataset", "dataset")) for r in rows)
    active_generation = st.session_state.get("generation_filter", "(all)")
    if active_generation != "(all)" and active_generation not in generation_names:
        st.session_state["generation_filter"] = "(all)"
        active_generation = "(all)"
    generation_options = ["(all)"] + generation_names
    selected_generation = st.sidebar.selectbox(
        "Generation dataset",
        generation_options,
        format_func=lambda name: name if name == "(all)" else f"{name} ({generation_counts[name]})",
        key="generation_filter",
    )


    good_count = len(
        rows_matching(
            rows, active_type, active_dataset, active_search, "good", selected_generation
        )
    )
    bad_count = len(
        rows_matching(
            rows, active_type, active_dataset, active_search, "bad", selected_generation
        )
    )
    all_quality_count = len(
        rows_matching(
            rows, active_type, active_dataset, active_search, "(all)", selected_generation
        )
    )
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
    acceptance_count_rows = rows_matching(
        rows, active_type, active_dataset, active_search, quality, selected_generation
    )
    accepted_count = sum(1 for row in acceptance_count_rows if bool(row.get("accepted")))
    rejected_count = len(acceptance_count_rows) - accepted_count
    acceptance_labels = {
        "(all)": f"All ({len(acceptance_count_rows)})",
        "accepted": f"Accepted ({accepted_count})",
        "rejected": f"Not accepted ({rejected_count})",
    }
    acceptance = st.sidebar.selectbox(
        "Acceptance",
        ["(all)", "accepted", "rejected"],
        format_func=lambda value: acceptance_labels[value],
        key="acceptance_filter",
    )


    type_count_rows = rows_matching(
        rows, "(all)", active_dataset, active_search, quality, selected_generation, acceptance
    )
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

    ds_count_rows = rows_matching(
        rows, selected_type, "(all)", active_search, quality, selected_generation, acceptance
    )
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
    return (
        selected_generation,
        selected_type,
        selected_dataset,
        search,
        sort_by,
        st.session_state["sort_asc"],
        quality,
        acceptance,
    )

def filter_records(
    rows: list[dict],
    sel_type: str,
    sel_dataset: str,
    search: str,
    quality: str,
    generation_dataset: str,
    acceptance: str,
) -> list[dict]:
    return rows_matching(
        rows, sel_type, sel_dataset, search, quality, generation_dataset, acceptance
    )


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
  position: relative;
  padding: 8px 30px 8px 10px;
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
        status_class = "accepted" if bool(rec.get("accepted")) else "rejected"
        status_label = "Accepted" if bool(rec.get("accepted")) else "Not accepted"
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
            f'    <span class="chart-card__status chart-card__status--{status_class}" title="{status_label}"></span>'
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
            generation_name = chart_row.get("generation_dataset")
            if generation_name:
                detail_rel = f"charts/{generation_name}/{chart_row['dataset_id']}/{chart_row['id']}/"
            else:
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
    accepted = bool(chart_row.get("accepted", rec.get("accepted")))
    status_class = "accepted" if accepted else "rejected"
    status_label = "Accepted" if accepted else "Not accepted"

    st.button("← Back to grid", on_click=clear_selection)
    st.subheader(f"{chart_block.get('type', 'Chart')}  —  {chart_row.get('canonical_type', '')}")
    st.markdown(
        f"""
        <div class="chart-meta-row">
          <code>{html.escape(str(gid))}</code>
          <span class="chart-status-tag chart-status-tag--{status_class}">
            <span class="chart-status-tag__dot">&#9679;</span>{status_label}
          </span>
          <span>&middot; generation dataset <code>{html.escape(str(chart_row.get('generation_dataset', 'dataset')))}</code>
          &middot; source dataset #{html.escape(str(ds.get('id', '?')))}</span>
        </div>
        <style>
          .chart-meta-row {{ display:flex; align-items:center; flex-wrap:wrap; gap:8px;
                             margin:-0.15rem 0 0.75rem; color:#6b7280; font-size:13px; }}
          .chart-status-tag {{ display:inline-flex; align-items:center; padding:2px 8px;
                               border-radius:999px; font-size:12px; font-weight:600; }}
          .chart-status-tag--accepted {{ color:#166534; background:#dcfce7;
                                         border:1px solid #86efac; }}
          .chart-status-tag--rejected {{ color:#991b1b; background:#fee2e2;
                                         border:1px solid #fca5a5; }}
          .chart-status-tag__dot {{ margin-right:5px; font-size:9px; }}
        </style>
        """,
        unsafe_allow_html=True,
    )

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

    page_view = render_page_navigation()
    if page_view == "stats":
        render_generation_metrics(manifest, "(all)", standalone=True)
        return

    st.title(APP_TITLE)
    st.caption(APP_SUBTITLE)

    (
        generation_dataset,
        sel_type,
        sel_dataset,
        search,
        sort_by,
        sort_asc,
        quality,
        acceptance,
    ) = render_sidebar(rows, manifest)
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
        selected = next((r for r in rows if r["id"] == st.session_state["selected_id"]), None)
        if selected is None:
            clear_selection()
            st.rerun()
        render_detail(selected, manifest)
        return

    render_generation_metrics(manifest, generation_dataset)

    filtered = filter_records(
        rows, sel_type, sel_dataset, search, quality, generation_dataset, acceptance
    )
    filtered = sort_records(filtered, sort_by, sort_asc)
    render_grid(filtered, filter_qp)


if __name__ == "__main__":
    main()
