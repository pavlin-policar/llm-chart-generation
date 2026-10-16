"""Run from the repository root: python -m streamlit run question-label/app.py."""
from __future__ import annotations

import os
import secrets
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import streamlit as st
from PIL import Image

from labeling import (
    annotation_statistics, discover_datasets, flag_label, index_dataset, load_chart_image, load_question, read_responses,
    response_path, save_response, selected_image, stratified_queue, validity_flags,
)


APP_DIR = Path(__file__).resolve().parent
REPO_DIR = APP_DIR.parent
CHOICES = ["Yes", "No", "Unsure", "Partly"]
VALUES = {"Yes": True, "No": False, "Unsure": None, "Partly": "partly"}


@st.cache_data(show_spinner="Indexing dataset questions…", max_entries=3)
def cached_index(path: str, size: int, modified_ns: int) -> dict:
    """Index charts with existing, nonempty selected images."""
    return index_dataset(Path(path))


@st.cache_data(show_spinner=False, max_entries=8)
def cached_chart_image(path: str, size: int, modified_ns: int) -> Image.Image:
    return load_chart_image(Path(path))


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def render_statistics(panel, entries: list[dict], responses: list[dict], annotator: str, dataset_name: str) -> None:
    stats = annotation_statistics(entries, responses)
    with panel:
        st.caption(f"Dataset: {dataset_name} · Annotator: {annotator} · Only saved responses for questions currently in this dataset are included.")
        columns = st.columns(4)
        columns[0].metric("Available questions", stats["total"])
        columns[1].metric("Graphs with questions", stats["graphs"])
        columns[2].metric("Saved reviews", stats["reviewed"])
        columns[3].metric("Review coverage", f"{stats['reviewed'] / stats['total']:.1%}" if stats["total"] else "N/A")
        st.markdown("**Model validity distribution — all available questions**")
        st.dataframe(stats["strata"], hide_index=True, width="stretch")
        st.markdown("**Model vs. human agreement — final saved labels**")
        st.caption("Human labels are the reference; model validator flags are predictions. Data and visual / semantic validity are evaluated separately. Agreement is the fraction of comparable pairs with matching labels.")

        def formatted(rows: list[dict]) -> list[dict]:
            return [{key: ("N/A" if value is None else f"{value:.1%}") if key in {"Agreement", "Precision", "Recall", "F1"} else value
                     for key, value in row.items()} for row in rows]

        st.dataframe(formatted([{key: value for key, value in row.items() if " / Model " not in key}
                                for row in stats["dimensions"]]), hide_index=True, width="stretch")
        st.markdown("**Precision, recall, and F1 — positive class specified per row**")
        st.caption("Precision: among model predictions of the specified class, how many humans labeled that class? Recall: among human labels of that class, how many the model found? F1 is their harmonic mean. Human support counts reference examples. N/A means a zero denominator.")
        st.dataframe(formatted(stats["classes"]), hide_index=True, width="stretch")
        st.markdown("**Confusion counts — human reference / model prediction**")
        st.dataframe([{key: value for key, value in row.items() if key == "Validity" or " / Model " in key}
                      for row in stats["dimensions"]], hide_index=True, width="stretch")
        st.caption("For both checks, Yes counts as Valid and No as Invalid. Partly and Unsure human judgments and missing model flags are excluded from binary metrics per validity axis; Partly is counted separately. Exclusion reasons can overlap. Metrics describe the reviewed stratified sample, rather than an estimate for the full dataset. They use final labels, which may have been edited after seeing model feedback.")
        st.caption(f"Reviews marked as having model feedback displayed before saving: {stats['feedback_seen']}. Older records may not track this.")
        if not stats["reviewed"]:
            st.info("Save a review with Next question to populate agreement, precision, and recall.")


def main() -> None:
    st.set_page_config(page_title="Question labeling", layout="wide")
    stats_panel = st.expander("Dataset statistics and validator agreement", expanded=False)
    st.title("Question labeling")
    st.write("Answer both checks, then use Display validity to reveal the model judgments. You can still edit your choices and notes. Next question saves your final response and advances.")

    with st.sidebar:
        st.header("Labeling session")
        annotator = st.text_input("Annotator ID", value="anonymous", help="Use a stable, distinct ID to resume your own work after restarting.").strip()
        try:
            datasets = discover_datasets(REPO_DIR)
        except OSError as error:
            st.error(f"Cannot read dataset location: {error}")
            return
        if not datasets:
            st.info(f"No metadata.jsonl files found under {REPO_DIR / 'dataset'}.")
            return
        dataset_name = st.selectbox("Dataset", list(datasets), index=None, placeholder="Select a dataset")
        if dataset_name and st.button("Refresh images", help="Rebuild the question index after restoring missing or empty image files. Saved labels are preserved."):
            cached_index.clear()
            cached_chart_image.clear()
            st.session_state.pop("context", None)
            st.rerun()
        st.caption("Sampling uses shuffled blocks with one question per available combination of the model's data and visual flags. Questions do not repeat for this annotator.")
    if not dataset_name or not annotator:
        stats_panel.info("Select a dataset and enter an annotator ID to see its statistics.")
        st.info("Select a dataset and enter an annotator ID to begin.")
        return

    dataset_dir = datasets[dataset_name]
    metadata = dataset_dir / "metadata.jsonl"
    output_dir = Path(os.environ.get("QUESTION_LABEL_OUTPUT_DIR", str(APP_DIR / "responses"))).expanduser().resolve()
    output = response_path(output_dir, dataset_dir, annotator)
    try:
        stat = metadata.stat()
        fingerprint = {"size": stat.st_size, "modified_ns": stat.st_mtime_ns}
        index = cached_index(str(metadata), stat.st_size, stat.st_mtime_ns)
        responses, malformed = read_responses(output)
    except (OSError, ValueError) as error:
        st.error(f"Could not load this labeling session: {error}")
        return
    if malformed:
        st.error(f"The response file has {malformed} malformed rows. Repair {output} before continuing; existing responses will not be overwritten.")
        return

    completed = {row["question_key"] for row in responses}
    render_statistics(stats_panel, index["entries"], responses, annotator, dataset_name)
    context = (str(metadata), stat.st_size, stat.st_mtime_ns, annotator, str(output), "save-on-next", "validated-images-v1", len(index["entries"]))
    if st.session_state.get("context") != context:
        seed = secrets.randbits(64)
        remaining_counts = Counter(entry["stratum"] for entry in index["entries"] if entry["question_key"] not in completed)
        st.session_state.context = context
        st.session_state.run = {
            "session_id": str(uuid4()), "seed": seed,
            "queue": stratified_queue(index["entries"], completed, seed),
            "position": 0, "revealed": False, "presented_at": now(),
            "labels_before_reveal": None, "validity_displayed_at": None,
            "skipped_charts": set(),
            "started": time.monotonic(),
            "stratum_remaining_at_start": dict(remaining_counts),
        }
    run = st.session_state.run
    # Another browser tab may have already saved a queued question.
    while run["position"] < len(run["queue"]) and (
        run["queue"][run["position"]]["question_key"] in completed
        or run["queue"][run["position"]]["offset"] in run.get("skipped_charts", set())
    ):
        run["position"] += 1
        run.update(presented_at=now(), started=time.monotonic(), revealed=False,
                   labels_before_reveal=None, validity_displayed_at=None)
    total = len(index["entries"])
    completed_count = sum(entry["question_key"] in completed for entry in index["entries"])
    with st.sidebar:
        st.metric("Labeled in this dataset", f"{completed_count} / {total}")
        if run.get("skipped_charts"):
            st.caption(f"Unreadable charts skipped this session: {len(run['skipped_charts'])}. Skips are not labels.")
        st.caption(f"Responses: {output}")
        if responses:
            st.download_button("Download responses (JSONL)", output.read_bytes(), file_name=f"{dataset_name}-labels.jsonl", mime="application/x-ndjson")
        if index["warnings"]:
            st.warning("Some records were skipped while indexing.")
            st.json(index["warnings"])
    if run["position"] >= len(run["queue"]):
        if run.get("skipped_charts"):
            st.info("No questions remain in this session. Some charts were skipped because their images could not be read. Restore the images and reload the app to review them.")
        elif total:
            st.success("All available questions in this dataset have been labeled by this annotator.")
        else:
            st.info("This dataset has no questions with usable chart images. Check the indexing warnings in the sidebar.")
        return

    entry = run["queue"][run["position"]]
    try:
        record, question, record_hash = load_question(metadata, entry)
        image, image_path = selected_image(record, dataset_dir)
    except (OSError, ValueError, KeyError, IndexError) as error:
        st.error(str(error))
        return

    try:
        if image_path is None:
            raise FileNotFoundError("The selected chart image is missing or empty.")
        image_stat = image_path.stat()
        chart_image = cached_chart_image(str(image_path), image_stat.st_size, image_stat.st_mtime_ns)
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as error:
        st.error(f"Cannot read the chart image for graph {entry['graph_id']}: {error}")
        st.code(str(image_path or image.get("path", "No image path available")), language=None)
        st.caption("Restore this image and retry, or skip this chart without saving any validity labels.")
        retry, skip = st.columns(2)
        if retry.button("Retry image"):
            st.rerun()
        if skip.button("Skip chart"):
            run.setdefault("skipped_charts", set()).add(entry["offset"])
            run.update(position=run["position"] + 1, revealed=False, labels_before_reveal=None,
                       validity_displayed_at=None, presented_at=now(), started=time.monotonic())
            st.rerun()
        return

    chart_col, question_col = st.columns([3, 2], gap="large")
    with chart_col:
        st.image(chart_image, width="stretch")
        st.caption(f"Graph {entry['graph_id']} · {image_path.name}")
        with st.expander("Open chart at original resolution"):
            st.image(chart_image, width="content")
    with question_col:
        st.subheader("Question")
        st.write(question["question"])
        st.markdown("**Proposed answer**")
        st.write(question.get("answer", "(No answer supplied)"))
        if question.get("explanation"):
            st.markdown("**Explanation to validate**")
            st.write(question["explanation"])
        dataset = record.get("dataset", {})
        if dataset.get("sanitized_description"):
            with st.expander("Allowed dataset context"):
                st.write(dataset["sanitized_description"])
        else:
            st.caption("No sanitized dataset context is available; judge visual validity using the chart and question.")

        with st.expander("Labeling guidelines"):
            st.markdown(
                "**Do the data agree with the chart?** Check the values, comparisons, calculations, and units in the question, answer, and explanation. Choose **Yes** if they agree, **No** if they do not, **Partly** if some claims agree and others do not, or **Unsure** if you cannot tell.\n\n"
                "**Does answering require reading the chart, and does it support the answer and explanation?** Choose **No** if the question is self-contained: its wording or the allowed context already lets you answer without reading the image, even if the answer is correct. Otherwise, check that readable chart evidence and the allowed context support the answer and explanation, without hidden observations or outside specialist knowledge. Reasonable estimates count. Choose **Yes** if all requirements are met, **Partly** if the image is needed but only some claims are supported, or **Unsure** if you cannot tell.\n\n"
                "Judge these independently. Use **Unsure** when the available evidence is insufficient."
            )
        widget_key = f"{run['session_id']}:{entry['question_key']}"
        data_input, data_model = st.columns([3, 2], vertical_alignment="center")
        with data_input:
            data_choice = st.radio(
                "Do the data in the question, answer, and explanation agree with the chart?",
                CHOICES, index=None, horizontal=True, key=f"{widget_key}:data-agreement",
                help="Check values, comparisons, calculations, and units. Partly means some claims agree and others do not; Unsure means there is not enough evidence to decide.",
            )
        with data_model:
            data_feedback = st.empty()
        visual_input, visual_model = st.columns([3, 2], vertical_alignment="center")
        with visual_input:
            visual_choice = st.radio(
                "Does answering require reading the chart, and does the chart support the answer and explanation?",
                CHOICES, index=None, horizontal=True, key=f"{widget_key}:visual-answerability",
                help="Choose No if the question can be answered from its wording or the allowed context alone, without reading the image—even if the answer is correct. The chart must support the answer and explanation without hidden data or outside specialist knowledge. Reasonable estimates count. Partly means the image is needed but only some claims are supported; Unsure means you cannot tell.",
            )
        with visual_model:
            visual_feedback = st.empty()
        with st.expander("Notes (optional)"):
            notes = st.text_area("Notes", key=f"{widget_key}:notes", label_visibility="collapsed")
        ready = data_choice is not None and visual_choice is not None
        reveal_action, next_action = st.columns(2)
        if reveal_action.button("Display validity", disabled=not ready or run["revealed"]):
            run.update(revealed=True, validity_displayed_at=now(),
                       labels_before_reveal={"data_valid": VALUES[data_choice], "visual_valid": VALUES[visual_choice]})
        flags = validity_flags(question)
        for feedback, field in ((data_feedback, "data_valid"), (visual_feedback, "visual_valid")):
            if run["revealed"]:
                color = "green" if flags[field] is True else "red" if flags[field] is False else "gray"
                reason = flags.get("data_reason" if field == "data_valid" else "visual_reason")
                feedback.markdown(f":{color}[**Model: {flag_label(flags[field])}**]", help=reason)
            else:
                feedback.caption("Model: hidden")
        st.caption("Choices stay editable. Next question saves. Hover over a model judgment for its reason.")
        if next_action.button("Next question", type="primary", disabled=not ready):
            response = {
                "schema_version": 4, "annotation_id": str(uuid4()),
                "question_key": entry["question_key"], "annotator_id": annotator,
                "session_id": run["session_id"], "presented_at": run["presented_at"],
                "submitted_at": now(), "duration_seconds": round(time.monotonic() - run["started"], 3),
                "dataset_name": dataset_name, "dataset_dir": str(dataset_dir),
                "source": {"metadata_path": str(metadata), "metadata_fingerprint": fingerprint,
                           "line": entry["line"], "byte_offset": entry["offset"], "byte_length": entry["length"],
                           "record_sha256": record_hash},
                "graph_id": entry["graph_id"], "prefix_id": record.get("prefix_id"),
                "graph_accepted": record.get("accepted"), "source_dataset_id": dataset.get("id"),
                "question_index": entry["question_index"], "question_id": question.get("id", question.get("question_id")),
                "question": question, "dataset": dataset,
                "graph": {key: value for key, value in record["graph"].items() if key != "questions"},
                "image_path": str(image_path), "image_metadata_path": image["path"],
                "image_selected": bool(image.get("selected")),
                "model_labels": validity_flags(question),
                "model_feedback_displayed": run["revealed"],
                "validity_displayed_at": run["validity_displayed_at"],
                "labels_before_reveal": run["labels_before_reveal"],
                "labels_changed_after_reveal": run["revealed"] and run["labels_before_reveal"] != {"data_valid": VALUES[data_choice], "visual_valid": VALUES[visual_choice]},
                "human_labels": {"data_valid": VALUES[data_choice], "visual_valid": VALUES[visual_choice]},
                "human_choices": {"data": data_choice, "visual": visual_choice}, "notes": notes,
                "sampling": {"strategy": "shuffled_joint_stratum_blocks_without_replacement",
                             "stratum": entry["stratum"], "seed": run["seed"], "draw_index": run["position"],
                             "stratum_population": dict(Counter(item["stratum"] for item in index["entries"])),
                             "stratum_remaining_at_start": run["stratum_remaining_at_start"]},
            }
            try:
                save_response(output, response)
            except (OSError, ValueError, TimeoutError) as error:
                st.error(f"Labels were not saved. Stay on this question and retry Next question: {error}")
            else:
                run.update(position=run["position"] + 1, revealed=False, labels_before_reveal=None,
                           validity_displayed_at=None, presented_at=now(), started=time.monotonic())
                st.rerun()


if __name__ == "__main__":
    main()
