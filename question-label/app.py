"""Run from the repository root: python -m streamlit run question-label/app.py."""
from __future__ import annotations

import hashlib
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
    REVIEW_CHOICES, REVIEW_QUESTIONS, derive_human_labels,
    annotation_statistics, discover_datasets, flag_label, index_dataset, load_chart_image, load_question, read_responses,
    next_question_position, random_queue, response_path, save_response, selected_image, validity_flags,
    parse_responses, relabel_entries, relabel_response_path,
)


APP_DIR = Path(__file__).resolve().parent
REPO_DIR = APP_DIR.parent


@st.cache_data(show_spinner="Indexing dataset questions…", max_entries=3)
def cached_index(path: str, size: int, modified_ns: int) -> dict:
    """Index charts with existing, nonempty selected images."""
    return index_dataset(Path(path))


@st.cache_data(show_spinner=False, max_entries=8)
def cached_chart_image(path: str, size: int, modified_ns: int) -> Image.Image:
    return load_chart_image(Path(path))


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def advance_question(run: dict, *, target: int | None = None, action: str = "next") -> None:
    """Advance without dropping intervening questions when staying on a chart."""
    if run.get("relabeling") and action != "next":
        return
    position = run["position"]
    if action in ("skip_question", "same_graph"):
        run.setdefault("skipped_questions", set()).add(run["queue"][position]["question_key"])
    if target is not None and target != position + 1:
        run["queue"].insert(position + 1, run["queue"].pop(target))
    run.update(
        position=position + 1, revealed=False, labels_before_reveal=None, choices_before_reveal=None,
        validity_displayed_at=None, presented_at=now(), started=time.monotonic(),
        selection_action=action,
        sampling_strategy=("source_file_order" if run.get("relabeling")
                           else "uniform_random_same_graph_without_replacement" if action == "same_graph"
                           else "uniform_random_without_replacement"),
    )


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
        st.caption("Metrics compare the saved human validity flags with the model flags. In seven-question reviews, Yes fails questions 1 and 7; No fails questions 3–6. Question 2 does not affect flags. Partly and Unsure human flags and missing model flags are excluded from binary metrics per axis; Partly is counted separately. Exclusion reasons can overlap. Metrics include earlier reviews and use final labels, which may have been edited after seeing model feedback.")
        st.caption(f"Reviews marked as having model feedback displayed before saving: {stats['feedback_seen']}. Older records may not track this.")
        if not stats["reviewed"]:
            st.info("Save a review with Next question to populate agreement, precision, and recall.")


def main() -> None:
    st.set_page_config(page_title="Question labeling", layout="wide")
    stats_panel = st.expander("Dataset statistics and validator agreement", expanded=False)
    st.title("Question labeling")
    st.write("Answer all seven questions to preview the flags that will be saved. Display validity reveals the model judgments. Your choices and notes stay editable. Next question saves your final response and advances.")
    output_dir = Path(os.environ.get("QUESTION_LABEL_OUTPUT_DIR", str(APP_DIR / "responses"))).expanduser().resolve()
    source_path = None
    uploaded_source = None

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
        relabeling = st.selectbox("Mode", ["Label dataset", "Relabel a response file"]) == "Relabel a response file"
        if relabeling:
            try:
                source_files = sorted(output_dir.glob("*.jsonl"))
            except OSError as error:
                st.error(f"Cannot list response files: {error}")
                return
            source_path = st.selectbox("Source response file", source_files, index=None,
                                       format_func=lambda path: path.name, placeholder="Select a labeler's file",
                                       key="relabel_source_file")
            uploaded_source = st.file_uploader("Or upload a response file", type=["jsonl"],
                                              help="An uploaded file takes precedence over the selected server file.")
            st.caption("Review only the questions in this file, in file order. Previous labels stay hidden. Skipping is disabled. New responses are saved separately.")
        if dataset_name and st.button("Refresh images", help="Rebuild the question index after restoring missing or empty image files. Saved labels are preserved."):
            cached_index.clear()
            cached_chart_image.clear()
            st.session_state.pop("context", None)
            st.rerun()
        if not relabeling:
            st.caption("Questions start in random order. Skip a question or choose another from the same graph without saving. Saved reviews do not repeat for this annotator.")
    if not dataset_name or not annotator:
        stats_panel.info("Select a dataset and enter an annotator ID to see its statistics.")
        st.info("Select a dataset and enter an annotator ID to begin.")
        return

    dataset_dir = datasets[dataset_name]
    metadata = dataset_dir / "metadata.jsonl"
    output = response_path(output_dir, dataset_dir, annotator)
    source_sha256 = None
    source_name = None
    source_by_key = {}
    try:
        stat = metadata.stat()
        fingerprint = {"size": stat.st_size, "modified_ns": stat.st_mtime_ns}
        index = cached_index(str(metadata), stat.st_size, stat.st_mtime_ns)
        entries = index["entries"]
        if relabeling:
            if uploaded_source is None and source_path is None:
                st.info("Select or upload a labeler's response file to begin relabeling.")
                return
            source_payload = uploaded_source.getvalue() if uploaded_source is not None else source_path.read_bytes()
            source_name = uploaded_source.name if uploaded_source is not None else source_path.name
            source_sha256 = hashlib.sha256(source_payload).hexdigest()
            source_rows, source_malformed = parse_responses(source_payload)
            if source_malformed:
                raise ValueError(f"The source response file has {source_malformed} malformed rows. Repair it before relabeling.")
            entries = relabel_entries(metadata, entries, source_rows)
            source_by_key = {row["question_key"]: row for row in reversed(source_rows)}
            output = relabel_response_path(output_dir, dataset_dir, annotator, source_sha256)
            if uploaded_source is None and (output.resolve() == source_path.resolve()
                                            or (output.exists() and output.samefile(source_path))):
                raise ValueError("The relabeling output must differ from the source response file.")
        responses, malformed = read_responses(output)
    except (OSError, ValueError) as error:
        st.error(f"Could not load this labeling session: {error}")
        return
    if malformed:
        st.error(f"The response file has {malformed} malformed rows. Repair {output} before continuing; existing responses will not be overwritten.")
        return

    completed = {row["question_key"] for row in responses}
    render_statistics(stats_panel, entries, responses, annotator, dataset_name)
    if relabeling:
        stats_panel.caption("Relabeling statistics cover only the questions in the selected source file and this annotator's new responses.")
    context = (str(metadata), stat.st_size, stat.st_mtime_ns, annotator, str(output), "seven-question-review-v1", "validated-images-v1", "uniform-random-v1", len(entries), relabeling, source_sha256)
    if st.session_state.get("context") != context:
        seed = secrets.randbits(64)
        remaining_counts = Counter(entry["stratum"] for entry in entries if entry["question_key"] not in completed)
        st.session_state.context = context
        st.session_state.run = {
            "session_id": str(uuid4()), "seed": seed,
            "queue": ([entry for entry in entries if entry["question_key"] not in completed]
                      if relabeling else random_queue(entries, completed, seed)),
            "relabeling": relabeling,
            "position": 0, "revealed": False, "presented_at": now(),
            "labels_before_reveal": None, "choices_before_reveal": None, "validity_displayed_at": None,
            "skipped_charts": set(), "skipped_questions": set(),
            "selection_action": "initial", "sampling_strategy": "source_file_order" if relabeling else "uniform_random_without_replacement",
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
                   labels_before_reveal=None, choices_before_reveal=None, validity_displayed_at=None,
                   selection_action="queue", sampling_strategy="source_file_order" if relabeling else "uniform_random_without_replacement")
    total = len(entries)
    completed_count = sum(entry["question_key"] in completed for entry in entries)
    with st.sidebar:
        st.metric("Relabeled from source file" if relabeling else "Labeled in this dataset", f"{completed_count} / {total}")
        if relabeling:
            st.caption(f"Source: {source_name} · {total} unique questions")
        if run.get("skipped_charts"):
            st.caption(f"Unreadable charts skipped this session: {len(run['skipped_charts'])}. Skips are not labels.")
        if run.get("skipped_questions"):
            st.caption(f"Questions skipped this session: {len(run['skipped_questions'])}. Skips are not labels.")
        st.caption(f"Responses: {output}")
        if responses:
            st.download_button("Download responses (JSONL)", output.read_bytes(), file_name=f"{dataset_name}-{'relabels' if relabeling else 'labels'}.jsonl", mime="application/x-ndjson")
        if index["warnings"]:
            st.warning("Some records were skipped while indexing.")
            st.json(index["warnings"])
    if run["position"] >= len(run["queue"]):
        if run.get("skipped_charts"):
            st.info("No questions remain in this session. Some charts were skipped because their images could not be read. Restore the images and reload the app to review them.")
        elif run.get("skipped_questions"):
            st.info("No questions remain in this session. Some questions were skipped without saving labels. Reload the app to review them again.")
        elif total:
            st.success("All questions in the source file have been relabeled by this annotator." if relabeling
                       else "All available questions in this dataset have been labeled by this annotator.")
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
        st.caption("Restore this image and retry. Skipping is disabled during relabeling." if relabeling
                   else "Restore this image and retry, or skip this chart without saving any validity labels.")
        retry, skip = st.columns(2)
        if retry.button("Retry image"):
            st.rerun()
        if skip.button("Skip chart", disabled=relabeling):
            run.setdefault("skipped_charts", set()).add(entry["offset"])
            advance_question(run, action="skip_chart")
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

        random_target = next_question_position(
            run["queue"], run["position"], completed, run.get("skipped_charts", set()),
        )
        graph_target = next_question_position(
            run["queue"], run["position"], completed, run.get("skipped_charts", set()), same_graph=True,
        )
        skip_question, same_graph = st.columns(2)
        if skip_question.button(
            "Skip question", disabled=relabeling or random_target is None,
            help=("Skipping is disabled during relabeling." if relabeling else
                  "Skip without saving and show the next random question. Skipped questions return after reloading."
                  if random_target is not None else "No other unreviewed questions remain in this session."),
        ):
            advance_question(run, target=random_target, action="skip_question")
            st.rerun()
        if same_graph.button(
            "Different question from same graph", disabled=relabeling or graph_target is None,
            help=("Changing questions without saving is disabled during relabeling." if relabeling else
                  "Skip without saving and show another random question from this chart."
                  if graph_target is not None else "No other unreviewed questions from this chart remain in this session."),
        ):
            advance_question(run, target=graph_target, action="same_graph")
            st.rerun()

        with st.expander("Labeling guidelines"):
            st.markdown(
                "Answer each question literally. **Yes** to question 1 (answer leakage) or 7 (required outside knowledge) fails that visual check. **No** to question 3 (readability) or 4 (chart necessity) also fails visual validity. **No** to question 5 (answer correctness) or 6 (values in the question) fails data validity.\n\n"
                "Any definite failure makes its flag **false**, even if another check is Partly or Unsure. Otherwise, any **Partly** yields `partly`, then any **Unsure** yields `null`; all passing checks yield **true**. Rate question 2 from **1** (does not support the task) to **5** (fully supports it); it is saved separately and does not affect flags.\n\n"
                "If the question contains no factual values to check, answer **Yes** to question 6. Judge the answer and explanation together in question 5. Use **Unsure** when evidence is insufficient. Hover over each question's help icon for detailed guidance and examples."
            )
        widget_key = f"{run['session_id']}:{entry['question_key']}"
        choices = {}
        review_tabs = st.tabs([f"Q{number}" for number in range(1, len(REVIEW_QUESTIONS) + 1)])
        for tab, item in zip(review_tabs, REVIEW_QUESTIONS):
            with tab:
                choices[item["key"]] = st.radio(
                    item["label"], REVIEW_CHOICES if "passing_choice" in item else [1, 2, 3, 4, 5],
                    index=None, horizontal=True, key=f"{widget_key}:{item['key']}", help=item["help"],
                )
        unanswered = [f"Q{number}" for number, choice in enumerate(choices.values(), start=1) if choice is None]
        st.caption(f"Answered {len(choices) - len(unanswered)} of {len(choices)}" +
                   (f" · Remaining: {', '.join(unanswered)}" if unanswered else " · All questions answered"))
        with st.expander("Notes (optional)"):
            notes = st.text_area("Notes", key=f"{widget_key}:notes", label_visibility="collapsed")
        ready = all(choice is not None for choice in choices.values())
        human_labels = derive_human_labels(choices)
        st.markdown("**Validity flags**")
        human_col, model_col = st.columns(2)
        human_feedback = human_col.empty()
        model_feedback = model_col.empty()
        reveal_action, next_action = st.columns(2)
        if reveal_action.button("Display validity", disabled=not ready or run["revealed"]):
            run.update(revealed=True, validity_displayed_at=now(),
                       labels_before_reveal=human_labels.copy(), choices_before_reveal=choices.copy())
        flags = validity_flags(question)
        with human_feedback.container():
            st.markdown("**Your flags to save**")
            if ready:
                for field in ("data_valid", "visual_valid"):
                    value = human_labels[field]
                    serialized = "true" if value is True else "false" if value is False else '"partly"' if value == "partly" else "null (Unsure)"
                    color = "green" if value is True else "red" if value is False else "orange" if value == "partly" else "gray"
                    st.markdown(f"{field}: :{color}[**{serialized}**]")
            else:
                st.caption("Answer all seven questions to preview your flags.")
        with model_feedback.container():
            st.markdown("**LLM flags**")
            for field in ("data_valid", "visual_valid"):
                if run["revealed"]:
                    color = "green" if flags[field] is True else "red" if flags[field] is False else "gray"
                    reason = flags.get("data_reason" if field == "data_valid" else "visual_reason")
                    st.markdown(f"{field}: :{color}[**Model: {flag_label(flags[field])}**]", help=reason)
                else:
                    st.caption(f"{field}: Model hidden")
        st.caption("Choices stay editable. Next question saves. Hover over a model judgment for its reason.")
        if next_action.button("Next question", type="primary", disabled=not ready):
            response = {
                "schema_version": 5, "annotation_id": str(uuid4()),
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
                "choices_before_reveal": run["choices_before_reveal"],
                "labels_changed_after_reveal": run["revealed"] and run["labels_before_reveal"] != human_labels,
                "choices_changed_after_reveal": run["revealed"] and run["choices_before_reveal"] != choices,
                "human_labels": human_labels,
                "human_choices": choices, "notes": notes,
                "sampling": {"strategy": run.get("sampling_strategy", "uniform_random_without_replacement"),
                             "selection_action": run.get("selection_action", "initial"),
                             "skipped_questions_in_session": len(run.get("skipped_questions", set())),
                             "stratum": entry["stratum"], "seed": run["seed"], "draw_index": run["position"],
                             "stratum_population": dict(Counter(item["stratum"] for item in entries)),
                             "stratum_remaining_at_start": run["stratum_remaining_at_start"]},
            }
            if relabeling:
                original = source_by_key[entry["source_question_key"]]
                response["relabel_source"] = {
                    "file_name": source_name, "file_sha256": source_sha256,
                    "question_key": original["question_key"],
                    "annotation_id": original.get("annotation_id"),
                    "annotator_id": original.get("annotator_id"),
                }
            try:
                save_response(output, response)
            except (OSError, ValueError, TimeoutError) as error:
                st.error(f"Labels were not saved. Stay on this question and retry Next question: {error}")
            else:
                advance_question(run)
                st.rerun()


if __name__ == "__main__":
    main()
