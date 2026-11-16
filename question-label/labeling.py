"""Dataset indexing, random sampling, and durable annotation storage."""
from __future__ import annotations

import errno
import hashlib
import json
import os
import random
import time
from collections import Counter
from contextlib import contextmanager
from io import BytesIO, StringIO
from pathlib import Path

from PIL import Image


REVIEW_CHOICES = ["Yes", "No", "Unsure", "Partly"]
# A passing answer depends on the wording: answer leakage and outside knowledge
# are undesirable, whereas readability, necessity, and correctness are required.
REVIEW_QUESTIONS = (
    {
        "key": "answer_in_question",
        "label": "1. Is the answer already contained in the question?",
        "passing_choice": "No",
        "help": (
            "Check whether the question states, paraphrases, or effectively gives away the requested answer. "
            "Try answering with the chart covered. Supplying the plotted values or intermediate results needed "
            "to calculate the answer also counts as leakage. For an axis-label question, copying the requested "
            "axis text into the wording gives it away. Definitions or background rules are allowed when the "
            "answer still requires reading the chart. Yes means the answer is contained in the question and "
            "fails this visual check; No means it is not. Partly means only part of the answer is supplied; "
            "Unsure means you cannot tell."
        ),
    },
    {
        "key": "chart_design_support",
        "label": "2. Does the chart type (design) support the type of question being asked?",
        "help": (
            "Rate how well the chart's design supports the requested task, independently of whether the "
            "proposed answer is correct. For a comparison between groups, the chart must actually display "
            "those groups and encode their values in a way that allows comparison. Consider whether its "
            "marks, axes, scales, grouping, and legend support the requested comparison, trend, distribution, "
            "or relationship. 1 = does not support the task; 2 = weak support with major limitations; "
            "3 = moderate support with noticeable limitations; 4 = good support with minor limitations; "
            "5 = fully supports the task. This rating is saved for analysis and does not change either validity flag."
        ),
    },
    {
        "key": "answer_readable",
        "label": "3. Can the answer actually be read from the chart?",
        "passing_choice": "Yes",
        "help": (
            "Judge whether an LLM given this chart image, the question, and the allowed sanitized dataset "
            "context should be able to read enough evidence to determine the answer. The relevant marks, "
            "categories, labels, and series must be distinguishable at the supplied image resolution. Simple "
            "arithmetic on readable values is allowed. Reasonable visual estimates are sufficient: extra "
            "decimal places in the reference answer or explanation do not fail this check if its approximate "
            "value and reasoning are recoverable from the axis, ticks, and marks. Respect units and axis "
            "multipliers. Choose No when the requested result requires hidden observations, an unreadable "
            "series, or precision the image cannot support. Assess readability here and proposed-answer "
            "correctness in question 5. Partly means only some required evidence is readable; Unsure means "
            "you cannot establish readability."
        ),
    },
    {
        "key": "chart_necessary",
        "label": "4. Is the chart even necessary to answer the question?",
        "passing_choice": "Yes",
        "help": (
            "Hide the chart and consider only the question, the allowed sanitized dataset context, and "
            "general knowledge. Could the question still be answered? Choose No if the answer is already "
            "provided in that text, follows from general knowledge alone, or can be calculated entirely "
            "from values supplied in the wording. A factually correct answer still fails this check if no "
            "chart reading is needed. Choose Yes when at least one necessary observation must come from "
            "the chart. The proposed answer and explanation are claims to evaluate, not extra evidence "
            "for this no-image check. Partly means only part of a multi-part question needs the chart; "
            "Unsure means you cannot decide."
        ),
    },
    {
        "key": "answer_correct",
        "label": "5. Is the answer correct?",
        "passing_choice": "Yes",
        "help": (
            "Verify the proposed answer and its supporting explanation against the available chart evidence "
            "and allowed context. Check every data-dependent claim, not just the final conclusion: recompute "
            "arithmetic, comparisons, differences, ratios, denominators, and threshold rules where possible. "
            "Respect units, axis scaling, processed quantities, and stated approximations; 'most' means more "
            "than half. A correct final answer does not excuse an incorrect or contradictory explanation. "
            "Reasonable estimates consistent with the chart are acceptable. Do not assume hidden raw data "
            "or treat the model's answer as evidence. Yes means the answer and supporting claims are correct; "
            "No means they are incorrect; Partly means some claims are correct and others are not. Use "
            "Unsure when the available evidence cannot establish correctness."
        ),
    },
    {
        "key": "question_value_correct",
        "label": "6. If a value is provided in the question, is that value correct?",
        "passing_choice": "Yes",
        "help": (
            "Check every value asserted as a chart or dataset fact in the question itself, including "
            "counts, percentages, comparisons, units, and scale factors. Distinguish a factual claim "
            "('group A has 20 observations') from a hypothetical rule or comparison threshold "
            "('which groups exceed 20?'); the latter need not be an observed chart value. Allow stated "
            "approximations supported by the chart. Choose Yes if all asserted values are correct, or "
            "if the question contains no factual values to verify. Choose No for incorrect values, "
            "Partly for a mixture of correct and incorrect values, or Unsure if the available evidence "
            "cannot verify them. Correct values may still leak the answer; assess that separately in question 1."
        ),
    },
    {
        "key": "external_knowledge_required",
        "label": "7. Does answering the question require non-general external knowledge?",
        "passing_choice": "No",
        "help": (
            "Identify every factual premise or interpretation rule needed to answer. Ordinary language, "
            "chart reading, arithmetic, logic, and everyday meanings are allowed: higher temperatures "
            "indicate warmer conditions, and scoring more goals than an opponent means winning. "
            "Specialist knowledge, such as what PCA components mean or how to interpret them, is "
            "non-general unless the necessary definitions and rules are supplied in the chart, question, "
            "or sanitized dataset context. Other examples include unstated clinical thresholds, category-code "
            "mappings, causal mechanisms, and scientific explanations. Choose Yes if such outside knowledge "
            "is required, even if you personally know it; No if all necessary premises are supplied or "
            "general. A supplied definition is allowed when chart reading is still necessary. Assertions "
            "in the proposed answer cannot fill missing context. Partly means only some requested parts "
            "need outside knowledge; Unsure means you cannot tell."
        ),
    },
)


def derive_human_labels(choices: dict) -> dict:
    """Aggregate criterion validity; a definite failure wins over uncertainty.

    Keep Partly and Unsure distinct as in earlier two-check reviews. The design
    rating is deliberately independent of both flags.
    """
    def combine(keys: tuple[str, ...]):
        results = []
        for item in REVIEW_QUESTIONS:
            if item["key"] not in keys:
                continue
            choice = choices.get(item["key"])
            if choice in (None, "Unsure"):
                results.append(None)
            elif choice == "Partly":
                results.append("partly")
            elif choice in ("Yes", "No"):
                results.append(choice == item["passing_choice"])
            else:
                raise ValueError(f"Unknown review choice: {choice!r}")
        if False in results:
            return False
        if "partly" in results:
            return "partly"
        if None in results:
            return None
        return True

    return {
        "data_valid": combine(("answer_correct", "question_value_correct")),
        "visual_valid": combine(("answer_in_question", "answer_readable", "chart_necessary", "external_knowledge_required")),
    }


@contextmanager
def response_lock(path: Path):
    """OS-level lock, released on process exit, on Windows and POSIX."""
    with path.open("a+b") as handle:
        if handle.seek(0, os.SEEK_END) == 0:
            handle.write(b"\0")
            handle.flush()
        deadline = time.monotonic() + 10
        while True:
            try:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as error:
                if error.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                    raise
                if time.monotonic() >= deadline:
                    raise TimeoutError("Another session is saving responses. Please retry.") from error
                time.sleep(0.05)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def discover_datasets(root: Path) -> dict[str, Path]:
    root = root.expanduser().resolve()
    base = root / "dataset" if (root / "dataset").is_dir() else root
    found = {base.name: base} if (base / "metadata.jsonl").is_file() else {}
    if base.is_dir():
        found.update({p.name: p for p in sorted(base.iterdir()) if p.is_dir() and (p / "metadata.jsonl").is_file()})
    return found


def validity_flags(question: dict) -> dict:
    visual_field = next((key for key in ("vlisual_valid", "visual_valid", "valid") if key in question), None)
    visual = question.get(visual_field)
    data = question.get("data_valid")
    return {
        "data_valid": data if isinstance(data, bool) else None,
        "visual_valid": visual if isinstance(visual, bool) else None,
        "visual_field": visual_field,
        "data_reason": question.get("data_reason"),
        "visual_reason": question.get("visual_reason"),
    }


def flag_label(value: bool | None) -> str:
    return "Valid" if value is True else "Invalid" if value is False else "Not evaluated"


def stratum_name(flags: dict) -> str:
    return f"data={flag_label(flags['data_valid'])}; visual={flag_label(flags['visual_valid'])}"


def selected_image(record: dict, dataset_dir: Path) -> tuple[dict, Path | None]:
    images = [image for image in record.get("images", []) if isinstance(image, dict) and image.get("path")]
    if not images:
        return {}, None

    def iteration(image: dict) -> int:
        try:
            return int(Path(image["path"]).stem.rsplit("_it", 1)[1])
        except (ValueError, IndexError):
            return -1

    image = next((image for image in images if image.get("selected")), max(images, key=iteration))
    stored = Path(image["path"])
    candidates = [stored] if stored.is_absolute() else [dataset_dir / stored, dataset_dir / "images" / stored.name]
    return image, next((path.resolve() for path in candidates if path.is_file() and path.stat().st_size > 0), None)


def load_chart_image(path: Path) -> Image.Image:
    """Validate and decode the same image bytes that will be displayed."""
    payload = path.read_bytes()
    with Image.open(BytesIO(payload)) as check:
        check.verify()
    with Image.open(BytesIO(payload)) as image:
        image.load()
        return image.copy()


def index_dataset(metadata: Path) -> dict:
    """Keep small question references in memory, never the multi-GB LLM traces."""
    entries = []
    warnings = Counter()
    seen = set()
    metadata_path = str(metadata.resolve())
    with metadata.open("rb") as handle:
        line_number = 0
        while True:
            offset = handle.tell()
            raw = handle.readline()
            if not raw:
                break
            line_number += 1
            if not raw.strip():
                continue
            try:
                record = json.loads(raw)
            except (ValueError, UnicodeDecodeError):
                warnings["Malformed metadata rows"] += 1
                continue
            if not isinstance(record, dict) or not isinstance(record.get("graph"), dict):
                warnings["Rows without a graph"] += 1
                continue
            _, image_path = selected_image(record, metadata.parent)
            if image_path is None:
                warnings["Graphs with missing or empty selected/final images"] += 1
                continue
            questions = record["graph"].get("questions", [])
            if not isinstance(questions, list):
                warnings["Graphs without a question list"] += 1
                continue
            graph_id = record.get("id") or record.get("graph_id") or record.get("chart_id") or record.get("prefix_id")
            for question_index, question in enumerate(questions):
                if not isinstance(question, dict) or not question.get("question"):
                    warnings["Questions without text"] += 1
                    continue
                key = digest([metadata_path, graph_id, question_index, question])
                if key in seen:
                    warnings["Duplicate questions"] += 1
                    continue
                seen.add(key)
                entries.append({
                    "question_key": key,
                    "graph_id": graph_id,
                    "question_index": question_index,
                    "offset": offset,
                    "length": len(raw),
                    "line": line_number,
                    "stratum": stratum_name(validity_flags(question)),
                })
    return {"entries": entries, "warnings": dict(warnings)}


def load_question(metadata: Path, entry: dict) -> tuple[dict, dict, str]:
    with metadata.open("rb") as handle:
        handle.seek(entry["offset"])
        raw = handle.read(entry["length"])
    record = json.loads(raw)
    question = record["graph"]["questions"][entry["question_index"]]
    expected_key = digest([str(metadata.resolve()), entry["graph_id"], entry["question_index"], question])
    if expected_key != entry["question_key"]:
        raise ValueError("Metadata changed during labeling. Reload the dataset before continuing.")
    return record, question, hashlib.sha256(raw).hexdigest()


def random_queue(entries: list[dict], completed: set[str], seed: int) -> list[dict]:
    """Uniformly shuffle all remaining questions, without replacement or stratification."""
    queue = [entry for entry in entries if entry["question_key"] not in completed]
    random.Random(seed).shuffle(queue)
    return queue


def next_question_position(
    queue: list[dict], position: int, completed: set[str], skipped_charts: set[int],
    *, same_graph: bool = False,
) -> int | None:
    """Find the next eligible question in the shuffled queue, optionally on this chart.

    Use the metadata row offset to identify a chart even when graph IDs are
    missing or reused. Consumed questions stay behind the current position.
    """
    current_offset = queue[position]["offset"]
    for candidate_position in range(position + 1, len(queue)):
        candidate = queue[candidate_position]
        if candidate["question_key"] in completed or candidate["offset"] in skipped_charts:
            continue
        if not same_graph or candidate["offset"] == current_offset:
            return candidate_position
    return None


def response_path(output_dir: Path, dataset_dir: Path, annotator: str) -> Path:
    identity = digest([str(dataset_dir.resolve()), annotator])[:20]
    return output_dir / f"labels-{identity}.jsonl"


def relabel_response_path(output_dir: Path, dataset_dir: Path, annotator: str, source_sha256: str) -> Path:
    """Keep repeat reviews separate from normal labels and other source files."""
    identity = digest([str(dataset_dir.resolve()), annotator, source_sha256])[:20]
    return output_dir / f"relabels-{identity}.jsonl"


def relabel_entries(metadata: Path, entries: list[dict], source_rows: list[dict]) -> list[dict]:
    """Select exactly the source questions, once each and in source file order.

    Refuse partial matches: changed questions or unavailable images must not
    silently reduce the set a second annotator is asked to review.
    """
    if not source_rows:
        raise ValueError("The source response file contains no questions.")
    available = {entry["question_key"]: entry for entry in entries}
    selected = {}
    missing = set()
    for row in source_rows:
        key = row["question_key"]
        if key not in available and isinstance(row.get("question"), dict) and "question_index" in row and "graph_id" in row:
            # Question keys include an absolute metadata path. Recompute it for
            # uploads from another machine, requiring the exact same snapshot.
            key = digest([str(metadata.resolve()), row["graph_id"], row["question_index"], row["question"]])
        if key not in available:
            missing.add(row["question_key"])
        elif key not in selected:
            selected[key] = {**available[key], "source_question_key": row["question_key"]}
    if missing:
        raise ValueError(
            f"{len(missing)} source questions are unavailable in the selected dataset. "
            "Select the same dataset used by the original labeler, keep its metadata unchanged, "
            "and restore any missing images before relabeling."
        )
    return list(selected.values())


def annotation_statistics(entries: list[dict], responses: list[dict]) -> dict:
    """Evaluate model flags against final saved human labels, per validity axis."""
    keys = {entry["question_key"] for entry in entries}
    saved = {row["question_key"]: row for row in responses if row["question_key"] in keys}

    def ratio(numerator: int, denominator: int) -> float | None:
        return numerator / denominator if denominator else None

    dimensions = []
    classes = []
    for name, field in (("Data validity", "data_valid"), ("Visual / semantic validity", "visual_valid")):
        counts = Counter()
        human_unsure = human_partly = model_missing = 0
        for row in saved.values():
            human = row.get("human_labels", {}).get(field)
            model = row.get("model_labels", {}).get(field)
            human_partly += human == "partly"
            human_unsure += not isinstance(human, bool) and human != "partly"
            model_missing += not isinstance(model, bool)
            if not isinstance(human, bool) or not isinstance(model, bool):
                continue
            counts[(human, model)] += 1
        tp, tn = counts[(True, True)], counts[(False, False)]
        fp, fn = counts[(False, True)], counts[(True, False)]
        n = tp + tn + fp + fn
        dimensions.append({
            "Validity": name, "Comparable pairs": n,
            "Agreement": ratio(tp + tn, n),
            "Excluded pairs": len(saved) - n,
            "Human partly": human_partly,
            "Human unsure / missing": human_unsure, "Model not evaluated": model_missing,
            "Human valid / Model valid": tp, "Human invalid / Model invalid": tn,
            "Human invalid / Model valid": fp, "Human valid / Model invalid": fn,
        })
        for positive, true_positive, false_positive, false_negative in (("Valid", tp, fp, fn), ("Invalid", tn, fn, fp)):
            classes.append({
                "Validity": name, "Positive class": positive,
                "Precision": ratio(true_positive, true_positive + false_positive),
                "Recall": ratio(true_positive, true_positive + false_negative),
                "F1": ratio(2 * true_positive, 2 * true_positive + false_positive + false_negative),
                "Human support": true_positive + false_negative,
                "Model predictions": true_positive + false_positive,
            })
    return {
        "total": len(entries), "reviewed": len(saved),
        "graphs": len({entry["offset"] for entry in entries}),
        "dimensions": dimensions, "classes": classes,
        "feedback_seen": sum(row.get("model_feedback_displayed") is True for row in saved.values()),
        "strata": [{"Model validity combination": stratum, "Questions": count}
                   for stratum, count in sorted(Counter(entry["stratum"] for entry in entries).items())],
    }


def parse_responses(payload: bytes) -> tuple[list[dict], int]:
    """Read server-side or uploaded UTF-8 JSONL responses without modifying them."""
    rows = []
    malformed = 0
    for line in StringIO(payload.decode("utf-8-sig")):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("question_key"), str) or not row["question_key"]:
                raise ValueError("Invalid annotation row")
            rows.append(row)
        except ValueError:
            malformed += 1
    return rows, malformed


def read_responses(path: Path) -> tuple[list[dict], int]:
    return parse_responses(path.read_bytes()) if path.exists() else ([], 0)


def save_response(path: Path, response: dict) -> dict:
    """Serialize concurrent submissions, deduplicate, and flush before advancing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with response_lock(Path(str(path) + ".lock")):
        rows, malformed = read_responses(path)
        if malformed:
            raise ValueError(f"Response file contains {malformed} malformed rows; repair it before saving.")
        existing = next((row for row in rows if row["question_key"] == response["question_key"]), None)
        if existing is not None:
            return existing
        # Also accept a valid final JSON object that lacks its terminating newline.
        with path.open("a+b") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            if size:
                handle.seek(-1, os.SEEK_END)
                if handle.read(1) != b"\n":
                    handle.write(b"\n")
            handle.write((json.dumps(response, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
    return response
