"""Dataset indexing, stratified sampling, and durable annotation storage."""
from __future__ import annotations

import errno
import hashlib
import json
import os
import random
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path

from PIL import Image


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


def stratified_queue(entries: list[dict], completed: set[str], seed: int) -> list[dict]:
    """Randomized blocks: one random remaining question per nonempty joint stratum."""
    rng = random.Random(seed)
    pools = defaultdict(list)
    for entry in entries:
        if entry["question_key"] not in completed:
            pools[entry["stratum"]].append(entry)
    for pool in pools.values():
        rng.shuffle(pool)
    queue = []
    while pools:
        strata = list(pools)
        rng.shuffle(strata)
        for stratum in strata:
            queue.append(pools[stratum].pop())
            if not pools[stratum]:
                del pools[stratum]
    return queue


def response_path(output_dir: Path, dataset_dir: Path, annotator: str) -> Path:
    identity = digest([str(dataset_dir.resolve()), annotator])[:20]
    return output_dir / f"labels-{identity}.jsonl"


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


def read_responses(path: Path) -> tuple[list[dict], int]:
    rows = []
    malformed = 0
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict) or not row.get("question_key"):
                        raise ValueError("Invalid annotation row")
                    rows.append(row)
                except ValueError:
                    malformed += 1
    return rows, malformed


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
