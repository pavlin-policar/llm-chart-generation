"""Prepare static per-chart files for deploying the dataset viewer.

The output directory is intended to be uploaded as-is to an HTTP(S)-reachable
FTP/web root. It contains a small global manifest plus one directory per chart.
Each chart directory contains only the metadata, images, and model-result
records needed for that chart.

Example:
    python prepare_ftp_bundle.py \
        --data-root ../../.. \
        --output /tmp/chart-viewer-ftp \
        --base-url https://example.com/chart-viewer-data/
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image


CANONICAL_TYPE_MAP = {
    "bar chart": "Bar Chart",
    "bar plot": "Bar Chart",
    "grouped bar chart": "Bar Chart",
    "stacked bar chart": "Bar Chart",
    "horizontal bar chart": "Bar Chart",
    "line chart": "Line Plot",
    "line plot": "Line Plot",
    "area chart": "Area Plot",
    "area plot": "Area Plot",
    "scatter plot": "Scatter Plot",
    "scatter chart": "Scatter Plot",
    "bubble chart": "Bubble Chart",
    "bubble plot": "Bubble Chart",
    "histogram": "Histogram",
    "density plot": "Density Plot",
    "kde plot": "Density Plot",
    "box plot": "Box / Violin",
    "boxplot": "Box / Violin",
    "violin plot": "Box / Violin",
    "heatmap": "Heatmap",
    "correlation heatmap": "Heatmap",
    "hexbin plot": "Hexbin",
    "hexbin": "Hexbin",
    "scatterplot matrix": "Scatter Matrix",
    "scatter matrix": "Scatter Matrix",
    "pair plot": "Scatter Matrix",
    "pairplot": "Scatter Matrix",
    "parallel coordinates": "Parallel Coordinates",
    "radar chart": "Radar Chart",
    "spider chart": "Radar Chart",
    "pie chart": "Pie Chart",
    "donut chart": "Pie Chart",
    "treemap": "Treemap",
    "ecdf plot": "ECDF / Q-Q",
    "q-q plot": "ECDF / Q-Q",
    "qq plot": "ECDF / Q-Q",
    "strip plot": "Categorical Scatter",
    "swarm plot": "Categorical Scatter",
    "jitter plot": "Categorical Scatter",
    "dot plot": "Categorical Scatter",
    "error bar plot": "Error Bar",
    "error bar chart": "Error Bar",
    "faceted plot": "Faceted Plot",
    "facet grid": "Faceted Plot",
    "projection plot": "Projection Plot",
    "pca plot": "Projection Plot",
    "sequence logo": "Sequence Logo",
    "3d surface": "3D Surface",
    "3d surface plot": "3D Surface",
}


def canonicalize_chart_type(raw_type: str) -> str:
    key = str(raw_type or "").strip().lower()
    return CANONICAL_TYPE_MAP.get(key, str(raw_type or "Unknown").strip() or "Unknown")


def parse_shuffle_seed(value: str) -> int | None:
    if value.lower() in {"none", "metadata", "off"}:
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("shuffle seed must be an integer or 'none'") from exc


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_no}: {exc}") from exc


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            f.write("\n")


def write_jsonl_gz(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=9) as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            f.write("\n")


def dataset_label(description: str) -> str:
    first = (description or "").strip().split(". ", 1)[0].strip()
    return first[:120] if first else "(no description)"


def image_iteration(path_str: str) -> int:
    stem = Path(path_str).stem
    if "_it" not in stem:
        return 0
    try:
        return int(stem.rsplit("_it", 1)[1])
    except ValueError:
        return 0


def final_image_path(record: dict[str, Any]) -> str | None:
    images = [
        img for img in record.get("images", [])
        if isinstance(img, dict) and isinstance(img.get("path"), str)
    ]
    if not images:
        return None
    return max(images, key=lambda img: image_iteration(img["path"]))["path"]


def quality(record: dict[str, Any], max_rounds: int = 3) -> str:
    images = [
        img for img in record.get("images", [])
        if isinstance(img, dict) and isinstance(img.get("path"), str)
    ]
    if not images:
        return "good"
    last = max(images, key=lambda img: image_iteration(img["path"]))
    if image_iteration(last["path"]) < max_rounds:
        return "good"
    feedback = last.get("feedback") or ""
    if isinstance(feedback, list):
        feedback = " ".join(str(item).strip() for item in feedback if str(item).strip())
    return "bad" if str(feedback).strip() else "good"


def chart_accepted(record: dict[str, Any]) -> bool:
    """Return the chart-level acceptance flag, with an iteration fallback."""
    accepted = record.get("accepted")
    if isinstance(accepted, bool):
        return accepted
    images = [
        img for img in record.get("images", [])
        if isinstance(img, dict) and isinstance(img.get("path"), str)
    ]
    if not images:
        return False
    return bool(max(images, key=lambda img: image_iteration(img["path"])).get("accept"))

def read_error_counts(source_dir: Path) -> dict[str, int]:
    """Load generation-stage error counts from JSON or JSONL error logs."""
    counts: Counter = Counter()
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

    rows: list[dict[str, Any]] = []
    if error_file.suffix == ".jsonl":
        rows = [row for row in read_jsonl(error_file) if isinstance(row, dict)]
    else:
        try:
            payload = json.loads(error_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {error_file}: {exc}") from exc
        if isinstance(payload, list):
            rows = [row for row in payload if isinstance(row, dict)]
        elif isinstance(payload, dict):
            nested = payload.get("errors")
            rows = (
                [row for row in nested if isinstance(row, dict)]
                if isinstance(nested, list)
                else [payload]
            )

    for row in rows:
        stage = row.get("stage")
        if stage:
            counts[str(stage)] += 1
    return dict(counts)


def build_generation_metrics(
    generation_name: str,
    records: list[dict[str, Any]],
    source_dir: Path,
) -> dict[str, Any]:
    """Compute cumulative acceptance and exact-denominator generation errors."""
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

    error_counts = read_error_counts(source_dir)
    execution_errors = int(error_counts.get("code_execution", 0))
    regeneration_errors = int(error_counts.get("code_regeneration", 0))
    return {
        "name": generation_name,
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



def normalize_image_path(path_str: str) -> str:
    """Return chart-directory-relative image path."""
    return f"images/{Path(path_str).name}"


def discover_dataset_dirs(data_root: Path) -> dict[str, Path]:
    """Find named dataset folders that contain a metadata.jsonl file."""
    datasets_root = data_root / "dataset"
    dataset_dirs: dict[str, Path] = {}

    # Backward compatibility for bundles generated from the former flat layout.
    if (datasets_root / "metadata.jsonl").is_file():
        dataset_dirs[datasets_root.name] = datasets_root

    if datasets_root.is_dir():
        for child in sorted(datasets_root.iterdir(), key=lambda path: path.name.lower()):
            if child.is_dir() and (child / "metadata.jsonl").is_file():
                dataset_dirs[child.name] = child

    if not dataset_dirs:
        raise FileNotFoundError(
            f"No dataset folders with metadata.jsonl found under {datasets_root}"
        )
    return dataset_dirs


def collect_metadata(dataset_dirs: dict[str, Path]) -> tuple[
    list[dict[str, Any]],
    dict[str, list[dict[str, Any]]],
    dict[str, dict[str, Any]],
    dict[str, str],
    dict[str, dict[str, Any]],
]:
    ordered_records: list[dict[str, Any]] = []
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    chart_to_dataset: dict[str, str] = {}
    dataset_info: dict[str, dict[str, Any]] = {}
    generation_info: dict[str, dict[str, Any]] = {}

    for generation_name, source_dir in dataset_dirs.items():
        generation_records: list[dict[str, Any]] = []
        for rec in read_jsonl(source_dir / "metadata.jsonl"):
            dataset = rec.get("dataset") or {}
            dataset_id = str(dataset.get("id", "?"))
            chart_id = str(rec["id"])
            if chart_id in chart_to_dataset:
                raise ValueError(
                    f"Duplicate chart id {chart_id!r} across generation datasets"
                )

            rec["_viewer_generation_dataset"] = generation_name
            ordered_records.append(rec)
            chart_to_dataset[chart_id] = dataset_id
            by_dataset[dataset_id].append(rec)
            generation_records.append(rec)
            if dataset_id not in dataset_info:
                dataset_info[dataset_id] = {
                    "id": dataset_id,
                    "description": dataset.get("description", ""),
                    "label": dataset_label(dataset.get("description", "")),
                }

        generation_info[generation_name] = build_generation_metrics(
            generation_name,
            generation_records,
            source_dir,
        )

    return (
        ordered_records,
        dict(by_dataset),
        dataset_info,
        chart_to_dataset,
        generation_info,
    )


def collect_results(
    results_dir: Path,
    chart_to_dataset: dict[str, str],
) -> tuple[dict[str, list[dict[str, Any]]], list[str], dict[str, dict[str, int]]]:
    by_chart: dict[str, list[dict[str, Any]]] = defaultdict(list)
    per_chart_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "incorrect": 0})
    models: list[str] = []

    for result_file in sorted(results_dir.glob("*.jsonl")):
        model = result_file.stem
        models.append(model)
        for rec in read_jsonl(result_file):
            chart_id = str(rec.get("chart_id") or rec.get("graph_id") or "")
            if not chart_id:
                continue
            dataset_id = chart_to_dataset.get(chart_id)
            if dataset_id is None:
                continue
            slim = dict(rec)
            slim["model"] = model
            by_chart[chart_id].append(slim)
            if rec.get("correct"):
                per_chart_stats[chart_id]["correct"] += 1
            else:
                per_chart_stats[chart_id]["incorrect"] += 1

    return dict(by_chart), models, dict(per_chart_stats)


def write_detail_image(src: Path, dest: Path, max_width: int, jpeg_quality: int) -> bool:
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(src) as im:
            im = im.convert("RGB")
            ratio = max_width / im.width if im.width > max_width else 1.0
            if ratio < 1.0:
                im = im.resize((max_width, int(im.height * ratio)), Image.LANCZOS)
            im.save(dest, format="JPEG", quality=jpeg_quality, optimize=True)
        return True
    except Exception as exc:
        print(f"Warning: failed to write detail image for {src}: {exc}")
        return False


def write_chart_images(
    record: dict[str, Any],
    source_dir: Path,
    out_dir: Path,
    max_width: int,
    jpeg_quality: int,
) -> tuple[dict[str, str], int, int]:
    path_map: dict[str, str] = {}
    written = 0
    missing = 0
    seen: set[str] = set()
    images_dir = source_dir / "images"
    dest_dir = out_dir / "images"

    for img in record.get("images", []) or []:
        path_str = img.get("path") if isinstance(img, dict) else None
        if not path_str:
            continue
        src_name = Path(path_str).name
        if src_name in seen:
            continue
        seen.add(src_name)
        src = source_dir / path_str
        if not src.exists():
            src = images_dir / src_name
        if not src.exists():
            missing += 1
            continue
        dest_name = f"{Path(src_name).stem}.jpg"
        if write_detail_image(src, dest_dir / dest_name, max_width, jpeg_quality):
            path_map[path_str] = f"images/{dest_name}"
            written += 1

    return path_map, written, missing


def rewrite_chart_image_paths(record: dict[str, Any], path_map: dict[str, str]) -> dict[str, Any]:
    rewritten = {key: value for key, value in record.items() if not key.startswith("_viewer_")}
    images = []
    for img in record.get("images", []) or []:
        if not isinstance(img, dict):
            continue
        img = dict(img)
        path_str = img.get("path")
        if isinstance(path_str, str):
            img["path"] = path_map.get(path_str, normalize_image_path(path_str))
        images.append(img)
    rewritten["images"] = images
    return rewritten


def resolve_source_image(source_dir: Path, path_str: str | None) -> Path | None:
    if not path_str:
        return None
    images_dir = source_dir / "images"
    src = source_dir / path_str
    if src.exists():
        return src
    src = images_dir / Path(path_str).name
    return src if src.exists() else None


def write_thumbnail(src: Path, dest: Path, width: int) -> bool:
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(src) as im:
            im = im.convert("RGB")
            ratio = width / im.width if im.width > width else 1.0
            if ratio < 1.0:
                im = im.resize((width, int(im.height * ratio)), Image.LANCZOS)
            im.save(dest, format="JPEG", quality=82, optimize=True)
        return True
    except Exception as exc:
        print(f"Warning: failed to create thumbnail for {src}: {exc}")
        return False


def build_global_chart_index(
    records: list[dict[str, Any]],
    per_chart_stats: dict[str, dict[str, int]],
    dataset_dirs: dict[str, Path],
    output: Path,
    base_url: str,
    thumbnail_width: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rec in records:
        dataset_id = str((rec.get("dataset") or {}).get("id", "?"))
        generation_name = str(rec["_viewer_generation_dataset"])
        source_dir = dataset_dirs[generation_name]
        graph = rec.get("graph") or {}
        chart_id = str(rec["id"])
        final_path = final_image_path(rec)
        thumb_rel = f"thumbnails/{generation_name}/{dataset_id}/{chart_id}.jpg"
        detail_rel = f"charts/{generation_name}/{dataset_id}/{chart_id}/"
        thumb_url = f"{base_url.rstrip('/')}/{thumb_rel}" if base_url else None
        detail_url = f"{base_url.rstrip('/')}/{detail_rel}" if base_url else None
        src = resolve_source_image(source_dir, final_path)
        thumbnail_available = False
        if src is not None:
            thumbnail_available = write_thumbnail(src, output / thumb_rel, thumbnail_width)
        rows.append({
            "id": chart_id,
            "dataset_id": dataset_id,
            "generation_dataset": generation_name,
            "dataset_description": (rec.get("dataset") or {}).get("description", ""),
            "canonical_type": canonicalize_chart_type(graph.get("type", "")),
            "graph_type": graph.get("type", ""),
            "short_description": graph.get("short_description", ""),
            "thumbnail": thumb_rel if thumbnail_available else None,
            "thumbnail_url": thumb_url if thumbnail_available else None,
            "detail": detail_rel,
            "detail_url": detail_url,
            "metadata": f"{detail_rel}metadata.json",
            "metadata_url": f"{detail_url}metadata.json" if detail_url else None,
            "results": f"{detail_rel}results.jsonl.gz",
            "results_url": f"{detail_url}results.jsonl.gz" if detail_url else None,
            "final_image": normalize_image_path(final_path or ""),
            "quality": quality(rec),
            "accepted": chart_accepted(rec),
            "correct": per_chart_stats.get(chart_id, {}).get("correct", 0),
            "incorrect": per_chart_stats.get(chart_id, {}).get("incorrect", 0),
        })
    return rows


def build(args: argparse.Namespace) -> None:
    data_root = args.data_root.resolve()
    output = args.output.resolve()
    dataset_dirs = discover_dataset_dirs(data_root)
    if args.generation_dataset:
        wanted_generation = set(args.generation_dataset)
        missing = wanted_generation.difference(dataset_dirs)
        if missing:
            raise ValueError(
                f"Unknown generation dataset folder(s): {', '.join(sorted(missing))}"
            )
        dataset_dirs = {
            name: path for name, path in dataset_dirs.items() if name in wanted_generation
        }

    results_dir = args.results_dir.resolve() if args.results_dir else data_root / "results"
    if not results_dir.exists() and args.results_dir is None and (data_root / "evaluation").exists():
        results_dir = data_root / "evaluation"
    if not results_dir.exists():
        raise FileNotFoundError(f"Missing results directory: {results_dir}")

    if args.clean and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    charts_dir = output / "charts"

    (
        ordered_records,
        by_dataset,
        dataset_info,
        chart_to_dataset,
        generation_info,
    ) = collect_metadata(dataset_dirs)
    if args.dataset_id:
        wanted = {str(dataset_id) for dataset_id in args.dataset_id}
        ordered_records = [
            rec for rec in ordered_records
            if str((rec.get("dataset") or {}).get("id", "?")) in wanted
        ]
        by_dataset = {k: v for k, v in by_dataset.items() if k in wanted}
        dataset_info = {k: v for k, v in dataset_info.items() if k in wanted}
        chart_to_dataset = {
            chart_id: dataset_id
            for chart_id, dataset_id in chart_to_dataset.items()
            if dataset_id in wanted
        }
    results_by_chart, models, per_chart_stats = collect_results(results_dir, chart_to_dataset)
    generation_counts = Counter(
        str(rec["_viewer_generation_dataset"]) for rec in ordered_records
    )
    generation_info = {
        name: {
            **info,
            "bundled_chart_count": generation_counts[name],
        }
        for name, info in generation_info.items()
        if generation_counts[name]
    }
    chart_index = build_global_chart_index(
        ordered_records,
        per_chart_stats,
        dataset_dirs,
        output,
        args.base_url,
        args.thumbnail_width,
    )
    if args.shuffle_seed is not None:
        random.Random(args.shuffle_seed).shuffle(chart_index)
    write_jsonl_gz(output / "charts.jsonl.gz", chart_index)

    dataset_entries: list[dict[str, Any]] = []
    for dataset_id, records in sorted(by_dataset.items(), key=lambda item: item[0]):
        dataset_image_count = 0
        dataset_result_count = 0
        dataset_detail_bytes = 0
        dataset_chart_entries: list[dict[str, Any]] = []
        for rec in records:
            chart_id = str(rec["id"])
            results = results_by_chart.get(chart_id, [])
            generation_name = str(rec["_viewer_generation_dataset"])
            source_dir = dataset_dirs[generation_name]
            detail_rel = f"charts/{generation_name}/{dataset_id}/{chart_id}/"
            dataset_result_count += len(results)
            graph = rec.get("graph") or {}
            chart_dir = output / detail_rel
            chart_dir.mkdir(parents=True, exist_ok=True)
            path_map, copied, missing = write_chart_images(
                rec,
                source_dir,
                chart_dir,
                args.detail_image_width,
                args.detail_jpeg_quality,
            )
            rewritten_record = rewrite_chart_image_paths(rec, path_map)
            write_json(chart_dir / "manifest.json", {
                "id": chart_id,
                "dataset_id": dataset_id,
                "dataset": dataset_info[dataset_id],
                "canonical_type": canonicalize_chart_type(graph.get("type", "")),
                "generation_dataset": generation_name,
                "accepted": chart_accepted(rec),
                "graph_type": graph.get("type", ""),
                "result_record_count": len(results),
                "detail_image_width": args.detail_image_width,
                "detail_jpeg_quality": args.detail_jpeg_quality,
                "models": models,
            })
            write_json(chart_dir / "metadata.json", rewritten_record)
            write_jsonl_gz(chart_dir / "results.jsonl.gz", results)
            dataset_image_count += copied
            if missing:
                print(f"Warning: chart {chart_id} is missing {missing} images")
            detail_bytes = sum(p.stat().st_size for p in chart_dir.rglob("*") if p.is_file())
            dataset_detail_bytes += detail_bytes

            dataset_chart_entries.append({
                "id": chart_id,
                "accepted": chart_accepted(rec),
                "generation_dataset": generation_name,
                "detail": detail_rel,
                "metadata": f"{detail_rel}metadata.json",
                "results": f"{detail_rel}results.jsonl.gz",
                "detail_bytes": detail_bytes,
                "result_record_count": len(results),
                "image_count": copied,
            })

        write_json(output / "datasets" / f"{dataset_id}.json", {
            **dataset_info[dataset_id],
            "chart_count": len(records),
            "result_record_count": dataset_result_count,
            "image_count": dataset_image_count,
            "detail_bytes": dataset_detail_bytes,
            "models": models,
            "charts": dataset_chart_entries,
        })
        write_jsonl(output / "datasets" / f"{dataset_id}.charts.jsonl", dataset_chart_entries)

        info = dataset_info[dataset_id]
        dataset_entries.append({
            **info,
            "chart_count": len(records),
            "image_count": dataset_image_count,
            "result_record_count": dataset_result_count,
            "detail_bytes": dataset_detail_bytes,
            "manifest": f"datasets/{dataset_id}.json",
            "manifest_url": f"{args.base_url.rstrip('/')}/datasets/{dataset_id}.json" if args.base_url else None,
            "charts": f"datasets/{dataset_id}.charts.jsonl",
            "charts_url": f"{args.base_url.rstrip('/')}/datasets/{dataset_id}.charts.jsonl" if args.base_url else None,
        })

    type_counts = Counter(row["canonical_type"] for row in chart_index)
    manifest = {
        "schema_version": 5,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "base_url": args.base_url,
        "chart_index": "charts.jsonl.gz",
        "chart_index_url": f"{args.base_url.rstrip('/')}/charts.jsonl.gz" if args.base_url else None,
        "dataset_count": len(dataset_entries),
        "generation_dataset_count": len(generation_info),
        "generation_datasets": [generation_info[name] for name in sorted(generation_info)],
        "chart_count": len(chart_index),
        "accepted_chart_count": sum(1 for row in chart_index if row["accepted"]),
        "rejected_chart_count": sum(1 for row in chart_index if not row["accepted"]),
        "chart_order": "shuffle" if args.shuffle_seed is not None else "metadata",
        "shuffle_seed": args.shuffle_seed,
        "model_count": len(models),
        "models": models,
        "detail_image_width": args.detail_image_width,
        "detail_jpeg_quality": args.detail_jpeg_quality,
        "canonical_type_counts": dict(sorted(type_counts.items())),
        "datasets": dataset_entries,
    }
    write_json(output / "manifest.json", manifest)
    print(f"Wrote {len(chart_index)} chart directories to {charts_dir}")
    print(f"Wrote manifest to {output / 'manifest.json'}")
    print(f"Wrote chart index to {output / 'charts.jsonl.gz'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repository root containing named folders under dataset/.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        help="Directory containing per-model JSONL results (defaults to results/ or evaluation/).",
    )
    parser.add_argument(
        "--generation-dataset",
        action="append",
        help="Only package this named dataset folder. May be supplied multiple times.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory to upload to the FTP/web server.",
    )
    parser.add_argument(
        "--base-url",
        default="",
        help="Optional public HTTP(S) base URL for the uploaded output directory.",
    )
    parser.add_argument(
        "--dataset-id",
        action="append",
        help="Only package this source dataset id. May be supplied multiple times.",
    )
    parser.add_argument(
        "--thumbnail-width",
        type=int,
        default=360,
        help="Width in pixels for global JPEG thumbnails.",
    )
    parser.add_argument(
        "--detail-image-width",
        type=int,
        default=1400,
        help="Maximum width in pixels for detail-view JPEG images.",
    )
    parser.add_argument(
        "--detail-jpeg-quality",
        type=int,
        default=88,
        help="JPEG quality for detail-view images.",
    )
    parser.add_argument(
        "--shuffle-seed",
        type=parse_shuffle_seed,
        default=42,
        help="Shuffle the global chart grid order with this fixed seed. Use --shuffle-seed none to preserve metadata order.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove the output directory before writing the new bundle.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
