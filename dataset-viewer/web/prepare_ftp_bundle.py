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


def normalize_image_path(path_str: str) -> str:
    """Return chart-directory-relative image path."""
    return f"images/{Path(path_str).name}"


def collect_metadata(metadata_file: Path) -> tuple[
    list[dict[str, Any]],
    dict[str, list[dict[str, Any]]],
    dict[str, dict[str, Any]],
    dict[str, str],
]:
    ordered_records: list[dict[str, Any]] = []
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    chart_to_dataset: dict[str, str] = {}
    dataset_info: dict[str, dict[str, Any]] = {}

    for rec in read_jsonl(metadata_file):
        ordered_records.append(rec)
        dataset = rec.get("dataset") or {}
        dataset_id = str(dataset.get("id", "?"))
        chart_id = str(rec["id"])
        chart_to_dataset[chart_id] = dataset_id
        by_dataset[dataset_id].append(rec)
        if dataset_id not in dataset_info:
            dataset_info[dataset_id] = {
                "id": dataset_id,
                "description": dataset.get("description", ""),
                "label": dataset_label(dataset.get("description", "")),
            }

    return ordered_records, dict(by_dataset), dataset_info, chart_to_dataset


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
    data_root: Path,
    out_dir: Path,
    max_width: int,
    jpeg_quality: int,
) -> tuple[dict[str, str], int, int]:
    path_map: dict[str, str] = {}
    written = 0
    missing = 0
    seen: set[str] = set()
    images_dir = data_root / "dataset" / "images"
    dest_dir = out_dir / "images"

    for img in record.get("images", []) or []:
        path_str = img.get("path") if isinstance(img, dict) else None
        if not path_str:
            continue
        src_name = Path(path_str).name
        if src_name in seen:
            continue
        seen.add(src_name)
        src = data_root / "dataset" / path_str
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
    rewritten = dict(record)
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


def resolve_source_image(data_root: Path, path_str: str | None) -> Path | None:
    if not path_str:
        return None
    images_dir = data_root / "dataset" / "images"
    src = data_root / "dataset" / path_str
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
    data_root: Path,
    output: Path,
    base_url: str,
    thumbnail_width: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rec in records:
        dataset_id = str((rec.get("dataset") or {}).get("id", "?"))
        graph = rec.get("graph") or {}
        chart_id = str(rec["id"])
        final_path = final_image_path(rec)
        thumb_rel = f"thumbnails/{dataset_id}/{chart_id}.jpg"
        detail_rel = f"charts/{dataset_id}/{chart_id}/"
        thumb_url = f"{base_url.rstrip('/')}/{thumb_rel}" if base_url else None
        detail_url = f"{base_url.rstrip('/')}/{detail_rel}" if base_url else None
        src = resolve_source_image(data_root, final_path)
        thumbnail_available = False
        if src is not None:
            thumbnail_available = write_thumbnail(src, output / thumb_rel, thumbnail_width)
        rows.append({
            "id": chart_id,
            "dataset_id": dataset_id,
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
            "correct": per_chart_stats.get(chart_id, {}).get("correct", 0),
            "incorrect": per_chart_stats.get(chart_id, {}).get("incorrect", 0),
        })
    return rows


def build(args: argparse.Namespace) -> None:
    data_root = args.data_root.resolve()
    output = args.output.resolve()
    metadata_file = data_root / "dataset" / "metadata.jsonl"
    results_dir = data_root / "results"

    if not metadata_file.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_file}")
    if not results_dir.exists():
        raise FileNotFoundError(f"Missing results directory: {results_dir}")

    if args.clean and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    charts_dir = output / "charts"

    ordered_records, by_dataset, dataset_info, chart_to_dataset = collect_metadata(metadata_file)
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
    chart_index = build_global_chart_index(
        ordered_records,
        per_chart_stats,
        data_root,
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
            dataset_result_count += len(results)
            graph = rec.get("graph") or {}
            chart_dir = charts_dir / dataset_id / chart_id
            chart_dir.mkdir(parents=True, exist_ok=True)
            path_map, copied, missing = write_chart_images(
                rec,
                data_root,
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
                "detail": f"charts/{dataset_id}/{chart_id}/",
                "metadata": f"charts/{dataset_id}/{chart_id}/metadata.json",
                "results": f"charts/{dataset_id}/{chart_id}/results.jsonl.gz",
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
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "base_url": args.base_url,
        "chart_index": "charts.jsonl.gz",
        "chart_index_url": f"{args.base_url.rstrip('/')}/charts.jsonl.gz" if args.base_url else None,
        "dataset_count": len(dataset_entries),
        "chart_count": len(chart_index),
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
        default=Path(__file__).resolve().parents[3],
        help="Repository root containing dataset/ and results/.",
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
        help="Only package this dataset id. May be supplied multiple times; mainly useful for tests.",
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
