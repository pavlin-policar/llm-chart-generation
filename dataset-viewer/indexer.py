"""Byte-offset indexer for per-graph lookup in model result jsonl files."""
from __future__ import annotations

import json
import pickle
from collections import defaultdict
from pathlib import Path


def _index_path(cache_dir: Path, results_file: Path) -> Path:
    return cache_dir / f"{results_file.stem}.idx.pkl"


def build_index(results_file: Path, cache_dir: Path) -> dict[str, list[tuple[int, int]]]:
    """Return {graph_id: [(offset, length), ...]}, caching to disk.

    Invalidated when the results file's mtime/size changes.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    idx_file = _index_path(cache_dir, results_file)

    stat = results_file.stat()
    fingerprint = (stat.st_size, int(stat.st_mtime))

    if idx_file.exists():
        try:
            with idx_file.open("rb") as f:
                cached = pickle.load(f)
            if cached.get("fingerprint") == fingerprint:
                return cached["index"]
        except (pickle.PickleError, EOFError, KeyError):
            pass

    index: dict[str, list[tuple[int, int]]] = defaultdict(list)
    with results_file.open("rb") as f:
        offset = 0
        for raw in f:
            length = len(raw)
            try:
                rec = json.loads(raw)
                gid = rec.get("graph_id")
                if gid:
                    index[gid].append((offset, length))
            except json.JSONDecodeError:
                pass
            offset += length

    with idx_file.open("wb") as f:
        pickle.dump({"fingerprint": fingerprint, "index": dict(index)}, f)

    return dict(index)


def read_records(results_file: Path, locations: list[tuple[int, int]]) -> list[dict]:
    """Read specific records from a jsonl by (offset, length) list."""
    out: list[dict] = []
    with results_file.open("rb") as f:
        for offset, length in locations:
            f.seek(offset)
            raw = f.read(length)
            try:
                out.append(json.loads(raw))
            except json.JSONDecodeError:
                continue
    return out
