# Local Dataset Viewer

Use `app.py` for local filesystem-backed inspection of the full dataset and raw
result files.

## Running

From the repository root:

```bash
.venv/bin/python -m streamlit run chart-generation/dataset-viewer/local/app.py
```

The app opens at `http://localhost:8501` by default.

## Data Location

By default, the local app looks for `dataset/` and `results/` at the repository
root.

Override these independently if needed:

```bash
DATA_DIR=/path/to/data \
RESULTS_DIR=/path/to/results \
.venv/bin/python -m streamlit run chart-generation/dataset-viewer/local/app.py
```

| Variable | Default | Contents |
|---|---|---|
| `DATA_DIR` | repository root | directory containing `dataset/metadata.jsonl` and `dataset/images/` |
| `RESULTS_DIR` | `$DATA_DIR/results` | directory containing per-model `*.jsonl` result files |

## Features

- Thumbnail grid of all charts, showing the final iteration image.
- Filters for canonical chart type, dataset, text search, and plot quality.
- Sorting by original order or number of incorrect model answers.
- Detail view with iteration images, feedback, code, descriptions, structured
  data, questions, and per-model answers.
- URL query parameters preserve filters, sorting, page, and selected chart.

## Caching

The local viewer builds disk caches under `dataset-viewer/local/.cache/`:

- per-model byte-offset indexes for fast result lookups;
- per-chart answer stats used for sorting.

These caches are safe to delete.
