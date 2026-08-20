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

By default, the local app discovers every `dataset/<name>/metadata.jsonl` at
the repository root. Results are read from `results/`, falling back to
`evaluation/` when that is the available directory.

Override these independently if needed:

```bash
DATA_DIR=/path/to/data \
RESULTS_DIR=/path/to/results \
.venv/bin/python -m streamlit run chart-generation/dataset-viewer/local/app.py
```

| Variable | Default | Contents |
|---|---|---|
| `DATA_DIR` | repository root | directory containing named `dataset/<name>/metadata.jsonl` and `dataset/<name>/images/` folders |
| `RESULTS_DIR` | `$DATA_DIR/results` or `$DATA_DIR/evaluation` | directory containing per-model `*.jsonl` result files |

## Features

- Thumbnail grid of all charts, showing the final iteration image and a red or
  green acceptance indicator.
- Generation-dataset selector populated from the folder names under `dataset/`.
- Per-generation-dataset metrics showing cumulative acceptance at every
  iteration and error rates loaded from `error.jsonl` (also supports
  `errors.jsonl`, `error.json`, and `errors.json`). Execution errors use
  chart count; regeneration errors use iteration outputs minus chart count.
- Filters for canonical chart type, dataset, acceptance, text search, and plot quality.
- A dedicated **Dataset statistics** page, opened from the sidebar, compares
  all generation folders as table rows and includes average iteration outputs
  through first acceptance per accepted chart.
- Sorting by original order or number of incorrect model answers.
- Detail view with iteration images, visible per-iteration feedback directly
  above the iteration code, acceptance status, descriptions, structured
  data, joined and typed LLM messages, image placeholders, reasoning traces,
  and access to the exact raw JSON from `metadata.jsonl`, plus questions and
  per-model answers.
- URL query parameters preserve filters, sorting, page, and selected chart.

## Caching

The local viewer builds disk caches under `dataset-viewer/local/.cache/`:

- per-model byte-offset indexes for fast result lookups;
- per-chart answer stats used for sorting.

These caches are safe to delete.
