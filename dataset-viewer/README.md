# Dataset Viewer

A Streamlit app for browsing the generated chart dataset, inspecting per-iteration images, questions, and model evaluation results.

## Setup

The viewer is self-contained. Create a virtual environment inside this directory and install the dependencies:

```bash
cd dataset-viewer
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running

Activate the environment, then run from inside `dataset-viewer/`:

```bash
source .venv/bin/activate
streamlit run app.py
```

Opens at `http://localhost:8501` by default.

### Data location

By default the app looks for `dataset/` and `results/` in the parent directory of `dataset-viewer/`. Both can be overridden independently via environment variables:

| Variable | Default | Contents |
|---|---|---|
| `DATA_DIR` | `../` | directory containing `dataset/metadata.jsonl` and `dataset/images/` |
| `RESULTS_DIR` | `$DATA_DIR/results` | directory containing per-model `*.jsonl` result files |

Example:

```bash
DATA_DIR=/path/to/data RESULTS_DIR=/path/to/results streamlit run app.py
```

## Features

**Grid view**
- Thumbnail grid of all 2,228 charts (24 per page), showing the final iteration image.
- Click any card to open the detail view.
- Pagination at the top and bottom.

**Filtering & sorting (sidebar)**
- Filter by canonical chart type (24 buckets, normalized via `chart_types.py`).
- Filter by dataset (74 datasets, labeled by description).
- Free-text search over dataset descriptions and chart summaries.
- Sort by original order or by number of incorrect model answers (ascending/descending).
- All filter and sort state is encoded in the URL — page reloads and shared links restore the exact view.

**Detail view**
- Iteration stepper (it0 → itN): shows each generation iteration's image, feedback, and code. Defaults to the final iteration.
- Chart summary, full description, dataset description, final code, and structured data.
- Per-model accuracy bar chart (sorted descending, y-axis fixed 0–100%).
- Question accordion list: each header is color-coded green→red based on the fraction of models that answered correctly.

## File structure

```
dataset-viewer/
├── app.py            # Main Streamlit application
├── indexer.py        # Byte-offset index builder for fast per-chart lookups in result jsonl files
├── chart_types.py    # Chart type normalization (canonical taxonomy, copied from evaluation/)
├── requirements.txt  # Python dependencies
├── .cache/           # Auto-generated index and stats cache files (safe to delete to force rebuild)
└── README.md
```

## Caching

On first run, the app builds two types of disk cache under `.cache/`:

- **Per-model byte-offset indexes** (`*.idx.pkl`) — one per result file, enables O(1) record lookup by `chart_id` (or legacy `graph_id`) without loading 50 MB files into memory.
- **Per-chart answer stats** (`per_chart_stats.pkl`) — correct/incorrect counts per chart summed across all models, used for sorting. Takes ~5s to build; instant on subsequent runs.

Both caches are invalidated automatically when the underlying result files change.
