# Chart Generation

Code for the structured LLM-based workflow that generates statistical figures from tabular data, along with a browser for the resulting dataset.

## Dataset Access

An interactive viewer for the generated dataset is available at
<https://llm-chart-generation.streamlit.app>.

The raw static data used by the viewer is available from the Biolab file server
at <https://file.biolab.si/llm-chart-generation/>. The server includes the
global manifest and chart index, plus per-chart metadata, model results, and
images.

## Directories

### `generation_pipeline/`

The dataset generation pipeline. Given tabular data, it runs a staged workflow — plot generation, iterative visual refinement, and aligned question-answer generation — producing chart images and structured metadata. Key entry points:

- `generation_job.py` — main generation script
- `evaluation_job.py` / `evaluation_online.py` — benchmarking generated QA pairs against models
- `render_chart_from_metadata.py` — re-renders charts from saved metadata
- `model_scripts/` — training and inference utilities for the connector/VLM stack

### `dataset-viewer/`

A Streamlit app for browsing the generated dataset. Displays the 2,228 charts with per-iteration images, generation feedback, code, questions, and per-model evaluation results. The public instance is available at <https://llm-chart-generation.streamlit.app>. See `dataset-viewer/README.md` for setup and usage.
