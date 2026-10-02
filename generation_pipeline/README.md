# Generation Pipeline

This folder contains the main dataset generation and evaluation pipeline for the chart generation paper.

The workflow is centered around transforming tabular data into structured visualizations, saving chart metadata, and evaluating generated question-answer pairs.

## Contents

- `generation_job.py`
  - Main pipeline script for generating charts and structured metadata.
  - Uses OpenML datasets, LLM calls, and rendering code to create charts with corresponding questions.
- `evaluation_job.py`
  - Offline batch evaluation using a local `vLLM`/transformers-based model stack.
  - Generates answers to chart questions and grades them with a separate checking model.
- `evaluation_online.py`
  - Online evaluation using an OpenAI/OpenRouter-compatible API.
  - Sends chart questions and images to a model, then checks answers for correctness.
- `render_chart_from_metadata.py`
  - Re-renders a saved chart from `dataset/metadata.jsonl`.
  - Supports local dataset overrides and paper-friendly figure sizing.
- `join_metadata.py`
  - Combines multiple `metadataN.jsonl` files into a single `metadata.jsonl` file.
- `rewrite_dataset.py`
  - Utility script for correcting image path prefixes inside dataset metadata files.
- `add_question_types.py`
  - Helper for labeling generated questions with categories such as `metadata`, `value extraction`, `comparison`, `trends`, and `reasoning`.
- `model_scripts/`
  - Auxiliary training/inference utilities for the vision-language model / connector stack.
- `archived/`
  - Older notebooks and evaluation parameter files from prior experiments.

## Typical Workflow

1. Start a local `vLLM` server for generation. (The code should probably be able to connect to a remote provider like OpenRouter, although this was not tested. so use carefully.)
2. Generate charts and metadata with `generation_job.py`, then close the vLLM server.
3. Join metadata files with `join_metadata.py` if you ran generation in parallel. 
4. OPTIONAL: You can rerun the question type assigment with `add_question_types.py`
5. Evaluate question-answer performance with `evaluation_job.py` or `evaluation_online.py`.
6. OPTIONAL: Re-render produced charts using `render_chart_from_metadata.py`

> For `generation_job.py`, we used a local `vLLM` server with `qwen3.5-27b` in our experiments.
> For `evaluation_job.py` and `evaluation_online.py`, we used `qwen3.5-9b` as the grading model.

## Requirements

The pipeline relies on several Python packages and services, such as:

- `langchain_openai`
- `langchain_core`
- `openai`
- `vllm`
- `transformers`
- `Pillow`
- `tqdm`
- `scikit-learn`
- `numpy`
- `matplotlib`
- `pandas`
- `openml`

It also assumes access to an LLM backend, either local via a `vLLM` endpoint or remote via an OpenAI/OpenRouter-compatible API.

## Data Expectations

The scripts expect a `dataset/images/` directory and a `evaluation/` directory in this folder. They should be created by the scripts, but better to be safe.

The evaluation scripts will expect 
- `metadata.jsonl`
- chart image files referenced by metadata in the `images/`

## Example Commands

- Start the local `vLLM` server for generation:
  - `vllm --model qwen3.5-27b --port 8888 --disable-remote-attention`

- Generate charts and metadata (required `--run_id`):
  - `python generation_pipeline/generation_job.py --metadata_file metadata.jsonl --datasets 10`
  - `python generation_pipeline/generation_job.py --run_id 1 --datasets 20 --regenerate`

- Offline evaluation with a local model:
  - `python generation_pipeline/evaluation_job.py --metadata_file metadata.jsonl --model_path /workspace/models --model_name qwen3.5-27b`

- Online evaluation via OpenRouter/OpenAI-compatible API:
  - `python generation_pipeline/evaluation_online.py --metadata_file metadata.jsonl --model_name qwen/qwen3-vl-32b-instruct --check_model_name qwen3.5-9b --max_workers 16`

- Join generated metadata files:
  - `python generation_pipeline/join_metadata.py --directory ./dataset --min-number 0 --filename metadata.jsonl`

- Re-render metadata charts:
  - `python generation_pipeline/render_chart_from_metadata.py --graph-id <UUID> --output paper/figures/example_chart.png`

- Repair dataset metadata image paths:
  - `python generation_pipeline/rewrite_dataset.py`

## Notes

- `generation_job.py` performs dataset selection, dataset semantic filtering, plot proposal, code generation, and metadata creation.
- Every selected dataset receives an additional `sanitize_description` LLM call after description formatting and variable replacement, so the sanitizer sees the final feature names. When formatting or replacement is disabled, sanitization uses the available rich description and unchanged feature names. It preserves variable meanings, units, and category codes while removing empirical statistics, result summaries, outcome-generation rules, and dataset lookup cues. Sanitization is mandatory for all generation configurations and uses the non-reasoning client without dataframe tools. A failed or empty response skips the dataset instead of falling back to unsanitized text.
- Chart proposal, code generation, and chart description generation use the rich formatted description. Metadata saves both the rich text as `dataset.description` and the sanitized text as `dataset.sanitized_description`, both using the final feature names. The current evaluation scripts continue to read `dataset.description`, so later dataset cleaning can choose which version to use. Existing metadata is not rewritten. LLM sanitization reduces answer leakage but does not guarantee its absence; semantic definitions can still answer context-only questions, which need separate question validation.
- Both question-generation modes receive the sanitized description as evaluator-visible context and retain the rich dataset description, chart description, structured data, and optional tools solely for verifying reference answers. Questions must require the image and be answerable from the image, sanitized description, and context explicitly supplied in the question, allowing visual estimates. Numeric reference answers retain exact verified plotted values when available; estimates such as 9.5 are acceptable for a reference of 9.543 at the displayed scale. The grounding judge receives only these evaluator-visible inputs. Evaluation still requires later dataset cleaning to select the sanitized description.
- The grounding judge allows definitions, background facts, assumptions, and reasoning rules explicitly supplied in the question, provided answering still requires distinguishable chart evidence. Questions remain invalid if they require unstated outside knowledge, reveal hidden plotted observations, or give away the answer. Basic language comprehension, chart reading, arithmetic, and logical deductions from supplied evidence remain allowed; assertions in a proposed answer do not supply missing context.
- `evaluation_job.py` is optimized for batch offline benchmarking, while `evaluation_online.py` is designed for API-based evaluation.
- `render_chart_from_metadata.py` can also load a local dataset file via `--dataset-file` to avoid remote fetching.
- If you have issues running newer models, installing a nightly `vLLM` build and then reinstalling `transformers` usually helps.

