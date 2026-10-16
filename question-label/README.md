# Question labeling

Label question-answer pairs independently of the model's data and visual
validators. Select a generation dataset, review the chart, question, proposed
answer and explanation, optionally add notes, and select both validities.
Click **Display validity** to reveal the model's flags and reasons after
selecting both validities. Each flag appears beside its selector; hover over
the model judgment to read its reason. Your selections and notes remain editable.
**Next question** saves your final response and advances; nothing is saved
when selecting labels or revealing model judgments. You may also save with
**Next question** without revealing the model judgments.

From the repository root, install and run:

```powershell
.venv/Scripts/python.exe -m pip install -r question-label/requirements.txt
.venv/Scripts/python.exe -m streamlit run question-label/app.py
```

On macOS/Linux use `.venv/bin/python` instead. Open the URL printed by Streamlit
(normally `http://localhost:8501`).

## Datasets and review criteria

The dataset selector automatically discovers `dataset/<name>/metadata.jsonl`
under this Git repository. No dataset location needs to be entered.

The chart uses the image marked `selected`, or otherwise the latest `_itN`
iteration. A graph whose selected/final image is missing or empty is excluded,
with counts shown in the sidebar. Images are fully validated and decoded before
display. For other damaged images, the app shows the affected path and offers
**Retry image** after restoration or **Skip chart** without saving labels.
Skipping excludes every question for that chart for the current session only;
it does not count as a completed review. After restoring missing or empty files,
use **Refresh images** in the sidebar to include those charts again.
The app indexes byte offsets instead of retaining
large generation traces in memory. Metadata size and modification time
invalidate the in-memory index automatically.

Judge these independently, following the model validator prompts:

- **Data agreement:** Do the data in the question, answer, and explanation agree with the chart? Choose **Yes**, **No**, **Unsure**, or **Partly**. Check values, calculations, comparisons, units, and reasonable estimates; **Partly** means some claims agree and others do not.
- **Visual / semantic check:** Does answering require reading the chart, and does the chart support the answer and explanation? Choose **No** for a self-contained question answerable from its wording or allowed context alone, even if correct. No hidden data, unstated specialist knowledge, or leaked chart observations/answers; reasonable estimates count. **Partly** means the image is needed but only some claims are supported; **Unsure** means you cannot tell.
- **Unsure:** insufficient evidence; saved as JSON `null`, alongside the
  explicit `Unsure` choice. Both checks use **Yes**, **No**, **Unsure**, and
  **Partly**; neither judgment is preselected.

The app displays the chart, question, proposed answer, explanation, and
sanitized dataset context. Use **Unsure** when this evidence cannot establish
data validity. Model judgments and reasons stay hidden until you click
**Display validity**. The log preserves the labels selected before the reveal
as well as your final labels, so changes after viewing model feedback can be
identified. A failed save keeps you on the current question with your edits.

## Dataset statistics

The first section of the page, **Dataset statistics and validator agreement**,
is collapsed by default. Expand it to see statistics for the selected dataset
and annotator, updated after each saved review:

- Available questions, graphs with questions, saved reviews, review coverage,
  and the full model validity combination distribution.
- Agreement separately for data and visual / semantic validity: the proportion
  of comparable human/model pairs whose labels match.
- Precision, recall, and F1 separately for each validity axis and for each
  positive class (**Valid** and **Invalid**). Human labels are the reference;
  model flags are predictions. Precision measures how often a model prediction
  of that class matches the human label. Recall measures how often the model
  identifies human-labeled examples of that class. F1 is their harmonic mean.
- Human support, model prediction counts, confusion counts with explicit
  human/model column labels, and numbers excluded from each comparison.

Metrics use only final saved responses for questions in the current dataset
and annotator. **Yes** maps to valid and **No** to invalid. Human **Partly**,
**Unsure**, and missing model judgments are excluded from binary metrics per
axis; **Partly** is counted separately from **Unsure**. Undefined metrics display
**N/A**. The reviewed sample is stratified,
so these metrics do not estimate full-dataset performance without weighting.
Final labels can be influenced by revealed model feedback; the section also
reports how many reviews recorded a reveal before saving.

## Sampling and resuming

Sampling is stratified by the **joint** model data/visual flags. Each shuffled
block draws one random remaining question from each nonempty stratum, without
replacement. This balances strata until smaller strata are exhausted; larger
strata then continue until every question is labeled. Missing judgments form
their own strata. Visual flags use `vlisual_valid`, then `visual_valid`, then
legacy `valid`. A missing `data_valid` stays unknown and is never inferred
from the visual flag.

Enter a stable **Annotator ID** (different IDs for different people). Saved
questions are skipped for that annotator and dataset, including after a server
restart or switching datasets. Refreshing before submission can select another
question. Concurrent sessions share a file lock and cannot save the same
question twice. Model disagreements do not change the human labels. To perform
a separate labeling pass, use another annotator ID.

## Response files

Responses append on **Next question** to `question-label/responses/labels-<hash>.jsonl`.
The hash identifies the dataset's absolute directory and annotator ID, avoiding
unsafe characters in filenames. Changing the dataset location creates a
different response stream. Set `QUESTION_LABEL_OUTPUT_DIR` to use another
persistent directory. The output directory is ignored by Git. The sidebar
provides the full output path and a JSONL download. Files are stored on the
Streamlit server; deployments need persistent storage to survive redeployment.

Each line contains:

- `schema_version`, `annotation_id`, `question_key`, `annotator_id`, `session_id`,
  UTC timestamps, review duration, and notes;
- `human_labels.data_valid` and `human_labels.visual_valid`: `true` for **Yes**,
  `false` for **No**, `null` for **Unsure**, or `"partly"` for **Partly**,
  alongside explicit `human_choices`. Older **Valid** / **Invalid** labels
  remain compatible;
- `model_labels` with separate normalized flags, the original visual flag field
  name, and the model's reasons when supplied;
- `model_feedback_displayed`, `validity_displayed_at`, `labels_before_reveal`,
  and `labels_changed_after_reveal` (new records use schema version 4);
- the complete original `question` object, its zero-based `question_index`, and
  a source question ID if supplied;
- `dataset_name`, `dataset_dir`, original dataset metadata and source dataset ID;
- `graph_id` (the chart record's `id`, with `graph_id`, `chart_id`, or `prefix_id`
  fallbacks), `prefix_id`, graph acceptance, graph metadata and code;
- absolute `image_path`, original `image_metadata_path`, and whether the image
  was explicitly selected;
- `source`: absolute metadata path, size/mtime fingerprint, one-based line,
  byte offset and length, and SHA-256 of the original chart record;
- `sampling`: algorithm, joint stratum, random seed, zero-based draw position,
  full stratum populations, and remaining populations when the session began.

The stable question key hashes the metadata path, graph ID, question index,
and complete question object. Editing a question or its model judgments makes
it eligible for a new review. Logs preserve the original snapshot for later
joins and audits. Sampling intentionally overrepresents rare strata; use the
logged populations when estimating dataset-wide agreement.

Writes are locked across processes and flushed to disk before the next question
appears. Malformed response rows block new submissions until repaired,
preserving the existing file.

## Checks

```powershell
.venv/Scripts/python.exe -m unittest discover -s question-label/tests -v
```
