# Question labeling

Label question-answer pairs independently of the model's data and visual
validators. Select a generation dataset, review the chart, question, proposed
answer and explanation, optionally add notes, and answer all seven review questions.
Use the **Q1–Q7** tabs to choose a review question without scrolling through the
whole form. Choices persist across tabs; the counter lists any unanswered questions.
The derived human flags appear beside the model flags as soon as every question
is answered, and update when you edit a choice.
Click **Display validity** to reveal the model's flags and reasons after
answering all seven questions. Each flag appears beside your derived flag; hover over
the model judgment to read its reason. Your selections and notes remain editable.
**Next question** saves your final response and advances; nothing is saved
when selecting labels or revealing model judgments. You may also save with
**Next question** without revealing the model judgments.

**Skip question** moves to the next question in the random queue without saving.
**Different question from same graph** skips the current question and brings
another remaining question from that chart forward. Both buttons work before
selecting labels and reset the choices, notes, and model reveal for the new
question. They are disabled when no eligible alternative remains. Skips apply
only to the current session; reload the app to review skipped questions again.

To repeat another person's labeling, choose **Mode → Relabel a response file**
in the sidebar, select the same dataset, and enter your own **Annotator ID**.
Select a JSONL file from the response directory or upload one from your computer
(an upload takes precedence). Review the file's questions in their original file
order, with blank choices and notes; the original person's labels are not shown.
**Skip question**, **Different question from same graph**, and **Skip chart** are
disabled in this mode. **Next question** saves to a separate
`responses/relabels-<hash>.jsonl` file; the source file and normal labeling file
remain intact. Selecting the same source file and annotator resumes that pass.
Statistics and progress cover only the questions from the selected source file.

Every unique source question must be available in the selected dataset before
relabeling can begin. Empty, malformed, or partially unmatched source files block
the session with an explanation. Files from another computer are supported when
their graph IDs, question indices, and complete question snapshots match this
dataset. Changed questions must be restored, and missing images restored and
refreshed. For a corrupt image, restore it and use **Retry image**.

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

Judge these independently, following the model validator prompts. Each question
has a detailed tooltip explaining the evidence, examples, and edge cases:

1. **Is the answer already contained in the question?** Yes fails visual validity;
   No passes. Consider direct answers and leaked observations or intermediate values.
2. **Does the chart type (design) support the type of question being asked?**
   Rate from **1** (does not support the task) to **5** (fully supports it).
   This rating is saved for analysis and does not affect either flag.
3. **Can the answer actually be read from the chart?** Yes passes visual validity;
   No fails. An LLM must be able to recover the required evidence from the supplied
   image, question, and sanitized context. Reasonable estimates and arithmetic count;
   hidden observations or unsupported precision do not.
4. **Is the chart even necessary to answer the question?** Yes passes visual
   validity; No fails. Hide the image: wording, context, or general knowledge alone
   must not supply the answer.
5. **Is the answer correct?** Yes passes data validity; No fails. Check both the
   answer and its explanation, including calculations, comparisons, units, and scaling.
6. **If a value is provided in the question, is that value correct?** Yes passes
   data validity; No fails. Choose Yes if there are no factual values to verify.
   A hypothetical threshold is not an assertion about an observed chart value.
7. **Does answering the question require non-general external knowledge?** Yes
   fails visual validity; No passes. Everyday meanings are allowed; necessary
   specialist definitions or rules (such as interpreting PCA) must be supplied in
   the allowed inputs.

All questions except question 2 use **Yes**, **No**, **Unsure**, and **Partly**.
Nothing is preselected. Answer the wording literally; Yes is unfavorable for
questions 1 and 7. For each flag, any failing check yields `false`. Otherwise,
any Partly yields `"partly"`, then any Unsure yields JSON `null`; all passing
checks yield `true`. Visual validity combines questions **1, 3, 4, 7**; data
validity combines **5, 6**. A definite failure overrides Partly and Unsure.

The app displays the chart, question, proposed answer, explanation, and
sanitized dataset context. Use **Unsure** when this evidence cannot establish
data validity. Model judgments and reasons stay hidden until you click
**Display validity**. The log preserves the labels selected before the reveal
as well as your final labels and all seven choices, so changes after viewing model feedback can be
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
and annotator. Metrics use the derived flags described above. Human **Partly**,
**Unsure**, and missing model judgments are excluded from binary metrics per
axis; **Partly** is counted separately from **Unsure**. Undefined metrics display
**N/A**. Metrics describe saved reviews, which may include earlier stratified
samples; check their logged sampling strategies before estimating full-dataset performance.
Final labels can be influenced by revealed model feedback; the section also
reports how many reviews recorded a reveal before saving.

## Sampling and resuming

All remaining questions in the selected dataset are shuffled uniformly at
random, without replacement or stratification. Every remaining question has
the same chance of appearing next, regardless of the model's validity flags
or which chart it belongs to. Questions from the same chart may appear consecutively.
Using **Different question from same graph** instead selects the first remaining
question for that chart in the shuffled queue, uniformly among its remaining
questions. Other queued questions retain their relative order. Saved questions
and charts skipped for unreadable images are excluded from both navigation
actions. Skipped questions do not count toward review coverage.
Model validity combinations are retained only for statistics and audit metadata;
they do not affect the question order. Visual flags use `vlisual_valid`, then
`visual_valid`, then legacy `valid`. A missing `data_valid` stays unknown and is
never inferred from the visual flag.

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
- `human_labels.data_valid` and `human_labels.visual_valid`: derived `true`,
  `false`, `null`, or `"partly"` flags using the aggregation above;
- `human_choices`: `answer_in_question`, `chart_design_support` (integer 1–5),
  `answer_readable`, `chart_necessary`, `answer_correct`,
  `question_value_correct`, and `external_knowledge_required`. Older records
  retain their original two-check choices and remain readable;
- `model_labels` with separate normalized flags, the original visual flag field
  name, and the model's reasons when supplied;
- `model_feedback_displayed`, `validity_displayed_at`, `labels_before_reveal`,
  `labels_changed_after_reveal`, `choices_before_reveal`, and
  `choices_changed_after_reveal` (new records use schema version 5). Choice
  changes are tracked even when the derived flags stay the same;
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
joins and audits. New reviews log `uniform_random_without_replacement` as the
sampling strategy for normal queue navigation. Questions selected with
**Different question from same graph** instead log
`uniform_random_same_graph_without_replacement`. Logs also record the selection
action and count of questions skipped in the session; manually selected or
skipped reviews can affect the representativeness of the saved sample.
Earlier reviews retain their original sampling metadata;
earlier stratified samples may overrepresent rare model validity combinations.

Writes are locked across processes and flushed to disk before the next question
appears. Malformed response rows block new submissions until repaired,
preserving the existing file.

Relabeling output hashes include the dataset directory, current annotator ID,
and SHA-256 of the complete source file. Changing the source file's contents
starts a different pass; keep the source unchanged to resume existing work.
Each new review includes `relabel_source` with the source filename, file SHA-256,
original question key, annotation ID, and annotator ID (when present). Its sampling
strategy is `source_file_order`. Repeated source questions are reviewed once,
using their first occurrence. The normal response format and save behavior also
apply to relabeling.

## Checks

```powershell
.venv/Scripts/python.exe -m unittest discover -s question-label/tests -v
```
