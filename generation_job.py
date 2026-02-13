import json, pandas as pd, numpy as np, matplotlib.pyplot as plt
from langchain_openai import ChatOpenAI
from sklearn import datasets
import os
from urllib.parse import urlencode
from urllib.request import urlopen, Request
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import base64
import openml
import base64
import json
from langchain_core.messages import HumanMessage
import time
from tqdm import tqdm
import uuid

from sklearn.datasets import fetch_openml

OPENML_LIST_URL = "https://www.openml.org/api/v1/json/data"
rng = np.random.default_rng(67)

llm = ChatOpenAI(
    model="qwen3vl",   # name doesn't matter much; vLLM lists it via /v1/models
    openai_api_key="EMPTY",  # required but ignored by vLLM
    openai_api_base="http://localhost:8000/v1"
)

# DATASET HELPPERS ----------------------------------------------------------------

def openml_list_uci(status="active"):
    """
    Returns a list of dataset metadata dicts from OpenML, filtered by tag='uci',
    using the OpenML *Python client* (no raw REST calls).

    Notes:
    - openml.datasets.list_datasets returns a dict keyed by dataset_id.
    - The OpenML Python client does not expose a true server-side offset the same
      way the REST endpoint does. We emulate offset/limit by slicing locally.
    """
    ds_dict = openml.datasets.list_datasets(tag="uci", status=status)

    # Deterministic ordering for paging
    all_ids = sorted(ds_dict.keys())

    # Return list of metadata dicts (including dataset id)
    datasets = []
    for did in all_ids:
        d = dict(ds_dict[did])  # copy
        d["did"] = int(did)     # mimic REST field name you used
        datasets.append(d)

    return datasets


def quality_to_dict(d):
    """
    Convert OpenML qualities into a dict.

    With the Python client:
    - list_datasets already returns qualities as flat keys (e.g. 'NumberOfInstances')
      rather than a list of {'name','value'} dicts.
    - If the input is from get_dataset(...).qualities it is also already a dict.

    This function keeps compatibility and just returns a dict view.
    """
    if d is None:
        return {}

    # If it already looks like a dict of qualities, return it
    if isinstance(d, dict):
        return d

    return {}


def pick_random_dataset_id(datasets, rng, min_instances=200, max_features=2000):
    """
    Pick a dataset at random from the given metadata list (from openml_list_uci),
    filtered by size constraints.
    """
    filtered = []

    for d in datasets:
        try:
            did = int(d["did"])

            # With list_datasets, these are typically direct keys
            n = int(float(d.get("NumberOfInstances", 0)))
            p = int(float(d.get("NumberOfFeatures", 0)))
        except Exception:
            continue

        if n >= min_instances and 1 <= p <= max_features:
            filtered.append(did)

    if not filtered:
        raise RuntimeError("No datasets passed the filters. Relax constraints.")

    return int(rng.choice(filtered))

def get_dataset_semantics(did, sleep_s=0.0):
    """
    Get dataset semantics using the OpenML Python client:
    - name
    - description
    - feature schema (name, data_type, is_target)
    """
    ds = openml.datasets.get_dataset(int(did))

    if sleep_s and sleep_s > 0:
        time.sleep(sleep_s)

    features = []
    for _, f in sorted(ds.features.items(), key=lambda kv: kv[0]):
        # f is usually an OpenMLDataFeature (attribute-based),
        # but keep a fallback for dict-like cases.
        name = getattr(f, "name", None) if not isinstance(f, dict) else f.get("name")
        data_type = getattr(f, "data_type", None) if not isinstance(f, dict) else f.get("data_type")
        is_target = getattr(f, "is_target", False) if not isinstance(f, dict) else f.get("is_target", False)

        features.append({
            "name": name,
            "data_type": data_type,
            "is_target": bool(is_target),
        })

    return {
        "id": int(did),
        "name": ds.name,
        "description": ds.description,
        "features": features,
    }

def get_random_ds(d_meta, rng):
    """
    Pick a random UCI dataset from metadata (list of dicts),
    ensure ARFF format, and load the data as a pandas DataFrame.
    """

    # Pick dataset id using existing helper
    data_id = pick_random_dataset_id(d_meta, rng=rng)

    # Find corresponding metadata entry (d_meta is a list)
    meta = next(d for d in d_meta if int(d["did"]) == int(data_id))

    # Enforce ARFF format
    fmt = str(meta.get("format", "")).upper()
    if fmt != "ARFF":
        raise RuntimeError(f"Picked dataset {data_id} is not in ARFF format (got {fmt}).")

    # Load dataset via OpenML client
    ds = openml.datasets.get_dataset(data_id)

    X, y, categorical_indicator, attribute_names = ds.get_data(
        dataset_format="dataframe"
    )

    # Match old behavior: return full data including target column
    if y is not None:
        data = X.copy()
        data[ds.default_target_attribute] = y
    else:
        data = X

    return data_id, data


# LLM CALLS ---------------------------------------------------------------------------------------------------

def after_think(text: str) -> str:
    return text.split("</think>", 1)[1] if "<think>" in text else text


def determine_dataset_call(metadata) -> dict:
    prompt = (
        "You are a data expert.\n"
        "You receive metadata about a dataset from OpenML in JSON format.\n\n"
        "Tasks:\n"
        "- Determine whether the dataset be used for generating visualizations which make semantic and visual sense, please be very strict here we do not want to generate semantically uninformative visuals.\n"
        "- Parse the description of the dataset (remove the authors and other non data related information) and its features into a readable format (strictly a string)\n"
        "Output format (STRICT):\n"
        "- Return ONLY a valid JSON object with EXACTLY these keys:\n"
        "  • \"useful\": true or false\n"
        "  • \"description\": string (description of dataset)\n"
        f"DATASET_METADATA:\n{json.dumps(metadata, ensure_ascii=False)}"
    )

    out = llm.invoke(prompt).content
    out = after_think(out)
    
    desc = json.loads(out.replace("```json", "").replace("```", ""))

    return desc

def graphs_call(features: dict, dataset_description: str) -> dict:
    prompt = (
        "You are a data visualization expert.\n"
        "You receive the head of a dataset in JSON format along with feature metadata.\n\n"
        "You also receive a description of the dataset.\n\n"
        "Task:\n"
        "Generate EXACTLY 10 different plot specifications that could be created from this dataset.\n\n"
        "Rules:\n"
        "- Each plot must be semantically valid given the provided features.\n"
        "- Include BOTH basic plots AND more advanced or complex plots\n"
        "- The plots should be similar to what human scientists would create for the given dataset and it's description, not just random plots.\n"
        "- You are encouraged to be creative with the plot types.\n"
        "- You should sometimes include multiple subplots or faceted plots to show more complex relationships or compare different classes (e.g 2 subplots layered horizontally or vertically.)\n"
        "- Do NOT include plots that would be extremely hard to read and don't make sense semantically (e.g., a bar plot with many tiny bars, a line plot of very scattered data, etc.)\n"
        "- Do NOT repeat plot types.\n"
        "- Do NOT generate code.\n"
        "- Do NOT describe the plots.\n\n"
        "Output format (STRICT):\n"
        "- Return ONLY a valid JSON array.\n"
        "- Each element must be a JSON object with EXACTLY these keys:\n"
        "  • \"type\": string (name of the plot type)\n"
        "  • \"features\": array of strings (feature names from the dataset used for this plot)\n"
        "  • \"description\": somewhat detailed description of what the plot is supposed to show, both semantically and visually\n"
        "- The listed features must exist in the provided FEATURES section.\n"
        "- Use all feature names EXACTLY as given.\n"
        "- No additional keys, comments, or text.\n\n"
        "- Consider that the person graphing can only use numpy, pandas, matplotlib, scikit-learn, and default python libraries, nothing else"
        f"FEATURES:\n{json.dumps(features, ensure_ascii=False)}\n"
        f"DATASET DESCRIPTION:\n{dataset_description}\n"
    )

    out = llm.invoke(prompt).content
    out = after_think(out)
    out = out[out.find("["): out.rfind("]") + 1]
    
    spec = json.loads(out)

    return spec

def graph_call(features, selected_plot, head) -> dict:
    prompt = (
        "You are a plot rendering agent.\n"
        "You are given:\n"
        "1) A pandas DataFrame named `df`.\n"
        "2) A JSON object named `selected_plot` that was produced by a previous model call.\n\n"
        "selected_plot has exactly these keys:\n"
        "  - \"type\": the required plot type to render\n"
        "  - \"features\": the exact list of column names that must be used for the plot\n"
        "  - \"style\": the matplotlib style to use for rendering\n\n"
        "Your job:\n"
        "- Render EXACTLY ONE plot whose plot type matches selected_plot[\"type\"].\n"
        f"- Save that plot with plt.savefig(), the path will be available in a variable named 'graph_file_path'\n"
        "- Use ONLY the columns listed in selected_plot[\"features\"].\n"
        "- You may derive temporary helper columns ONLY from those listed features "
        "(e.g., binning a numeric feature, extracting month from a datetime feature), "
        "but you must not use any other df columns.\n"
        "- You may replace feature names in titles or labels with clearer semantic equivalents when their meaning can be reliably inferred.\n\n"
        "- If there are outliers that would worsen the quality of the graph you can remove them."
        "Libraries:\n"
        "- Use ONLY pandas, numpy, matplotlib, scikit-learn and default python libraries. Do NOT use pandas plotting options always use matplotlib directly.\n\n"
        "CRITICAL: After the code runs, define BOTH:\n"
        "1) A pandas DataFrame named `graph_df` containing the FINAL PROCESSED DATA actually used for plotting\n"
        "   (after all filtering, aggregation, binning, and transformations).\n"
        "2) A JSON-serializable dict named `graph_data` with EXACTLY these keys (all keys required):\n\n"
        "graph_data = {\n"
        "  \"plot_type\": string,                     # must equal selected_plot[\"type\"]\n"
        "  \"features_expected\": list[str],          # must equal selected_plot[\"features\"] exactly\n"
        "  \"features_used\": list[str],              # columns actually used (include derived names if created)\n"
        "  \"derived_features\": list[str],           # names of any derived helper columns you create\n"
        "  \"x\": string or null or array of values if multiple subplots,\n"
        "  \"y\": string or null or array of values if multiple subplots,\n"
        "  \"hue\": string or null or array of values if multiple subplots,\n"
        "  \"facet\": string or null\n"
        "  \"aggregation\": string or null or array of values if multiple subplots,\n"
        "  \"binning\": string or null or array of values if multiple subplots,\n"
        "  \"transformations\": list[str] or array of values if multiple subplots,\n"
        "  \"filters\": list[str],\n"
        "  \"n_rows_input\": int\n"
        "  \"n_rows_plotted\": int,                   # MUST equal len(graph_df)\n"
        "  \"title\": string\n"
        "}\n\n"
        "Validation rules:\n"
        "- `graph_df` must contain ONLY columns listed in `features_used`.\n"
        "- `graph_df` must reflect EXACTLY what is plotted (no extra rows or columns).\n"
        "- If any feature in selected_plot[\"features\"] is missing from df.columns, pick a different plot approach "
        "that still matches selected_plot[\"type\"] but uses the remaining provided features only; "
        "ALWAYS keep graph_data[\"features_expected\"] unchanged.\n"
        "- Do NOT change selected_plot[\"type\"].\n"
        "- Do NOT use columns outside selected_plot[\"features\"].\n"
        "- Do NOT print anything.\n"
        "- Output ONLY MINIMAL executable Python code.\n\n"
        "- You should be careful to properly space elements in graphs to avoid overlap and improve readability.\n"
        "- For radablility purposes you NEED TO make the figures large enough to fit all the elements in the plots.\n"
        "- The use of plt.tight_layout() is encouraged to improve spacing. You should make figures that have multiple subplots bigger so they are readable.\n"
        "Inputs you must rely on:\n"
        f"selected_plot = {json.dumps(selected_plot, ensure_ascii=False)}\n\n"
        f"FEATURES_METADATA:\n{json.dumps(features, ensure_ascii=False)}\n"
        f"HEAD:\n {json.dumps(head)}"
    )
    out = llm.invoke(prompt).content
    out = after_think(out)

    code = out.replace("```python", "").replace("```", "")

    return code

def check_call(image_path: str, plot_code: str) -> str:
    review_prompt = (
        "You are a visualization QA reviewer.\n"
        "\n"
        "You will be given:\n"
        "1) An IMAGE of a chart (the rendered plot).\n"
        "2) The PYTHON CODE that generated the chart.\n"
        "\n"
        "Goal:\n"
        "Provide practical feedback on readability and distinguishability. Be helpful, not overly strict.\n"
        "Focus on whether someone can interpret the plot correctly at a glance.\n"
        "Be very careful with your feedback as wrong feedback can severey impact goodness of graphs. If there are no sever mistakes just set the correction key to false"
        "\n"
        "What to check (prioritize):\n"
        "- Visibility: text size, tick labels, title, axis labels, legend readability.\n"
        "- Overlap/clutter: label collisions, dense points, overlapping bars/lines, crowded legend.\n"
        "- Distinguishability: lines/markers too similar, categories hard to tell apart, missing/unclear legend.\n"
        "- Scaling/layout: axes limits, aspect ratio, too much empty space, cut-off labels, rotated ticks.\n"
        "- Data-ink issues: overplotting, too many categories, need aggregation/binning/faceting.\n"
        "- Accessibility basics: ensure contrast is adequate and the plot doesn’t rely on subtle differences only.\n"
        "\n"
        "Output requirements (STRICT):\n"
        "- Output a JSON with keys:\n"
        "  - feedback: the feedback of the graph, this is strictly only a string.\n"
        "  - correction: true or false based on whether you think the mistakes in the graph are large enough to warrant re-generation of the plot\n"
        "- Keep it very concise, return only faults of the graph and no positive feedback or explanations.\n"
        "- Each bullet MUST include:\n"
        "  (a) the issue (if any),\n"
        "  (b) a concrete fix in text\n"
        "- Do NOT nitpick minor stylistic preferences.\n"
        "- Do NOT propose changing the plot type.\n"
    )

    # Load and base64-encode the image
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    # Multimodal message (works with LangChain OpenAI-compatible chat models that support vision)
    msg = HumanMessage(
        content=[
            {"type": "text", "text": review_prompt},
            {"type": "text", "text": f"CODE:\n{plot_code}"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
        ]
    )

    # Invoke the LLM
    resp = llm.invoke([msg])

    out = resp.content
    out = after_think(out)

    feedback = json.loads(out.replace("```json", "").replace("```", ""))

    return feedback

def recode_call(features, selected_plot, previous_code, corrections, head) -> dict:
    prompt = (
        "You are a plot rendering agent.\n"
        "You are given:\n"
        "1) A pandas DataFrame named `df`.\n"
        "2) A JSON object named `selected_plot` that was produced by a previous model call.\n"
        "3) OPTIONAL: `previous_code` (the code used previously).\n"
        "4) OPTIONAL: `corrections` (feedback describing what to fix).\n\n"
        "selected_plot has exactly these keys:\n"
        "  - \"type\": the required plot type to render\n"
        "  - \"features\": the exact list of column names that must be used for the plot\n"
        "  - \"style\": the matplotlib style to use for rendering\n\n"
        "Your job:\n"
        "- Render EXACTLY ONE plot whose plot type matches selected_plot[\"type\"].\n"
        f"- Save that plot with plt.savefig(), the path will be available in a variable named 'graph_file_path'\n"
        "- Use ONLY the columns listed in selected_plot[\"features\"].\n"
        "- You may derive temporary helper columns ONLY from those listed features "
        "(e.g., binning a numeric feature, extracting month from a datetime feature), "
        "but you must not use any other df columns.\n"
        "- You may replace feature names in titles or labels with clearer semantic equivalents when their meaning can be reliably inferred.\n\n"
        "If `previous_code` and `corrections` are provided:\n"
        "- Start from `previous_code` and apply ONLY the requested corrections.\n"
        "- Do NOT change the plot type.\n"
        "- Do NOT change which data is plotted (no changes to filters/aggregation/binning/transformations) unless the corrections explicitly require it.\n"
        "- Prefer purely visual/layout fixes (figure size, fonts, rotation, legend placement, alpha, linewidth, markers, margins).\n\n"
        "Libraries:\n"
        "- Use ONLY pandas, numpy, matplotlib, scikit-learn and default python libraries. Do NOT use pandas plotting options; use matplotlib directly.\n\n"
        "CRITICAL: After the code runs, define BOTH:\n"
        "1) A pandas DataFrame named `graph_df` containing the FINAL PROCESSED DATA actually used for plotting\n"
        "   (after all filtering, aggregation, binning, and transformations).\n"
        "2) A JSON-serializable dict named `graph_data` with EXACTLY these keys (all keys required):\n\n"
        "graph_data = {\n"
        "  \"plot_type\": string,                     # must equal selected_plot[\"type\"]\n"
        "  \"features_expected\": list[str],          # must equal selected_plot[\"features\"] exactly\n"
        "  \"features_used\": list[str],              # columns actually used (include derived names if created)\n"
        "  \"derived_features\": list[str],           # names of any derived helper columns you create\n"
        "  \"x\": string or null,\n"
        "  \"y\": string or null,\n"
        "  \"hue\": string or null,\n"
        "  \"facet\": string or null,\n"
        "  \"aggregation\": string or null,\n"
        "  \"binning\": string or null,\n"
        "  \"transformations\": list[str],\n"
        "  \"filters\": list[str],\n"
        "  \"n_rows_input\": int,\n"
        "  \"n_rows_plotted\": int,                   # MUST equal len(graph_df)\n"
        "  \"title\": string\n"
        "}\n\n"
        "Validation rules:\n"
        "- `graph_df` must contain ONLY columns listed in `features_used`.\n"
        "- `graph_df` must reflect EXACTLY what is plotted (no extra rows or columns).\n"
        "- If any feature in selected_plot[\"features\"] is missing from df.columns, pick a different plot approach "
        "that still matches selected_plot[\"type\"] but uses the remaining provided features only; "
        "ALWAYS keep graph_data[\"features_expected\"] unchanged.\n"
        "- Do NOT use columns outside selected_plot[\"features\"].\n"
        "- Do NOT print anything.\n"
        "- Output ONLY MINIMAL executable Python code.\n\n"
        "Inputs you must rely on:\n"
        f"selected_plot = {json.dumps(selected_plot, ensure_ascii=False)}\n\n"
        f"FEATURES_METADATA:\n{json.dumps(features, ensure_ascii=False)}\n\n"
        f"HEAD:\n {json.dumps(head)}\n\n"
        f"previous_code = {json.dumps(previous_code or '', ensure_ascii=False)}\n\n"
        f"corrections = {json.dumps(corrections or '', ensure_ascii=False)}\n"
    )

    out = llm.invoke(prompt).content
    out = after_think(out)

    code = out.replace("```python", "").replace("```", "")

    return code

def describe_graph_png(png_path, plot_code, graph_data, graph_df, dataset_desc, plot_description) -> str:
    describe_prompt = (
        "You are a meticulous chart description call.\n"
        "\n"
        "You will be given:\n"
        "1) An IMAGE of a chart.\n"
        "2) The PYTHON CODE used to generate it.\n"
        "3) `graph_data`: a JSON-serializable dict describing the plot (authoritative).\n"
        "4) `graph_df`: the FINAL processed pandas DataFrame actually plotted (authoritative).\n"
        "5) A textual DESCRIPTION of the dataset used.\n"
        "6) A textual DESCRIPTION of the plot type and purpose.\n"
        "\n"
        "Goal:\n"
        "Write an extremely detailed description of the chart.\n"
        "Everything you claim MUST be inferable from the image, the provided `graph_data`/`graph_df` AND/OR the dataset and plot descriptions.\n"
        "If something cannot be confidently inferred, explicitly say it is unknown or not determinable.\n"
        "You do NOT have to mention where each fact comes from, but you must NOT invent any details.\n"
        "Do NOT guess.\n"
        "\n"
        "Output rules:\n"
        "- Output ONLY plain text.\n"
        "- Use clear section headers exactly as provided below.\n"
        "- Be very detailed, but never invent values or categories.\n"
        "- When describing numeric ranges, counts, or extrema, compute them from `graph_df` (not from the image).\n"
        "- When describing colors, line styles, marker shapes, layout, and visual structure, use the image.\n"
        "\n"
        "Write the description with these sections (use these headers verbatim):\n"
        "1) Chart type, purpose and semantic meaning\n"
        "2) What is plotted (variables and encodings)\n"
        "3) Data shown (from graph_df)\n"
        "4) Patterns and relationships visible\n"
        "5) Image/visual properties (from the image)\n"
        "6) Caveats and unknowns\n"
        "\n"
        "Section requirements:\n"
        "1) Chart type and purpose\n"
        "- State the chart type (must match graph_data[\"plot_type\"]).\n"
        "- Explain what question this chart helps answer, based on encodings and variables.\n"
        "- Explain the semantic meaning of the chart in the context of the dataset.\n"
        "- Explain what the chart entails in the context of the dataset.\n"
        "\n"
        "2) What is plotted (variables and encodings)\n"
        "- Name x, y, hue, facet exactly from graph_data (if null, state that explicitly).\n"
        "- Describe how each variable is encoded: position, color, marker, line, size, panels.\n"
        "- Mention any aggregation/binning/transformations/filters using graph_data fields.\n"
        "\n"
        "3) Data shown (from graph_df)\n"
        "- Report: number of rows plotted (len(graph_df)).\n"
        "- List the columns present in graph_df.\n"
        "- For each plotted numeric column: min, max, mean, median, and (if relevant) quantiles.\n"
        "- For each plotted categorical column: number of categories, and the top categories by frequency.\n"
        "- If the chart is aggregated (e.g., bars of means/counts), describe the granularity and what each mark represents.\n"
        "  Only state exact numbers if they can be computed from graph_df.\n"
        "\n"
        "4) Patterns and relationships visible\n"
        "- Describe trends, clusters, outliers, group differences, correlations, or distributions.\n"
        "- Do NOT claim a relationship if it cannot be supported by the image or graph_df.\n"
        "\n"
        "5) Image/visual properties (from the image)\n"
        "- Describe layout: orientation, gridlines, axes, tick density, legend placement, title presence.\n"
        "- Describe visual encodings: colors used (approximate names), line widths, marker shapes, alpha/transparency, bar widths.\n"
        "- Describe structure: number of series/lines/bars/panels as visible.\n"
        "- Mention readability aspects visible in the image (overlap, clutter, label rotation).\n"
        "\n"
        "6) Caveats and unknowns\n"
        "- List anything not determinable unless proven by code/image.\n"
        "- If code contradicts the image, treat the image as truth for visuals, and graph_df/graph_data as truth for data.\n"
    )

    # Load PNG
    with open(png_path, "rb") as f:
        png_b64 = base64.b64encode(f.read()).decode("utf-8")

    # Keep graph_df payload small
    graph_df_preview = graph_df.head(25).to_json(orient="records", date_format="iso")

    msg = HumanMessage(
        content=[
            {"type": "text", "text": describe_prompt},
            {"type": "text", "text": f"graph_data:\n{json.dumps(graph_data, ensure_ascii=False)}"},
            {"type": "text", "text": f"graph_df preview (head 25):\n{graph_df_preview}"},
            {"type": "text", "text": f"CODE:\n{plot_code}"},
            {"type": "text", "text": f"DATASET DESCRIPTION:\n{dataset_desc}"},
            {"type": "text", "text": f"PLOT DESCRIPTION:\n{plot_description}"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{png_b64}"}},
        ]
    )

    resp = llm.invoke([msg])
    return resp.content

def generate_graph_questions(png_path: str, dataset_desc: str, plot_desc: str, graph_data: dict) -> list[dict]:
    qa_prompt = (
        "You are a chart QA generator.\n"
        "\n"
        "You will be given:\n"
        "1) An IMAGE of a chart.\n"
        "2) A detailed DESCRIPTION of the chart/plot (authoritative).\n"
        "3) A DESCRIPTION of the dataset/context (authoritative).\n"
        "4) Some structured data of the graph in graph_data. \n"
        "\n"
        "Task:\n"
        "Generate EXACTLY 20 questions about the chart.\n"
        "Include a mix of difficulties:\n"
        "- 8 easy (direct reading: titles, axes, legend, counts, obvious comparisons)\n"
        "- 8 medium (interpretation: comparisons across groups, trends, approximate ranges, notable patterns)\n"
        "- 4 hard (multi-step reasoning grounded in the chart + context, but still definitively answerable)\n"
        "\n"
        "CRITICAL CONSTRAINTS:\n"
        "- Every question MUST be definitively answerable from the provided IMAGE and/or the provided chart/dataset descriptions.\n"
        "- Do NOT ask questions that require external knowledge.\n"
        "- Do NOT ask questions that require more data than what is shown/described.\n"
        "- Do NOT produce questions that are just instructions like 'analyze' or 'explain how'.\n"
        "- Avoid vague questions. Each must have a single, checkable answer.\n"
        "- If exact numeric values are not available, ask questions that accept approximate answers only when the chart clearly supports approximation.\n"
        "\n"
        "Output format (STRICT):\n"
        "Return ONLY valid JSON: an array of exactly 20 objects.\n"
        "Each object must have EXACTLY these keys:\n"
        "{\n"
        "  \"question\": string,\n"
        "  \"answer\": string,             # must be concrete, not instructions\n"
        "  \"answer_basis\": \"image\"|\"description\"|\"both\"  # where the answer comes from\n"
        "}\n"
        "\n"
        "Quality requirements:\n"
        "- Questions should cover BOTH:\n"
        "  (a) chart mechanics/visual properties (axes, legend, encodings, layout), and\n"
        "  (b) semantics in dataset context (what variables represent, what patterns mean).\n"
        "- Do not repeat the same question pattern; vary them.\n"
        "- Questions should be related to the graph and the data in the graph, do not ask general questions about the dataset that do not directly relate to the chart.\n"
        "- Do NOT include any extra text outside the JSON.\n"
    )


    with open(png_path, "rb") as f:
        png_b64 = base64.b64encode(f.read()).decode("utf-8")

    msg = HumanMessage(content=[
        {"type": "text", "text": qa_prompt},
        {"type": "text", "text": f"DATASET DESCRIPTION:\n{dataset_desc}"},
        {"type": "text", "text": f"PLOT DESCRIPTION:\n{plot_desc}"},
        {"type": "text", "text": f"graph_data:\n{json.dumps(graph_data, ensure_ascii=False)}"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{png_b64}"}},
    ])

    resp = llm.invoke([msg]).content

    start = resp.find("[")
    end = resp.rfind("]")
    return json.loads(resp[start:end+1])


if __name__ == "__main__": 
    print("Start generation...")

    DATASET_FOLDER = "dataset"
    IMAGES_FOLDER = os.path.join(DATASET_FOLDER, "images")
    GENERATE_DS_IMAGES = 10
    FEEDBACK = False

    n_files = sum(
        1 for f in os.listdir(IMAGES_FOLDER)
        if os.path.isfile(os.path.join(IMAGES_FOLDER, f))
    )

    datasets_meta = openml_list_uci()

    index = n_files

    """ pbar = tqdm(total=GENERATE_DS_IMAGES*10)
    pbar.update(0) """
    while index < GENERATE_DS_IMAGES*10 + n_files:
        # Get DATASET
        print(f"Fetching random dataset...")
        while True:
            try:
                ds_id, df = get_random_ds(datasets_meta, rng) # Fetch random UCI dataset
                dataset_sem = get_dataset_semantics(ds_id, sleep_s=1.0) # Get dataset semantics

                desc_dataset = determine_dataset_call(dataset_sem) # Determine if dataset is useful and format description

                if not desc_dataset["useful"]:
                    print(f"Dataset {ds_id} deemed not useful, picking another...")
                    continue

                break

            except Exception as e:
                print(f"Error fetching dataset, retrying... {e}")
                continue

        head_json = df.head(5).to_dict(orient="records")

        # Generate GRAPH TYPES
        print(f"Generating graph types for dataset {ds_id}...")
        retr = 0
        while True:
            try:
                graph_types = graphs_call(json.dumps(head_json), dataset_sem["description"])

                templates = plt.style.available
                for i, t in enumerate(graph_types):
                    graph_types[i]["style"] = rng.choice(templates)

                break
            
            except Exception as e:
                retr+=1
                print(f"Error generating graph types, retrying ({retr}) with dataset {ds_id}... {e}")
                continue

        # Generate and RENDER GRAPHS
        i = 0
        rerun = 0
        while i < len(graph_types):
            if rerun > 2:
                rerun = 0
                i += 1
                continue
            matplotlib.rcParams.update(matplotlib.rcParamsDefault)
            plt.style.use("default")
            print(f"Generating graph {i+1}/{len(graph_types)} for dataset {ds_id}...")
            graph_file_path = f"dataset/images/{index}_it0.png"
            selected_plot = graph_types[i]
            try:
                imgs = []
                code = graph_call(dataset_sem.get("features"), graph_types[i], json.dumps(head_json))

                exec_ns = {
                    "df": df,
                    "selected_plot": selected_plot,
                    "graph_file_path": graph_file_path,
                    "__builtins__": __builtins__,
                }

                exec(code, exec_ns, exec_ns)

                graph_data = exec_ns.get("graph_data", None)
                graph_df   = exec_ns.get("graph_df", None)
                
                # CHECK and REGENERATE if needed
                
                if FEEDBACK:
                    regen_count = 0
                    while True:
                        try:
                            feedback = check_call(graph_file_path, code)

                            imgs.append({ 
                                "path": graph_file_path, 
                                "feedback": feedback["feedback"], 
                                "code": code
                            })

                            if feedback["correction"] and regen_count < 3:
                                print(f"Graph {i+1} needs correction, regenerating ({regen_count})...")

                                graph_file_path = f"dataset/images/{index}_it{regen_count + 1}.png"

                                code = recode_call(dataset_sem.get("features"), graph_types[i], code, feedback["feedback"], json.dumps(head_json))

                                exec_ns = {
                                    "df": df,
                                    "selected_plot": selected_plot,
                                    "graph_file_path": graph_file_path,
                                    "__builtins__": __builtins__,
                                }

                                exec(code, exec_ns, exec_ns)

                                graph_data = exec_ns.get("graph_data", None)
                                graph_df   = exec_ns.get("graph_df", None)


                                regen_count += 1
                                imgs.append(graph_file_path)
                            else:
                                imgs.append({ 
                                    "path": graph_file_path,
                                    "feedback": feedback["feedback"],
                                    "code": code
                                })
                                break
                            break
                        except Exception as e:
                            print(f"Error during graph checking, retrying... {e}")
                            graph_file_path = f"dataset/images/{index}_it{regen_count}.png"
                            regen_count += 1
                            continue
                else:
                    imgs.append({ 
                        "path":graph_file_path, 
                        "feedback": None,
                        "code": code
                    })

                # DESCRIBE GRAPH
                graph_data = graph_data  # from executed code   
                graph_df = graph_df  # from executed code

                description = describe_graph_png(imgs[-1]["path"], code, graph_data, graph_df, desc_dataset["description"], graph_types[i]["description"])

                questions = generate_graph_questions(imgs[-1]["path"], desc_dataset["description"], description, graph_data)

                with open(os.path.join(DATASET_FOLDER, "metadata.jsonl"), "a", encoding="utf-8") as f:
                    obj = {
                        "id": str(uuid.uuid4()),
                        "dataset": {
                            "id": ds_id,
                            "description": desc_dataset["description"],
                        },
                        "graph":{
                            "type": graph_types[i]["type"],
                            "style": graph_types[i]["style"],
                            "full_description": description,
                            "short_description": graph_types[i]["description"],
                            "code": code,
                            "structured_data": graph_data,
                            "questions": questions
                        },
                        "images": imgs
                    }
                    f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                
                index += 1
                i += 1
                """ pbar.update(1) """
                rerun = 0

            except Exception as e:
                print(f"Error generating graph, retrying... {e}")
                rerun += 1
                continue
