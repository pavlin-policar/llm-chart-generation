from langchain_core.messages import HumanMessage

from helpers import after_think, strip_code_fences

import json
import re
import numpy as np
import base64

def determine_dataset_call(llm, metadata) -> dict: # No reasoning
    """
    Calls LLM -> tells us whether the datataset is useful for creating visualizations. 
    It also formats the description.
    """

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
    _, out = after_think(out)
    
    desc = json.loads(out.replace("```json", "").replace("```", ""))

    return desc

def graphs_call(llm, features: dict, dataset_description: str, num_graphs) -> dict: # Reasoning
    """
    Calls LLM -> returns 10 specifications for 10 graphs that could be made from this dataset. 
    The specifications consist of graph type, short description, and features that should be used.

    LLM gets a random creativity rating that encourages it to produce more standard or more unusual graphs.
    """

    creat = np.random.beta(2, 4, size=1)[0]

    #TODO: implement amount of graphs chosen based on dataset

    prompt = (
        "You are a data visualization expert.\n"
        "You receive the head of a dataset in JSON format along with feature metadata.\n\n"
        "You also receive a description of the dataset.\n\n"
        "Task:\n"
        f"Generate EXACTLY {num_graphs} different plot specifications that could be created from this dataset.\n\n"
        "Rules:\n"
        "- Each plot must be semantically valid given the provided features.\n"
        "- Include BOTH basic plots AND more advanced or complex plots\n"
        "- The plots should be similar to what human scientists would create for the given dataset and it's description, not just random plots.\n"
        "- You are encouraged to be creative with the plot types.\n"
        "- Examples (you are NOT limited to this) of creative plot types Circos plot, Sankey diagram, Chord diagram, Sunburst chart, Treemap, Radar chart, Streamgraph, Parallel coordinates plot, Network graph, Heatmap, Violin plot, Ridgeline plot, Hexbin plot, Contour plot, Bubble chart, Alluvial diagram, Marimekko chart, Waterfall chart, Funnel chart, Polar area chart, Nightingale rose chart, Voronoi diagram, Dendrogram, Icicle chart, Bump chart, Lollipop chart, Dot matrix chart, Packed bubble chart, Arc diagram, Gantt chart"
        "- You should sometimes include multiple subplots or faceted plots to show more complex relationships or compare different classes (e.g 2 subplots layered horizontally or vertically.)\n"
        "- Do NOT include plots that would be extremely hard to read and don't make sense semantically (e.g., a bar plot with many tiny bars, a line plot of very scattered data, etc.)\n"
        "- Do NOT repeat plot types.\n"
        "- Do NOT generate code.\n"
        "- Do NOT describe the plots.\n\n"
        "- DO NOT propose plots with many different subplots. It should contain a maximum of 5 subplots per row and 5 per column."
        "Output format (STRICT):\n"
        "- Return ONLY a valid JSON array.\n"
        "- Each element must be a JSON object with EXACTLY these keys:\n"
        "  • \"type\": string (name of the plot type)\n"
        "  • \"features\": array of strings (feature names from the dataset used for this plot)\n"
        "  • \"description\": somewhat detailed description of what the plot is supposed to show, both semantically and visually\n"
        "- The listed features must exist in the provided FEATURES section.\n"
        "- Use all feature names EXACTLY as given.\n"
        "- No additional keys, comments, or text.\n\n"
        "- Consider that the person graphing can only use numpy, pandas, matplotlib, scikit-learn, and default python libraries, nothing else."
        f"You are given a creativity level: {creat:.02f} on a scale from 0 to 1.\n"
        "Interpret this as:\n"
        "- 0.0 = choose the most standard and safest graph type\n"
        "- 0.5 = allow moderately uncommon but still clear graph choices\n"
        "- 1.0 = prefer more novel but still valid and interpretable graph choices\n\n"
        f"FEATURES:\n{json.dumps(features, ensure_ascii=False)}\n"
        f"DATASET DESCRIPTION:\n{dataset_description}\n"
    )

    out = llm.invoke(prompt).content
    _, out = after_think(out)
    out = out[out.find("["): out.rfind("]") + 1]
    
    spec = json.loads(out)

    return spec

def replace_vars_call(llm, features: dict, dataset_description: str) -> dict: # No reasoning
    """
    Calls LLM -> replaces feature names in the dataset with a more semantically meaningful equaivalent.
    """

    prompt = (
        "You are a data expert specializing in dataset interpretation.\n\n"
        "Your task is to rename feature names so they are more semantically meaningful"
        "based ONLY on the dataset description provided.\n\n"
        "Feature names should be renamed so they make sense if they are presented without dataset description or as a label of an axis on a graph."
        "Rules:\n"
        "1. Do NOT add or remove any features.\n"
        "2. The number of returned feature names MUST exactly match the number of input features.\n"
        "3. Keep the original order.\n"
        "4. If you cannot confidently infer a better semantic name, return the original name unchanged.\n"
        "5. Return ONLY a valid array of strings.\n\n"
        f"INPUT FEATURES:\n{json.dumps(features, ensure_ascii=False)}\n\n"
        f"DATASET DESCRIPTION:\n{dataset_description}\n"
    )

    out = llm.invoke(prompt).content
    _, out = after_think(out)
    
    desc = json.loads(out.replace("```json", "").replace("```", ""))

    return desc

def compute_info_call(llm, features, selected_plot, head): # No reasoning
    """
    Calls LLM -> returns detailed instructions on how to make a specified plot.

    NOTE: This method is not used anymore and is kept here just in case.
    """

    prompt = (
        "You are a senior data visualization expert.\n"
        "You receive information about:\n"
        "- A graph type\n"
        "- The graph name\n"
        "- The dataset feature names\n"
        "- The dataset feature types (numerical, categorical, ordinal, binary, datetime, etc.)\n\n"
        "Your task:\n"
        "- Provide clear and practical instructions for a coding agent that will generate this plot.\n"
        "- Focus ONLY on what the coding agent must watch out for to ensure the plot is readable, correct, and visually meaningful.\n"
        "- Do NOT write code.\n"
        "- Do NOT explain theory.\n"
        "- Do NOT restate the inputs.\n\n"
        "The instructions should include considerations such as:\n"
        "- Axis selection and scaling\n"
        "- Handling categorical vs numerical features\n"
        "- Label clarity and rotation if needed\n"
        "- Dealing with skewed distributions\n"
        "- Overplotting and transparency\n"
        "- Sorting categories when appropriate\n"
        "- Aggregation requirements (if necessary)\n"
        "- Log-scaling if appropriate\n"
        "- Color usage and legend clarity\n"
        "- Handling missing values\n"
        "- Ensuring the title reflects the actual data mapping\n\n"
        "Be concise but thorough.\n"
        "Return ONLY a structured bullet-point list of actionable instructions. Also keep it very brief.\n\n"
        f"INPUT:\n"
        f"selected_plot = {json.dumps(selected_plot, ensure_ascii=False)}\n\n"
        f"FEATURES_METADATA:\n{json.dumps(features, ensure_ascii=False)}\n"
        f"HEAD:\n {json.dumps(head)}"
    )

    out = llm.invoke(prompt).content
    _, out = after_think(out)

    return out

def plan_call(llm, features, selected_plot) -> str:
    plan_prompt = (
        "You are a plotting planner.\n"
        "You will be given:\n"
        "- selected_plot (type, required features list, matplotlib style)\n"
        "- FEATURES_METADATA (column types/semantics)\n"
        "- HEAD (first rows)\n\n"
        "Output ONLY a single JSON object (no code, no extra text).\n"
        "JSON schema (all keys required):\n"
        "{\n"
        "  \"notes\": string"
        "}\n\n"
        "Rules:\n"
        "- Give the plotting agent some notes about how to proceed with the writing of the code and what to watch out for."
        "- Keep notes concise but not too short. (100-150 words)\n\n"
        f"selected_plot = {json.dumps(selected_plot, ensure_ascii=False)}\n\n"
        f"FEATURES_METADATA:\n{json.dumps(features, ensure_ascii=False)}\n\n"
    )

    plan_raw = llm.invoke(plan_prompt).content
    _, plan_raw = after_think(plan_raw)
    plan = strip_code_fences(plan_raw)

    return plan


def graph_call(llm, features, selected_plot, plan) -> str: # No reasoning plan, reasoning code
    """
    Calls LLM -> generates the code needed to plot the graph.

    It uses a planning step before the actual call, since this seems to improve the generation.
    """

    code_prompt = (
        "You are a plot rendering agent.\n"
        "You are given:\n"
        "1) A pandas DataFrame named `df`.\n"
        "2) A JSON object named `selected_plot` that was produced by a previous model call.\n"
        "selected_plot has exactly these keys:\n"
        "  - \"type\": the required plot type to render\n"
        "  - \"features\": the exact list of column names that must be used for the plot\n"
        "Your job:\n"
        "- Render EXACTLY ONE plot whose plot type matches selected_plot[\"type\"].\n"
        "- Follow `plan` for x/y/hue/facet/aggregation/binning/filters/figsize/title.\n"
        "- Save that plot with plt.savefig(), the path will be available in a variable named 'graph_file_path', use this variable but don't change it.\n"
        "- Use ONLY the columns listed in selected_plot[\"features\"].\n"
        "- You may derive temporary helper columns ONLY from those listed features.\n\n"
        "Libraries:\n"
        "- Use ONLY pandas, numpy, matplotlib, scikit-learn and default python libraries. Do NOT use seaborn!\n"
        "- Do NOT use pandas plotting; always use matplotlib directly.\n\n"
        "CRITICAL: After the code runs, define BOTH:\n"
        "1) A pandas DataFrame named `graph_df` containing the FINAL PROCESSED DATA actually used for plotting.\n"
        "2) A JSON-serializable dict named `graph_data` with EXACTLY these keys (all keys required):\n\n"
        "graph_data = {\n"
        "  \"plot_type\": string,\n"
        "  \"features_expected\": list[str],\n"
        "  \"features_used\": list[str],\n"
        "  \"derived_features\": list[str],\n"
        "  \"x\": string or null or array of values if multiple subplots,\n"
        "  \"y\": string or null or array of values if multiple subplots,\n"
        "  \"hue\": string or null or array of values if multiple subplots,\n"
        "  \"facet\": string or null,\n"
        "  \"aggregation\": string or null or array of values if multiple subplots,\n"
        "  \"binning\": string or null or array of values if multiple subplots,\n"
        "  \"transformations\": list[str] or array of values if multiple subplots,\n"
        "  \"filters\": list[str],\n"
        "  \"n_rows_input\": int,\n"
        "  \"n_rows_plotted\": int,\n"
        "  \"title\": string\n"
        "}\n\n"
        "Validation rules:\n"
        "- `graph_df` must contain ONLY columns listed in `features_used`.\n"
        "- `graph_df` must reflect EXACTLY what is plotted (no extra rows/cols).\n"
        "- If any feature in selected_plot[\"features\"] is missing from df.columns, pick a different plot approach\n"
        "  that still matches selected_plot[\"type\"] but uses the remaining provided features only;\n"
        "  ALWAYS keep graph_data[\"features_expected\"] unchanged.\n"
        "- Do NOT change selected_plot[\"type\"].\n"
        "- Do NOT use columns outside selected_plot[\"features\"].\n"
        "- Do NOT print anything.\n"
        "- Output ONLY MINIMAL executable Python code.\n"
        "- Use large enough figures; use plt.tight_layout().\n\n"
        "Inputs you must rely on:\n"
        f"selected_plot = {json.dumps(selected_plot, ensure_ascii=False)}\n\n"
        f"FEATURES_METADATA:\n{json.dumps(features, ensure_ascii=False)}\n\n"
        f"HEAD:\n{json.dumps(head, ensure_ascii=False)}\n"
        f"PLAN from planning agent: {json.dumps(plan, ensure_ascii=False)}\n" if plan is not None else ""
    )

    out = llm.invoke(code_prompt).content
    try:
        _, out = after_think(out)
    except Exception:
        pass

    code = strip_code_fences(out)
    return code

def check_call(llm, image_path: str, plot_code: str) -> dict: # Reasoning
    """
    Calls LLM -> checks the graph and produces feedback and tells us whether it needs to be regenerated.

    """

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
        "Be very careful with your feedback as wrong feedback can severey impact goodness of graphs. If there are no severe mistakes just set the correction key to false"
        "\n"
        "What to check (prioritize):\n"
        "- Visibility: text size, tick labels, title, axis labels, legend readability.\n"
        "- Overlap/clutter: label collisions, dense points, overlapping bars/lines, crowded legend.\n"
        "- Distinguishability: lines/markers too similar, categories hard to tell apart, missing/unclear legend.\n"
        "- Scaling/layout: axes limits, aspect ratio, too much empty space, cut-off labels, rotated ticks.\n"
        "- Data-ink issues: overplotting, too many categories, need aggregation/binning/faceting.\n"
        "- Accessibility basics: ensure contrast is adequate and the plot doesn’t rely on subtle differences only.\n"
        "IMPORTANT: Do NOT propose a change if it doesn't heavily impact readability. Suggesting changes that would result in only minor improvements is NOT allowed.\n"
        "\n"
        "Output requirements (STRICT):\n"
        "- Output only a valid JSON with keys:\n"
        "  - feedback: the feedback of the graph, this is strictly only a string.\n"
        "  - correction: true or false based on whether you think the mistakes in the graph are large enough to warrant re-generation of the plot. If graph cannot be improved without changing its type then set to false.\n"
        "- Keep it very concise, return only faults of the graph and no positive feedback or explanations.\n"
        "- Each bullet MUST include:\n"
        "  (a) the issue (if any),\n"
        "  (b) a concrete fix in text\n"
        "- You should output the negative feedback even if it's not severe enough to warrant re-generation."
        "- Do NOT nitpick minor stylistic preferences. \n"
        "- Do NOT propose changing the plot type.\n"
    )

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

    resp = llm.invoke([msg])

    out = resp.content
    _, out = after_think(out)

    feedback = json.loads(out.replace("```json", "").replace("```", ""))

    return feedback

def recode_call(llm, features, selected_plot, previous_code, corrections) -> dict: # Reasoning
    """
    Calls LLM -> given the previous code and and the feedback, regenerate the code to hopefully fix the mistakes.

    """

    prompt = (
        "You are a plot rendering agent.\n"
        "You are given:\n"
        "1) A pandas DataFrame named `df`.\n"
        "2) A JSON object named `selected_plot` that was produced by a previous model call.\n"
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
        f"previous_code: {json.dumps(previous_code or '', ensure_ascii=False)}\n\n"
        f"corrections: {json.dumps(corrections or '', ensure_ascii=False)}\n"
    )

    out = llm.invoke(prompt).content
    _, out = after_think(out)

    code = out.replace("```python", "").replace("```", "")

    return code

def rejection_call(llm, image_path: str) -> dict:
    """
    Calls LLM -> given the image, return whether graph is good enough to be used in the dataset.
    """

    reject_prompt = (
        "You are a visualization QA reviewer.\n"
        "\n"
        "You will be given:\n"
        "1) An IMAGE of a chart or plot.\n"
        "\n"
        "Goal:\n"
        "Decide whether the chart is readable and coherent enough to be accepted into a dataset as a valid visualization of data.\n"
        "The main question is: can a person understand that this is a meaningful chart and read the visualized data reasonably well?\n"
        "\n"
        "Be lenient. The chart does NOT need to be beautiful, optimal, or publication-quality.\n"
        "Only reject charts with serious readability or coherence problems.\n"
        "\n"
        "Accept the chart if:\n"
        "- It clearly appears to visualize data.\n"
        "- The main plotted values, trends, or patterns are visible.\n"
        "- Axes, labels, legend, or context are readable enough when they are needed for interpretation.\n"
        "- Categories, lines, bars, points, or other visual elements are distinguishable enough.\n"
        "- The chart is not so cluttered, cut off, distorted, or low-quality that the data cannot be understood.\n"
        "\n"
        "Reject the chart ONLY if it has a severe issue that makes it unsuitable for the dataset, such as:\n"
        "- Text, tick labels, legend, or axis labels are unreadable when needed for interpretation.\n"
        "- Important chart elements are cut off or missing.\n"
        "- The plot is too cluttered or overlapped to understand the data.\n"
        "- Colors, lines, markers, or categories are impossible to distinguish.\n"
        "- The chart is visually corrupted, nonsensical, empty, or does not appear to convey data.\n"
        "- The layout or scaling makes the data misleading or unreadable.\n"
        "\n"
        "Do NOT reject charts for minor stylistic issues, small imperfections, suboptimal design choices, or changes that would only slightly improve the chart.\n"
        "Do NOT suggest changing the chart type.\n"
        "Do NOT nitpick.\n"
        "\n"
        "Output requirements STRICT:\n"
        "- Output only valid JSON.\n"
        "- Use exactly these keys:\n"
        "  - accept: true or false, whether the chart should be accepted into the dataset.\n"
        "  - reason: a concise string explaining the reason for rejection. If accept is true, use an empty string.\n"
        "\n"
        "Rules:\n"
        "- If the chart is usable, return: {\"accept\": true, \"reason\": \"\"}\n"
        "- If the chart is not usable, return: {\"accept\": false, \"reason\": \"brief reason for rejection\"}\n"
        "- The reason should only describe severe readability, coherence, or usability issues.\n"
        "- Do not include positive feedback.\n"
        "- Do not include minor issues.\n"
        "- Do not include explanations outside the JSON.\n"
    )

    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    # Multimodal message (works with LangChain OpenAI-compatible chat models that support vision)
    msg = HumanMessage(
        content=[
            {"type": "text", "text": reject_prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
        ]
    )

    resp = llm.invoke([msg])

    out = resp.content
    _, out = after_think(out)

    feedback = json.loads(out.replace("```json", "").replace("```", ""))

    return feedback

def describe_graph_png(llm, png_path, plot_code, graph_data, graph_df, dataset_desc, plot_description) -> str: # Reasoning
    """
    Calls LLM -> given the image, code, structured metadata, data, dataset description and short plot description
    generate a longer, detailed description for the graph.

    """

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
        "While the description should be detailed it should not be over around 2500 words.\n"
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

    resp = llm.invoke([msg]).content

    _, out = after_think(resp)

    return out

def generate_graph_questions(llm, png_path, dataset_desc, plot_desc, graph_data, num) -> list[dict]:  # Reasoning
    """
    Calls LLM -> given the image, dataset desctiption, metadata and full graph description,
    generate 20 question and answer pairs.

    """

    #TODO: Implement variable questions
    
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
        "- 7 easy (direct reading: titles, axes, legend, counts, obvious comparisons)\n"
        "- 6 medium (interpretation: comparisons across groups, trends, approximate ranges, notable patterns)\n"
        "- 7 hard (multi-step reasoning grounded in the chart + context, but still definitively answerable, these questions should be questions that experts would ask when looking at the chart.)\n"
        "\n"
        "CRITICAL CONSTRAINTS:\n"
        "- Every question MUST be definitively answerable from the provided IMAGE and/or the provided chart/dataset descriptions.\n"
        "- Do NOT ask questions that require external knowledge.\n"
        "- Do NOT ask questions that require more data than what is shown/described.\n"
        "- Do NOT produce questions that are just instructions like 'analyze' or 'explain how'.\n"
        "- Avoid vague questions. Each must have a single, checkable answer.\n"
        "- If exact numeric values are not available, ask questions that accept approximate answers only when the chart clearly supports approximation.\n"
        "- While you can help yourself with the description to answer a question more accurately, do NOT ask questions about something that can't be answered ONLY from the image."
        "\n"
        "Output format (STRICT):\n"
        "Return ONLY valid JSON: an array of exactly 20 objects.\n"
        "Each object must have EXACTLY these keys:\n"
        "{\n"
        "  \"question\": string,\n"
        "  \"answer\": string,             # must be concrete, not instructions\n"
        "  \"answer_basis\": \"image\"|\"both\"  # where the answer comes from\n"
        "}\n"
        "\n"
        "Quality requirements:\n"
        "- Questions should cover BOTH:\n"
        "  (a) chart mechanics/visual properties (axes, legend, encodings, layout), and\n"
        "  (b) semantics in dataset context (what variables represent, what patterns mean).\n"
        "- Do not repeat the same question pattern; vary them.\n"
        "- Questions should be related to the graph and the data in the graph, do NOT ask general questions about the dataset that do not directly relate to the chart.\n"
        "- Do NOT ask questions like how many rows are in the data or how many rows were left out, unless that is specified on the image itself.\n"
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

    _, out = after_think(resp)

    start = out.find("[")
    end = out.rfind("]")
    return json.loads(out[start:end+1])

def give_question_types(llm, questions): # No reasoning
    """
    Calls LLM -> categorizes questions into types, for better evaluation.

    """

    prompt = (
        "You are a question categorizer.\n"
        "YOu will be given questions and you have to assing types to them.\n"
        "You can assign questions only these labels:\n"
        "  - 'metadata': The question asks for chart text or styling directly visible in the image, such as the title, axis labels, legend entries, colors, units, or tick values.\n"
        "  - 'value extraction': The question asks for the value of a specific plotted element or local visual quantity, such as a bar height, point value, coordinate, count, or frequency.\n"
        "  - 'comparison': The question asks to compare two or more visual elements, categories, series, or values.\n"
        "  - 'trends': The question asks about the overall pattern, structure, or distribution in the plot, such as increase/decrease, skewness, clustering, gaps, outliers, seasonality, or general shape.\n"
        "  - 'reasoning': The question requires combining visual evidence from the image with arithmetic, dataset context, or external information to infer the answer. Overall requires a more complex reasoning process.\n\n"
        "Rules:"
        "   - Respond only with a valid array of string values from the previous list.\n"
        "   - You may NOT add or remove questions.\n"
        "   - You must give each question EXACTLY ONE label.\n\n"
        "Questions:\n"
        f"{json.dumps(questions, indent=2)}"
    
    )

    out = llm.invoke(prompt).content
    _, out = after_think(out)

    out = json.loads(out.replace("```json", "").replace("```", ""))

    return out
