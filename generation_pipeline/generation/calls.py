import base64
import json
from typing import Literal

import numpy as np
from helpers import after_think, strip_code_fences
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, ConfigDict, Field, RootModel
from tools import invoke_with_tools


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DatasetUsability(StrictModel):
    useful: bool


class DatasetDescription(StrictModel):
    description: str


class GraphSpec(StrictModel):
    type: str
    features: list[str]
    description: str


class GraphSpecs(RootModel[list[GraphSpec]]):
    pass


class FeatureNames(RootModel[list[str]]):
    pass


class PlotPlan(StrictModel):
    notes: str


class GraphEvaluation(StrictModel):
    rating: int = Field(ge=1, le=5)
    error_type: list[
        Literal[
            "none",
            "uninformative",
            "variable_semantics",
            "aggregation_transformation",
            "misleading_encoding",
            "excessive_cardinality",
            "missing_invalid_data",
            "visibility",
            "overlap_clutter",
            "distinguishability",
            "scaling_layout",
            "missing_elements",
            "rendering_error",
            "other",
        ]
    ]
    feedback: str


class GraphEvaluationError(StrictModel):
    type: Literal[
        "none",
        "uninformative",
        "variable_semantics",
        "aggregation_transformation",
        "misleading_encoding",
        "excessive_cardinality",
        "missing_invalid_data",
        "visibility",
        "overlap_clutter",
        "distinguishability",
        "scaling_layout",
        "missing_elements",
        "rendering_error",
        "other",
    ]
    severity: int = Field(ge=1, le=5)
    description: str
    feedback: str


class PerErrorGraphEvaluation(StrictModel):
    errors: list[GraphEvaluationError]


class GraphDescription(StrictModel):
    description: str = Field(min_length=1)


class GraphQuestion(StrictModel):
    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    answer_basis: Literal["image", "both"]


class GraphQuestions(StrictModel):
    questions: list[GraphQuestion]


class QuestionTypes(
    RootModel[
        list[
            Literal[
                "metadata",
                "value extraction",
                "comparison",
                "trends",
                "reasoning",
            ]
        ]
    ]
):
    pass


def invoke_llm(llm, messages, df=None, use_tools=False, call_metadata=None, selected_plot=None):
    config = {"metadata": call_metadata} if call_metadata else None
    if use_tools and df is not None:
        return invoke_with_tools(llm, messages, df, config=config, selected_plot=selected_plot)
    return llm.invoke(messages, config=config)


def invoke_structured_llm(llm, messages, schema, df=None, use_tools=False, call_metadata=None):
    config = {"metadata": call_metadata} if call_metadata else None
    if use_tools and df is not None:
        response = invoke_with_tools(
            llm,
            messages,
            df,
            response_format=schema,
            config=config,
        )
        parsed = response.additional_kwargs.get("parsed")
        if parsed is None:
            raise ValueError("Structured response was not returned")
        return schema.model_validate(parsed)

    return llm.with_structured_output(
        schema,
        method="json_schema",
    ).invoke(messages, config=config)


def determine_dataset_usability_call(llm, metadata, call_metadata=None) -> dict:
    """
    Determines whether the dataset is suitable for generating meaningful
    visualizations.
    """

    prompt = (
        "You are a strict data visualization expert.\n"
        "You receive metadata about an OpenML dataset in JSON format.\n\n"
        "Task:\n"
        "Determine whether the dataset can be used to generate visualizations "
        "that are both semantically meaningful and visually informative.\n\n"
        "Be strict. Reject datasets when:\n"
        "- The feature meanings are unclear or anonymized.\n"
        "- There are too few meaningful features.\n"
        "- The metadata does not provide enough context to interpret the data.\n"
        "- Any generated visualizations would likely be arbitrary or misleading.\n\n"
        "Output format (STRICT):\n"
        "- Return ONLY a valid JSON object with EXACTLY this key:\n"
        '  "useful": true or false\n\n'
        f"DATASET_METADATA:\n{json.dumps(metadata, ensure_ascii=False)}"
    )

    response = invoke_structured_llm(llm, prompt, DatasetUsability, call_metadata=call_metadata)
    return response.model_dump()


def format_dataset_description_call(llm, metadata, call_metadata=None) -> dict:
    """
    Converts the dataset metadata into a clean, readable description.
    """

    prompt = (
        "You are a data documentation expert.\n"
        "You receive metadata about an OpenML dataset in JSON format.\n\n"
        "Task:\n"
        "Create a concise and readable description of the dataset and its "
        "features.\n\n"
        "Requirements:\n"
        "- Describe what the dataset contains.\n"
        "- Explain the meaning of its features when that information is available.\n"
        "- Remove author names, citations, acknowledgements, download instructions, "
        "and other information unrelated to the data itself.\n"
        "- Do not invent information that is not present in the metadata.\n"
        "- The description must be a single string.\n\n"
        "Output format (STRICT):\n"
        "- Return ONLY a valid JSON object with EXACTLY this key:\n"
        '  "description": string\n\n'
        f"DATASET_METADATA:\n{json.dumps(metadata, ensure_ascii=False)}"
    )

    response = invoke_structured_llm(llm, prompt, DatasetDescription, call_metadata=call_metadata)
    return response.model_dump()


def graphs_call(
    llm, features: dict, dataset_description: str, num_graphs: int, creativity: float, alpha: float, beta: float, call_metadata=None
) -> list[dict]:  # Reasoning
    """
    Calls LLM -> returns 10 specifications for 10 graphs that could be made from this dataset.
    The specifications consist of graph type, short description, and features that should be used.

    LLM gets a random creativity rating that encourages it to produce more standard or more unusual graphs.


    In parameters set creativity score from 0 to 1 for fixed creativity score or set to 'random' and set parameters alpha and beta to sample the creativity score from beta distribution.

    """
    if creativity is not None:
        creat = np.clip(creativity, 0.0, 1.0)

    else:
        creat = np.random.beta(alpha, beta)

    # TODO: implement amount of graphs chosen based on dataset

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
        "- Do NOT describe the plots.\n"
        "- DO NOT propose plots with too many different subplots. It should contain a maximum of 5 subplots per row and 5 per column."
        "Output format (STRICT):\n"
        "- Return ONLY a valid JSON array.\n"
        "- Each element must be a JSON object with EXACTLY these keys:\n"
        '  • "type": string (name of the plot type)\n'
        '  • "features": array of strings (feature names from the dataset used for this plot)\n'
        '  • "description": somewhat detailed description of what the plot is supposed to show, both semantically and visually\n'
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

    response = invoke_structured_llm(llm, prompt, GraphSpecs, call_metadata=call_metadata)
    return [spec.model_dump() for spec in response.root]


def replace_vars_call(llm, features: dict, dataset_description: str, call_metadata=None) -> list[str]:  # No reasoning
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

    response = invoke_structured_llm(llm, prompt, FeatureNames, call_metadata=call_metadata)
    return response.root


def compute_info_call(llm, features, selected_plot, head, call_metadata=None):  # No reasoning
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

    out = invoke_llm(llm, prompt, call_metadata=call_metadata).content
    _, out = after_think(out)

    return out

def plan_call(llm, features, selected_plot, df=None, use_tools=False, call_metadata=None) -> str:
    plan_prompt = (
        "You are a plotting planner.\n"
        "You will be given:\n"
        "- selected_plot (type, required features list, matplotlib style)\n"
        "- FEATURES_METADATA (column types/semantics)\n"
        "- HEAD (first rows)\n\n"
        "Output ONLY a single JSON object (no code, no extra text).\n"
        "JSON schema (all keys required):\n"
        "{\n"
        '  "notes": string'
        "}\n\n"
        "Rules:\n"
        "- Give the plotting agent some notes about how to proceed with the writing of the code and what to watch out for."
        "- Keep notes concise but not too short. (100-150 words)\n\n"
        f"selected_plot = {json.dumps(selected_plot, ensure_ascii=False)}\n\n"
        f"FEATURES_METADATA:\n{json.dumps(features, ensure_ascii=False)}\n\n"
    )

    response = invoke_structured_llm(
        llm,
        plan_prompt,
        PlotPlan,
        df,
        use_tools,
        call_metadata,
    )
    return json.dumps(response.model_dump(), ensure_ascii=False)

CODE_TOOL_INSTRUCTIONS = (
    "TOOL USAGE REQUIREMENTS:\n"
    "- Tools are only for inspecting data and validating code. A tool result is NEVER the final answer.\n"
    "- You MUST validate the complete candidate code with `execute_plot_code`.\n"
    "- If execution fails, correct the COMPLETE code and validate it again.\n"
    "- When execution succeeds, your next response MUST contain exactly the complete Python code passed to the successful tool call.\n"
    "- Do NOT call another tool after successful execution.\n"
    "- Do NOT summarize, explain, confirm success, or describe the resulting chart.\n"
    "- Do NOT return phrases such as \"the plot was successfully rendered.\"\n"
    "- The final output must contain ONLY the code and no other text.\n"
    "- The final response must still define `graph_df` and `graph_data` and save the image to `graph_file_path`.\n\n"
)

def graph_call(
    llm,
    features,
    selected_plot,
    head,
    plan,
    df=None,
    use_tools=False,
    call_metadata=None,
) -> str:  # No reasoning plan, reasoning code
    """
    Calls LLM -> generates the code needed to plot the graph.

    If previous_code and execution_error are provided, the model repairs the
    previous implementation instead of generating a new visualization approach.
    """

    code_prompt = (
        "You are a plot rendering agent.\n"
        "You are given:\n"
        "1) A pandas DataFrame named `df`.\n"
        "2) A JSON object named `selected_plot` that was produced by a previous model call.\n"
        "selected_plot has exactly these keys:\n"
        '  - "type": the required plot type to render\n'
        '  - "features": the exact list of column names that must be used for the plot\n'
        "Your job:\n"
        '- Render EXACTLY ONE plot whose plot type matches selected_plot["type"].\n'
        "- Follow `plan` for x/y/hue/facet/aggregation/binning/filters/figsize/title.\n"
        "- Save that plot with plt.savefig(), the path will be available in a variable named 'graph_file_path', use this variable but don't change it.\n"
        '- Use ONLY the columns listed in selected_plot["features"].\n'
        "- You may derive temporary helper columns ONLY from those listed features.\n\n"
        "Libraries:\n"
        "- Use ONLY pandas, numpy, matplotlib, scikit-learn and default python libraries. Do NOT use seaborn!\n"
        "- Do NOT use pandas plotting; always use matplotlib directly.\n\n"
        "CRITICAL: After the code runs, define BOTH:\n"
        "1) A pandas DataFrame named `graph_df` containing the FINAL PROCESSED DATA actually used for plotting.\n"
        "2) A JSON-serializable dict named `graph_data` with EXACTLY these keys (all keys required):\n\n"
        "graph_data = {\n"
        '  "plot_type": string,\n'
        '  "features_expected": list[str],\n'
        '  "features_used": list[str],\n'
        '  "derived_features": list[str],\n'
        '  "x": string or null or array of values if multiple subplots,\n'
        '  "y": string or null or array of values if multiple subplots,\n'
        '  "hue": string or null or array of values if multiple subplots,\n'
        '  "facet": string or null,\n'
        '  "aggregation": string or null or array of values if multiple subplots,\n'
        '  "binning": string or null or array of values if multiple subplots,\n'
        '  "transformations": list[str] or array of values if multiple subplots,\n'
        '  "filters": list[str],\n'
        '  "n_rows_input": int,\n'
        '  "n_rows_plotted": int,\n'
        '  "title": string\n'
        "}\n\n"
        "Validation rules:\n"
        "- `graph_df` must contain ONLY columns listed in `features_used`.\n"
        "- `graph_df` must reflect EXACTLY what is plotted (no extra rows/cols).\n"
        '- If any feature in selected_plot["features"] is missing from df.columns, pick a different plot approach\n'
        '  that still matches selected_plot["type"] but uses the remaining provided features only;\n'
        '  ALWAYS keep graph_data["features_expected"] unchanged.\n'
        '- Do NOT change selected_plot["type"].\n'
        '- Do NOT use columns outside selected_plot["features"].\n'
        "- Do NOT mutate selected_plot in any way in general.\n"
        "- Do NOT print anything.\n"
        "- Output ONLY MINIMAL executable Python code.\n"
        "- Use large enough figures; use plt.tight_layout().\n\n"
    )

    if use_tools:
        code_prompt += CODE_TOOL_INSTRUCTIONS

    code_prompt += (
        "Inputs you must rely on:\n"
        f"selected_plot = {json.dumps(selected_plot, ensure_ascii=False)}\n\n"
        f"FEATURES_METADATA:\n{json.dumps(features, ensure_ascii=False)}\n\n"
        f"HEAD:\n{json.dumps(head, ensure_ascii=False)}\n"
    )

    if plan is not None:
        code_prompt += f"\nPLAN from planning agent:\n{json.dumps(plan, ensure_ascii=False)}\n"

    out = invoke_llm(llm, code_prompt, df, use_tools, call_metadata, selected_plot).content

    try:
        _, out = after_think(out)
    except Exception:
        pass

    code = strip_code_fences(out)
    return code


def recode_call(
    llm,
    features,
    selected_plot,
    previous_code,
    corrections,
    df=None,
    use_tools=False,
    call_metadata=None,
) -> dict:  # Reasoning
    """
    Calls LLM -> given the previous code and and the feedback, regenerate the code to hopefully fix the mistakes.

    """

    code_prompt = (
        "You are a plot rendering agent.\n"
        "You are given:\n"
        "1) A pandas DataFrame named `df`.\n"
        "2) A JSON object named `selected_plot` that was produced by a previous model call.\n"
        "selected_plot has exactly these keys:\n"
        '  - "type": the required plot type to render\n'
        '  - "features": the exact list of column names that must be used for the plot\n'
        '  - "style": the matplotlib style to use for rendering\n\n'
        "Your job:\n"
        '- Render EXACTLY ONE plot whose plot type matches selected_plot["type"].\n'
        f"- Save that plot with plt.savefig(), the path will be available in a variable named 'graph_file_path'\n"
        '- Use ONLY the columns listed in selected_plot["features"].\n'
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
        '  "plot_type": string,                     # must equal selected_plot["type"]\n'
        '  "features_expected": list[str],          # must equal selected_plot["features"] exactly\n'
        '  "features_used": list[str],              # columns actually used (include derived names if created)\n'
        '  "derived_features": list[str],           # names of any derived helper columns you create\n'
        '  "x": string or null,\n'
        '  "y": string or null,\n'
        '  "hue": string or null,\n'
        '  "facet": string or null,\n'
        '  "aggregation": string or null,\n'
        '  "binning": string or null,\n'
        '  "transformations": list[str],\n'
        '  "filters": list[str],\n'
        '  "n_rows_input": int,\n'
        '  "n_rows_plotted": int,                   # MUST equal len(graph_df)\n'
        '  "title": string\n'
        "}\n\n"
        "Validation rules:\n"
        "- `graph_df` must contain ONLY columns listed in `features_used`.\n"
        "- `graph_df` must reflect EXACTLY what is plotted (no extra rows or columns).\n"
        '- If any feature in selected_plot["features"] is missing from df.columns, pick a different plot approach '
        'that still matches selected_plot["type"] but uses the remaining provided features only; '
        'ALWAYS keep graph_data["features_expected"] unchanged.\n'
        '- Do NOT use columns outside selected_plot["features"].\n'
        "- Do NOT mutate selected_plot in any way in general.\n"
        "- Do NOT print anything.\n"
        "- Output ONLY MINIMAL executable Python code.\n\n"
    )

    if use_tools:
        code_prompt += CODE_TOOL_INSTRUCTIONS

    code_prompt += (
        "Inputs you must rely on:\n"
        f"selected_plot = {json.dumps(selected_plot, ensure_ascii=False)}\n\n"
        f"FEATURES_METADATA:\n{json.dumps(features, ensure_ascii=False)}\n\n"
        f"previous_code: {json.dumps(previous_code or '', ensure_ascii=False)}\n\n"
        f"corrections: {json.dumps(corrections or '', ensure_ascii=False)}\n"
    )

    out = invoke_llm(llm, code_prompt, df, use_tools, call_metadata, selected_plot).content
    _, out = after_think(out)

    code = strip_code_fences(out)

    return code


def graph_error_call(
    llm,
    features,
    selected_plot,
    head,
    previous_code,
    execution_error,
    df=None,
    use_tools=False,
    call_metadata=None,
) -> str:
    """
    Calls LLM -> repairs graph code that failed during execution.

    The previous code is treated as the source of truth for the intended
    visualization. The model should only fix execution issues and should not
    redesign or semantically improve the chart.
    """

    code_prompt = (
        "You are a plot code repair agent.\n"
        "You are given Python code that was intended to render a plot but failed during execution.\n"
        "Your job is to fix the execution error while preserving the intended visualization.\n\n"
        "Repair rules:\n"
        "- Treat the PREVIOUS CODE as the source of truth for the intended visualization.\n"
        "- Fix ONLY problems necessary for the code to execute successfully.\n"
        "- Preserve the current plot type, features, transformations, aggregation, "
        "binning, filters, axes, and visual semantics as much as possible.\n"
        "- Do NOT redesign, simplify, or semantically improve the chart unless required to fix the error.\n"
        "- Do NOT change plot type.\n"
        '- Use ONLY the columns listed in selected_plot["features"].\n'
        "- You may derive temporary helper columns ONLY from those listed features.\n"
        "- Save the plot with plt.savefig(); the path is available in a variable named "
        "`graph_file_path`. Use this variable but do not change it.\n\n"
        "Libraries:\n"
        "- Use ONLY pandas, numpy, matplotlib, scikit-learn and default python libraries. "
        "Do NOT use seaborn!\n"
        "- Do NOT use pandas plotting; always use matplotlib directly.\n\n"
        "CRITICAL: The corrected code must still define BOTH:\n"
        "1) A pandas DataFrame named `graph_df` containing the FINAL PROCESSED DATA actually used for plotting.\n"
        "2) A JSON-serializable dict named `graph_data` with EXACTLY these keys (all keys required):\n\n"
        "graph_data = {\n"
        '  "plot_type": string,\n'
        '  "features_expected": list[str],\n'
        '  "features_used": list[str],\n'
        '  "derived_features": list[str],\n'
        '  "x": string or null or array of values if multiple subplots,\n'
        '  "y": string or null or array of values if multiple subplots,\n'
        '  "hue": string or null or array of values if multiple subplots,\n'
        '  "facet": string or null,\n'
        '  "aggregation": string or null or array of values if multiple subplots,\n'
        '  "binning": string or null or array of values if multiple subplots,\n'
        '  "transformations": list[str] or array of values if multiple subplots,\n'
        '  "filters": list[str],\n'
        '  "n_rows_input": int,\n'
        '  "n_rows_plotted": int,\n'
        '  "title": string\n'
        "}\n\n"
        "Validation rules:\n"
        "- `graph_df` must contain ONLY columns listed in `features_used`.\n"
        "- `graph_df` must reflect EXACTLY what is plotted (no extra rows/cols).\n"
        '- ALWAYS keep graph_data["features_expected"] equal to selected_plot["features"].\n'
        "- Do NOT print anything.\n"
        "- Output ONLY MINIMAL executable Python code.\n"
        "- Return the COMPLETE corrected code, not a patch or explanation.\n\n"
        "Inputs you must rely on:\n"
        f"selected_plot = {json.dumps(selected_plot, ensure_ascii=False)}\n\n"
        f"FEATURES_METADATA:\n{json.dumps(features, ensure_ascii=False)}\n\n"
        f"HEAD:\n{json.dumps(head, ensure_ascii=False)}\n\n"
        "PREVIOUS CODE:\n"f"{previous_code}\n\n"
        "EXECUTION ERROR:\n"f"{execution_error}\n"
    )

    if use_tools:
        code_prompt += CODE_TOOL_INSTRUCTIONS

    code_prompt += (
        "Inputs you must rely on:\n"
        f"selected_plot = {json.dumps(selected_plot, ensure_ascii=False)}\n\n"
        f"FEATURES_METADATA:\n{json.dumps(features, ensure_ascii=False)}\n\n"
        f"HEAD:\n{json.dumps(head, ensure_ascii=False)}\n\n"
        "PREVIOUS CODE:\n"f"{previous_code}\n\n"
        "EXECUTION ERROR:\n"f"{execution_error}\n"
    )

    out = invoke_llm(llm, code_prompt, df, use_tools, call_metadata, selected_plot).content

    try:
        _, out = after_think(out)
    except Exception:
        pass

    code = strip_code_fences(out)
    return code


def graph_evaluation_call(
    llm,
    image_path: str,
    plot_code: str,
    call_metadata=None,
    feedback_type: Literal["rating", "per_error"] = "rating",
) -> dict:
    """
    Evaluates whether a graph is good enough for the final dataset.
    If it is not, returns concrete feedback for the next regeneration.
    """

    fixed_evaluation_prompt = (
        "You are a visualization QA reviewer.\n"
        "\n"
        "You will be given:\n"
        "1) An IMAGE of a chart or plot.\n"
        "2) The PYTHON CODE that generated it.\n"
        "\n"
        "Goal:\n"
        "Evaluate whether the chart is readable, semantically valid, and informative "
        "enough to be included in a visualization dataset.\n"
        "\n"
        "Use the image as primary evidence and the code to understand variables, "
        "transformations, grouping, aggregation, filtering, and visual encoding.\n"
        "\n"
        "Be lenient about minor styling. Penalize only issues that meaningfully affect "
        "interpretation, semantic validity, or informativeness. The chart does not "
        "need to be optimal or publication-quality.\n"
        "\n"
        "The chart type is fixed. NEVER recommend changing it. Corrections may change "
        "variables, aggregation, grouping, filtering, binning, ordering, scaling, "
        "normalization, labels, limits, sampling, or other code-level details.\n"
        "\n"
        "Check for:\n"
        "\n"
        "SEMANTIC / INFORMATIONAL PROBLEMS:\n"
        "- Variables used with invalid or misleading semantics, such as IDs or "
        "categorical codes treated as meaningful continuous measurements.\n"
        "- Grouping, aggregation, filtering, normalization, ordering, scaling, or "
        "other transformations that create a misleading or trivial result.\n"
        "- Charts with effectively no meaningful comparison or variation because of "
        "their construction (e.g. one effective point/group or constant values).\n"
        "- Excessive category cardinality or granularity that prevents interpretation.\n"
        "- Labels, axes, titles, legends, or encodings implying unsupported meaning.\n"
        "- Missing, invalid, artifact, or default values materially distorting the chart.\n"
        "- A technically valid chart that communicates almost no useful information "
        "because of its variable or encoding choices.\n"
        "\n"
        "Do NOT penalize a chart simply because the true relationship is weak, the "
        "distribution is simple, or no strong pattern exists.\n"
        "\n"
        "READABILITY PROBLEMS:\n"
        "- Unreadable or clipped labels, ticks, titles, legends, or annotations.\n"
        "- Overlap, clutter, or excessive density that prevents interpretation.\n"
        "- Categories, colors, lines, bars, points, or markers that cannot be "
        "distinguished adequately.\n"
        "- Scaling, limits, or layout that hide important information or mislead.\n"
        "- Missing required context or rendering failures.\n"
        "\n"
        "Do not nitpick minor aesthetic issues.\n"
        "\n"
    )

    rating_evaluation_prompt = (
        "Rating scale:\n"
        "- 5: Excellent; no meaningful problems.\n"
        "- 4: Good; minor issues only, fully usable.\n"
        "- 3: Acceptable but noticeably flawed; still interpretable and defensible.\n"
        "- 2: Poor; major issue substantially harms the chart and warrants regeneration.\n"
        "- 1: Invalid; severe semantic, informational, readability, or rendering failure.\n"
        "\n"
        "Use 1-2 only when regeneration is justified. Minor imperfections should "
        "normally receive 4-5.\n"
        "\n"
        "Error types:\n"
        "- uninformative\n"
        "- variable_semantics\n"
        "- aggregation_transformation\n"
        "- misleading_encoding\n"
        "- excessive_cardinality\n"
        "- missing_invalid_data\n"
        "- visibility\n"
        "- overlap_clutter\n"
        "- distinguishability\n"
        "- scaling_layout\n"
        "- missing_elements\n"
        "- rendering_error\n"
        "- other\n"
        "\n"
        "Output ONLY valid JSON with exactly these keys:\n"
        '  - "rating": integer from 1 to 5\n'
        '  - "error_type": list of applicable error types; use empty array [] if there are no errors.\n'
        '  - "feedback": concise description of the most important issue and a '
        "concrete code-level fix that preserves the chart type\n"
        "\n"
        "Rules:\n"
        "- Rating 5 with no issue: use an empty array [] and an empty feedback string.\n"
        "- Ratings 1-2: feedback must explain why regeneration is warranted and give "
        "a concrete fix.\n"
        "- Ratings 3-4: feedback may mention meaningful issues that do not require "
        "regeneration.\n"
        "- Include faults only; do NOT include any positive feedback, Markdown, or extra text.\n"
        "- Do NOT suggest changing the chart type. This is very important, so you MUST NOT suggest changing the chart type.\n"
    )

    per_error_evaluation_prompt = (
        "Error types:\n"
        "- uninformative\n"
        "- variable_semantics\n"
        "- aggregation_transformation\n"
        "- misleading_encoding\n"
        "- excessive_cardinality\n"
        "- missing_invalid_data\n"
        "- visibility\n"
        "- overlap_clutter\n"
        "- distinguishability\n"
        "- scaling_layout\n"
        "- missing_elements\n"
        "- rendering_error\n"
        "- other\n"
        "\n"
        "Severity levels:\n"
        "- 1: A nice-to-have improvement, such as better coloring or aesthetics.\n"
        "- 2: A minor issue; the chart remains usable and interpretable.\n"
        "- 3: A meaningful issue that harms interpretation and warrants regeneration.\n"
        "- 4: A major issue that severely harms interpretation.\n"
        "- 5: The chart is completely unreadable or impossible to interpret.\n"
        "\n"
        'Output ONLY valid JSON with exactly one key, "errors", containing a list of errors.\n'
        "Each error must have exactly these keys:\n"
        '  - "type": one applicable error type from the list above\n'
        '  - "severity": integer from 1 to 5\n'
        '  - "description": concise description of the fault\n'
        '  - "feedback": very short, broad instructions for how to fix it while preserving the chart type\n'
        "\n"
        "Rules:\n"
        '- If there are no errors, return {"errors": []}.\n'
        "- Report each distinct fault separately.\n"
        "- Include faults only; do NOT include any positive feedback, Markdown, or extra text.\n"
        "- Do NOT suggest changing the chart type. This is very important, so you MUST NOT suggest changing the chart type.\n"
    )

    if feedback_type == "rating":
        evaluation_prompt = fixed_evaluation_prompt + rating_evaluation_prompt
        evaluation_schema = GraphEvaluation
    elif feedback_type == "per_error":
        evaluation_prompt = fixed_evaluation_prompt + per_error_evaluation_prompt
        evaluation_schema = PerErrorGraphEvaluation
    else:
        raise ValueError(f"Unsupported feedback_type: {feedback_type}")

    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    msg = HumanMessage(
        content=[
            {"type": "text", "text": evaluation_prompt},
            {"type": "text", "text": f"CODE:\n{plot_code}"},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_b64}"},
            },
        ]
    )

    response = invoke_structured_llm(llm, [msg], evaluation_schema, call_metadata=call_metadata)
    return response.model_dump()


def describe_graph_png(
    llm,
    png_path,
    plot_code,
    graph_data,
    graph_df,
    dataset_desc,
    plot_description,
    use_tools=False,
    call_metadata=None,
) -> str:  # Reasoning
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
        "- Put the complete description in the `description` response field.\n"
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
        '- State the chart type (must match graph_data["plot_type"]).\n'
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

    response = invoke_structured_llm(
        llm,
        [msg],
        GraphDescription,
        graph_df,
        use_tools,
        call_metadata,
    )
    return response.description


def generate_graph_questions(
    llm,
    png_path,
    dataset_desc,
    plot_desc,
    graph_data,
    num,
    graph_df=None,
    use_tools=False,
    call_metadata=None,
) -> list[dict]:  # Reasoning
    """
    Calls LLM -> given the image, dataset desctiption, metadata and full graph description,
    generate 20 question and answer pairs.

    """

    # TODO: Implement variable questions

    easy = round(num * 0.35)
    medium = round(num * 0.30)
    hard = num - easy - medium

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
        f"Generate EXACTLY {num} questions about the chart.\n"
        "Include a mix of difficulties:\n"
        f"- {easy} easy (direct reading: titles, axes, legend, counts, obvious comparisons)\n"
        f"- {medium} medium (interpretation: comparisons across groups, trends, approximate ranges, notable patterns)\n"
        f"- {hard} hard (multi-step reasoning grounded in the chart + context, but still definitively answerable, these questions should be questions that experts would ask when looking at the chart.)\n"
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
        "Return the questions in the `questions` response field.\n"
        "Each question must have EXACTLY these keys:\n"
        "{\n"
        '  "question": string,\n'
        '  "answer": string,             # must be concrete, not instructions\n'
        '  "answer_basis": "image"|"both"  # where the answer comes from\n'
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

    msg = HumanMessage(
        content=[
            {"type": "text", "text": qa_prompt},
            {"type": "text", "text": f"DATASET DESCRIPTION:\n{dataset_desc}"},
            {"type": "text", "text": f"PLOT DESCRIPTION:\n{plot_desc}"},
            {"type": "text", "text": f"graph_data:\n{json.dumps(graph_data, ensure_ascii=False)}"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{png_b64}"}},
        ]
    )

    response = invoke_structured_llm(
        llm,
        [msg],
        GraphQuestions,
        graph_df,
        use_tools,
        call_metadata,
    )
    if len(response.questions) != num:
        raise ValueError(f"Expected {num} questions, received {len(response.questions)}")
    return [question.model_dump() for question in response.questions]


def generate_graph_question_one(
    llm,
    png_path,
    dataset_desc,
    plot_desc,
    graph_data,
    previous_questions,
    graph_df=None,
    use_tools=False,
    call_metadata=None,
) -> dict:
    """Generate one chart question using all previous questions as context."""

    # TODO: No access to previous questions option

    prompt = (
        "You are a chart QA generator.\n"
        "Generate EXACTLY ONE new question and answer about the chart.\n"
        "The question must be definitively answerable from the chart and the "
        "provided context, with a single checkable answer.\n"
        "Do not repeat or closely paraphrase any previous question.\n"
        "Prefer asking about a different visual element, relationship, or reasoning pattern than the previous questions.\n"
        "Return one response object with EXACTLY these fields:\n"
        "{\n"
        '  "question": string,\n'
        '  "answer": string,             # must be concrete, not instructions\n'
        '  "answer_basis": "image"|"both"  # where the answer comes from\n'
        "}\n"
        "Quality requirements:\n"
        "- Questions should cover BOTH:\n"
        "  (a) chart mechanics/visual properties (axes, legend, encodings, layout), and\n"
        "  (b) semantics in dataset context (what variables represent, what patterns mean).\n"
        "- Do not repeat the same question pattern; vary them.\n"
        "- Questions should be related to the graph and the data in the graph, do NOT ask general questions about the dataset that do not directly relate to the chart.\n"
        "- Do NOT ask questions like how many rows are in the data or how many rows were left out, unless that is specified on the image itself.\n"
        "- Do NOT include any extra text outside the JSON.\n"
    )

    with open(png_path, "rb") as file:
        png_b64 = base64.b64encode(file.read()).decode("utf-8")

    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {"type": "text", "text": f"DATASET DESCRIPTION:\n{dataset_desc}"},
            {"type": "text", "text": f"PLOT DESCRIPTION:\n{plot_desc}"},
            {
                "type": "text",
                "text": (f"GRAPH DATA:\n{json.dumps(graph_data, ensure_ascii=False)}"),
            },
            {
                "type": "text",
                "text": (f"PREVIOUS QUESTIONS:\n{json.dumps(previous_questions, ensure_ascii=False)}"),
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{png_b64}"},
            },
        ]
    )

    response = invoke_structured_llm(
        llm,
        [message],
        GraphQuestion,
        graph_df,
        use_tools,
        call_metadata,
    )
    return response.model_dump()


def give_question_types(llm, questions, call_metadata=None):  # No reasoning
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

    response = invoke_structured_llm(llm, prompt, QuestionTypes, call_metadata=call_metadata)
    return response.root
