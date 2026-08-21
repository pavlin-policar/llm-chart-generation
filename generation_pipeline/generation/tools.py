import ast
import json
import os
import tempfile

import numpy as np
import pandas as pd
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool

MAX_TOOL_CALLS = 8
MAX_RESULT_ROWS = 100
MAX_RESULT_COLUMNS = 40

SAFE_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "range": range,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
}

FORBIDDEN_CALLS = {
    "compile",
    "delattr",
    "dir",
    "eval",
    "exec",
    "getattr",
    "globals",
    "help",
    "input",
    "locals",
    "open",
    "setattr",
    "vars",
}

FORBIDDEN_METHODS = {
    "ExcelFile",
    "ExcelWriter",
    "HDFStore",
    "eval",
    "load",
    "memmap",
    "plot",
    "save",
    "savez",
    "savez_compressed",
    "to_clipboard",
    "to_csv",
    "to_excel",
    "to_feather",
    "to_gbq",
    "to_hdf",
    "to_html",
    "to_json",
    "to_latex",
    "to_markdown",
    "to_orc",
    "to_parquet",
    "to_pickle",
    "to_sql",
    "to_stata",
    "to_xml",
}

FORBIDDEN_NODES = (
    ast.AsyncFunctionDef,
    ast.Await,
    ast.ClassDef,
    ast.Delete,
    ast.FunctionDef,
    ast.Global,
    ast.Import,
    ast.ImportFrom,
    ast.Nonlocal,
    ast.Raise,
    ast.Try,
    ast.While,
    ast.With,
)


class PandasCodeValidator(ast.NodeVisitor):
    def visit(self, node):
        if isinstance(node, FORBIDDEN_NODES):
            raise ValueError(f"{type(node).__name__} is not allowed")
        return super().visit(node)

    def visit_Attribute(self, node):
        if node.attr.startswith("_"):
            raise ValueError("Private attributes are not allowed")
        self.generic_visit(node)

    def visit_Name(self, node):
        if node.id.startswith("_"):
            raise ValueError("Private names are not allowed")
        self.generic_visit(node)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
            raise ValueError(f"{node.func.id}() is not allowed")

        if isinstance(node.func, ast.Attribute):
            if node.func.attr in FORBIDDEN_METHODS or node.func.attr.startswith("read_"):
                raise ValueError(f"{node.func.attr}() is not allowed")
            for keyword in node.keywords:
                if keyword.arg == "inplace" and isinstance(keyword.value, ast.Constant) and keyword.value.value is True:
                    raise ValueError("inplace=True is not allowed")

        self.generic_visit(node)


def _validate_pandas_code(code):
    tree = ast.parse(code, mode="exec")
    PandasCodeValidator().visit(tree)

    assigned_names = {
        node.id for statement in tree.body for node in ast.walk(statement) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    }
    if "result" not in assigned_names:
        raise ValueError("The code must assign its final value to `result`")

    return tree


def _serialize_result(result):
    if isinstance(result, pd.DataFrame):
        preview = result.iloc[:MAX_RESULT_ROWS, :MAX_RESULT_COLUMNS]
        return json.dumps(
            {
                "type": "DataFrame",
                "shape": list(result.shape),
                "truncated": preview.shape != result.shape,
                "data": json.loads(preview.to_json(orient="split", date_format="iso")),
            },
            ensure_ascii=False,
        )

    if isinstance(result, pd.Series):
        preview = result.iloc[:MAX_RESULT_ROWS]
        return json.dumps(
            {
                "type": "Series",
                "length": len(result),
                "name": str(result.name),
                "truncated": len(preview) != len(result),
                "data": json.loads(preview.to_json(date_format="iso")),
            },
            ensure_ascii=False,
        )

    if isinstance(result, pd.Index):
        result = result[:MAX_RESULT_ROWS].tolist()
    elif isinstance(result, np.ndarray):
        result = result.flatten()[:MAX_RESULT_ROWS].tolist()
    elif isinstance(result, (np.integer, np.floating, np.bool_)):
        result = result.item()
    elif isinstance(result, (pd.Timestamp, pd.Timedelta)):
        result = str(result)

    return json.dumps(result, ensure_ascii=False, default=str)


def create_dataframe_tools(df):
    @tool
    def run_pandas(code: str) -> str:
        """Run general read-only pandas code on `df`.

        `pd` and `np` are also available. The code may use multiple statements,
        filtering, grouping, aggregation, pivoting, joining, reshaping, and
        calculations. Assign the value to return to a variable named `result`.
        The dataframe is a deep copy, and imports, private attributes, file I/O,
        plotting, and inplace mutation are blocked.

        Example:
        result = (
            df.groupby("category", dropna=False)["value"]
              .agg(["count", "mean", "median"])
              .sort_values("mean", ascending=False)
        )
        """

        tree = _validate_pandas_code(code)
        namespace = {
            "__builtins__": SAFE_BUILTINS,
            "df": df.copy(deep=True),
            "np": np,
            "pd": pd,
        }
        exec(compile(tree, "<pandas-tool>", "exec"), namespace, namespace)
        return _serialize_result(namespace["result"])

    return [run_pandas]


def create_code_execution_tool(df, selected_plot):
    @tool
    def execute_plot_code(code: str) -> str:
        """Execute plotting code with `df`, `selected_plot`, and `graph_file_path` available.

        Returns the execution error and whether a non-empty image was created.
        """

        with tempfile.TemporaryDirectory() as temp_dir:
            graph_file_path = os.path.join(temp_dir, "graph.png")
            namespace = {
                "df": df.copy(deep=True),
                "selected_plot": selected_plot,
                "graph_file_path": graph_file_path,
                "__builtins__": __builtins__,
            }
            error = None
            try:
                exec(compile(code, "<plot-code-tool>", "exec"), namespace, namespace)
            except Exception as exception:
                error = f"{type(exception).__name__}: {exception}"

            image_created = os.path.isfile(graph_file_path) and os.path.getsize(graph_file_path) > 0
            if error is None and not image_created:
                error = "Generated code did not save an image"

            return json.dumps({"error": error, "image_created_successfully": error is None and image_created})

    return execute_plot_code


def invoke_with_tools(llm, messages, df, response_format=None, config=None, selected_plot=None):
    """Invoke an LLM and execute any requested dataframe analysis."""

    tools = create_dataframe_tools(df)
    if selected_plot is not None:
        tools.append(create_code_execution_tool(df, selected_plot))
    tools_by_name = {dataframe_tool.name: dataframe_tool for dataframe_tool in tools}
    bind_kwargs = {}
    if response_format is not None:
        bind_kwargs = {"response_format": response_format, "strict": True}
    tool_llm = llm.bind_tools(tools, **bind_kwargs)
    history = list(messages) if isinstance(messages, list) else [HumanMessage(content=messages)]

    for _ in range(MAX_TOOL_CALLS):
        response = tool_llm.invoke(history, config=config)
        history.append(response)

        if not response.tool_calls:
            return response

        for tool_call in response.tool_calls:
            dataframe_tool = tools_by_name.get(tool_call["name"])
            if dataframe_tool is None:
                result = f"Unknown tool: {tool_call['name']}"
            else:
                try:
                    result = dataframe_tool.invoke(tool_call.get("args", {}))
                except Exception as error:
                    result = f"Tool error: {error}"

            history.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call["id"],
                )
            )

    history.append(HumanMessage(content="Tool-call limit reached. Return the requested final answer now."))
    return tool_llm.invoke(history, config=config)
