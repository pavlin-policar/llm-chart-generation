"""Execute generated plotting code without changing the source dataframe."""
import copy
import json
import os


def execute_graph_code(code, df, selected_plot, graph_file_path):
    if os.path.exists(graph_file_path):
        os.remove(graph_file_path)

    # Each attempt starts from the source, even if an earlier attempt failed
    # after changing its working data. Copy axis storage as well: pandas can
    # otherwise share the underlying index/column arrays across deep copies.
    plot_df = df.copy(deep=True)
    # Pandas deep copies do not recursively copy containers in object cells.
    for position, dtype in enumerate(df.dtypes):
        if dtype == object:
            plot_df.isetitem(position, df.iloc[:, position].map(copy.deepcopy).astype(object))
    plot_df.index = df.index.copy(deep=True)
    plot_df.columns = df.columns.copy(deep=True)
    exec_namespace = {
        "df": plot_df,
        "selected_plot": selected_plot,
        "graph_file_path": graph_file_path,
        "__builtins__": __builtins__,
    }

    exec(code, exec_namespace, exec_namespace)

    if not os.path.exists(graph_file_path):
        raise ValueError("Generated code did not save image")

    graph_data = exec_namespace.get("graph_data")
    graph_df = exec_namespace.get("graph_df")

    if not isinstance(graph_data, dict):
        raise ValueError("Generated code did not define graph_data as a dictionary")

    try:
        json.dumps(graph_data)
    except (TypeError, ValueError) as error:
        raise ValueError("graph_data is not JSON-serializable") from error

    if not isinstance(graph_df, type(df)):
        raise ValueError("Generated code did not define graph_df as a pandas DataFrame")

    # Preserve the actual processed data, including intentional changes to the
    # plotting copy, for description generation, questions, and data validation.
    return graph_data, graph_df
