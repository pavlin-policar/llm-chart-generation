"""Regression checks for structured tool outcomes and recoverable failures."""

import json
import unittest

import pandas as pd
from langchain_core.messages import AIMessage, ToolMessage

from generation_pipeline.generation.tools import (
    create_code_execution_tool,
    create_dataframe_tools,
    invoke_with_tools,
)


class ToolOutcomeTests(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({"value": [1, 2, 6]})
        self.pandas_tool = create_dataframe_tools(self.df)[0]

    def run_code(self, code):
        return json.loads(self.pandas_tool.invoke({"code": code}))

    def test_success_preserves_scalar_and_dataframe_results(self):
        self.assertEqual(self.run_code("result = df['value'].mean()"), {"status": "ok", "result": 3.0})
        outcome = self.run_code("result = df")
        self.assertEqual(outcome["status"], "ok")
        self.assertEqual(outcome["result"]["type"], "DataFrame")
        self.assertEqual(outcome["result"]["shape"], [3, 1])
        self.assertEqual(outcome["result"]["data"]["data"], [[1], [2], [6]])

    def test_validation_failures_include_actionable_errors(self):
        cases = [
            ("df['value'].mean()", "ValueError", "result"),
            ("import pandas as pd\nresult = len(df)", "ValueError", "already available"),
            ("print(df['value'].mean())", "ValueError", "Assign the value"),
            ("result = (", "SyntaxError", "never closed"),
            ("result = df.to_csv('out.csv')", "ValueError", "not allowed"),
        ]
        for code, error_type, message in cases:
            with self.subTest(code=code):
                outcome = self.run_code(code)
                self.assertEqual(outcome["status"], "error")
                self.assertEqual(outcome["error_code"], "invalid_code")
                self.assertEqual(outcome["error_type"], error_type)
                self.assertIn(message, outcome["message"])
                self.assertNotIn("result", outcome)

    def test_runtime_failure_returns_original_exception(self):
        outcome = self.run_code("result = df['missing'].mean()")
        self.assertEqual(outcome["status"], "error")
        self.assertEqual(outcome["error_code"], "execution_error")
        self.assertEqual(outcome["error_type"], "KeyError")
        self.assertIn("missing", outcome["message"])

    def test_execution_uses_a_copy_and_preserves_preview_limits(self):
        self.run_code("df['value'] = 0\nresult = df['value'].sum()")
        self.assertEqual(self.df["value"].tolist(), [1, 2, 6])
        outcome = self.run_code("result = pd.DataFrame({'value': range(101)})")
        self.assertTrue(outcome["result"]["truncated"])
        self.assertEqual(outcome["result"]["shape"], [101, 1])
        self.assertEqual(len(outcome["result"]["data"]["data"]), 100)

    def test_plot_outcomes_keep_image_status_fields(self):
        plot_tool = create_code_execution_tool(self.df, {})
        for code, error_code in [("raise RuntimeError('bad plot')", "execution_error"), ("pass", "image_not_created")]:
            with self.subTest(code=code):
                outcome = json.loads(plot_tool.invoke({"code": code}))
                self.assertEqual(outcome["status"], "error")
                self.assertEqual(outcome["error_code"], error_code)
                self.assertFalse(outcome["image_created_successfully"])
                self.assertTrue(outcome["error"])
                self.assertTrue(outcome["message"])
        outcome = json.loads(plot_tool.invoke({"code": "with open(graph_file_path, 'wb') as image:\n    image.write(b'test')"}))
        self.assertEqual(outcome["status"], "ok")
        self.assertTrue(outcome["image_created_successfully"])
        self.assertIsNone(outcome["error"])

    def test_tool_loop_reports_errors_and_allows_a_corrected_call(self):
        calls = [
            {"name": "unknown", "args": {}, "id": "unknown"},
            {"name": "run_pandas", "args": {}, "id": "invalid-args"},
            {"name": "run_pandas", "args": {"code": "df['value'].mean()"}, "id": "invalid-code"},
            {"name": "run_pandas", "args": {"code": "result = df['value'].mean()"}, "id": "corrected"},
        ]

        class FakeLLM:
            def __init__(self):
                self.responses = iter([AIMessage(content="", tool_calls=[call]) for call in calls] + [AIMessage(content="done")])

            def bind_tools(self, tools):
                return self

            def invoke(self, history, config=None):
                return next(self.responses)

        response, history = invoke_with_tools(FakeLLM(), "Calculate the mean", self.df, return_history=True)
        self.assertEqual(response.content, "done")
        messages = [message for message in history if isinstance(message, ToolMessage)]
        self.assertEqual([message.tool_call_id for message in messages], [call["id"] for call in calls])
        outcomes = [json.loads(message.content) for message in messages]
        self.assertEqual([outcome.get("error_code") for outcome in outcomes], ["unknown_tool", "tool_invocation_error", "invalid_code", None])
        self.assertEqual(outcomes[-1], {"status": "ok", "result": 3.0})


if __name__ == "__main__":
    unittest.main()
