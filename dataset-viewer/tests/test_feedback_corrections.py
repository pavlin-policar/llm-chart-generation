"""Checks for the correction counts shown by both dataset viewers."""

import json
import sys
import tempfile
import unittest
from pathlib import Path


VIEWER_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VIEWER_DIR))
sys.path.insert(0, str(VIEWER_DIR / "web"))

from feedback_corrections import correction_rows, summarize_error_corrections
from prepare_ftp_bundle import build_generation_metrics


def image(iteration, types=None, *, structured=True):
    result = {"path": f"images/chart_it{iteration}.png"}
    if structured:
        result["errors"] = [{"type": error_type} for error_type in (types or [])]
    return result


class CorrectionStatisticsTests(unittest.TestCase):
    def test_persistence_recurrence_and_missing_feedback(self):
        records = [
            {"images": [
                image(2),
                image(0, ["layout", "layout", "label"]),
                image(1, ["layout"]),
                image(2, ["layout"]),  # duplicate path is ignored
                image(3, ["layout"]),
                image(4),
            ]},
            {"images": [image(0, ["unknown"]), image(1, structured=False), image(2)]},
            {"images": [image(0, ["unresolved"])]},
        ]
        totals = summarize_error_corrections(records)

        self.assertEqual(totals["layout"]["episodes"], 2)
        self.assertEqual(totals["layout"]["corrected"], 2)
        self.assertEqual(totals["layout"]["by_revisions"], {2: 1, 1: 1})
        self.assertEqual(totals["label"]["by_revisions"], {1: 1})
        self.assertEqual(totals["unknown"]["corrected"], 0)
        self.assertEqual(totals["unresolved"]["corrected"], 0)

        rows = correction_rows([{"feedback_corrections": totals}])
        layout = next(row for row in rows if row["Error type"] == "layout")
        self.assertEqual(layout["Corrected share"], "100.0%")
        self.assertEqual(layout["1 revision"], 1)
        self.assertEqual(layout["2 revisions"], 1)
        self.assertEqual(layout["Mean revisions"], 1.5)

    def test_web_bundle_metrics_include_corrections(self):
        records = [{"images": [image(0, ["layout"]), image(1)]}]
        with tempfile.TemporaryDirectory() as temp:
            metrics = build_generation_metrics("example", records, Path(temp))
        self.assertEqual(metrics["feedback_corrections"]["layout"]["corrected"], 1)
        self.assertEqual(metrics["feedback_corrections"]["layout"]["by_revisions"], {1: 1})
        manifest_metrics = json.loads(json.dumps(metrics))
        self.assertEqual(correction_rows([manifest_metrics])[0]["1 revision"], 1)


if __name__ == "__main__":
    unittest.main()
