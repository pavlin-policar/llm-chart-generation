import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, PropertyMock, patch

import pandas as pd
from openml.exceptions import OpenMLServerError, OpenMLServerException
from requests import Response
from requests.exceptions import ConnectionError, HTTPError


HELPERS_PATH = Path(__file__).resolve().parents[1] / "generation" / "helpers.py"
spec = importlib.util.spec_from_file_location("openml_pull_helpers", HELPERS_PATH)
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)


def http_error(status):
    response = Response()
    response.status_code = status
    return HTTPError(f"HTTP {status}", response=response)


def dataset():
    return SimpleNamespace(
        format="ARFF",
        name="sample",
        description="Sample data",
        features={0: {"name": "x", "data_type": "numeric", "is_target": False}},
        default_target_attribute="target",
        get_data=Mock(return_value=(pd.DataFrame({"x": [1, 2]}), pd.Series([0, 1]), [], [])),
    )


class OpenMLPullTests(unittest.TestCase):
    def setUp(self):
        self.sleep = patch.object(helpers.time, "sleep").start()
        patch.object(helpers, "uniform", return_value=0).start()
        patch("builtins.print").start()
        self.addCleanup(patch.stopall)

    def test_wrapped_503_retries_same_fixed_id_until_fifth_attempt(self):
        errors = [OpenMLServerError("Unexpected server error\nStatus code: 503") for _ in range(4)]
        with patch.object(helpers.openml.datasets, "get_dataset", side_effect=errors + [dataset()]) as fetch:
            did, frame = helpers.get_random_ds(None, None, preselect_id=26367)
        self.assertEqual(did, 26367)
        self.assertEqual(frame["target"].tolist(), [0, 1])
        self.assertEqual([call.args for call in fetch.call_args_list], [(26367,)] * 5)
        self.assertEqual([call.args[0] for call in self.sleep.call_args_list], [2, 4, 8, 16])

    def test_download_503_retries_data_fetch(self):
        source = dataset()
        successful_data = source.get_data.return_value
        source.get_data.side_effect = [http_error(503), successful_data]
        with patch.object(helpers.openml.datasets, "get_dataset", return_value=source):
            _, frame = helpers.get_random_ds(None, None, preselect_id=26367)
        self.assertEqual(source.get_data.call_count, 2)
        self.assertEqual(frame.shape, (2, 2))

    def test_metadata_and_lazy_feature_fetch_retry(self):
        source = dataset()
        features = source.features
        lazy_source = Mock(name=source.name, description=source.description)
        lazy_source.name = source.name
        lazy_source.description = source.description
        type(lazy_source).features = PropertyMock(side_effect=[http_error(503), features])
        with patch.object(helpers.openml.datasets, "get_dataset", side_effect=[ConnectionError("temporary"), lazy_source, lazy_source]) as fetch:
            result = helpers.get_dataset_semantics(26367, sleep_s=1)
        self.assertEqual(fetch.call_count, 3)
        self.assertEqual(result["features"], list(features.values()))
        self.assertEqual([call.args[0] for call in self.sleep.call_args_list], [2, 4, 1])

    def test_catalog_retries_503(self):
        with patch.object(helpers.openml.datasets, "list_datasets", side_effect=[http_error(503), {2: {"name": "sample"}}]) as fetch:
            result = helpers.openml_list_uci()
        self.assertEqual(fetch.call_count, 2)
        self.assertEqual(result, [{"name": "sample", "did": 2}])

    def test_exhaustion_raises_after_five_attempts(self):
        error = http_error(503)
        with patch.object(helpers.openml.datasets, "get_dataset", side_effect=error) as fetch:
            with self.assertRaises(HTTPError) as caught:
                helpers.get_random_ds(None, None, preselect_id=26367)
        self.assertIs(caught.exception, error)
        self.assertEqual(fetch.call_count, 5)
        self.assertEqual(self.sleep.call_count, 4)

    def test_transient_http_statuses_retry(self):
        for status in (408, 429, 500, 502, 503, 504):
            with self.subTest(status=status):
                operation = Mock(side_effect=[http_error(status), "ok"])
                self.assertEqual(helpers._retry_openml_pull(operation, "pull"), "ok")
                self.assertEqual(operation.call_count, 2)

    def test_permanent_errors_do_not_retry(self):
        errors = [http_error(400), http_error(403), http_error(404),
                  OpenMLServerException("Dataset does not exist", code=111),
                  OpenMLServerException("Application error", code=503),
                  OpenMLServerError("Status code: 404"),
                  ValueError("Malformed dataset")]
        for error in errors:
            with self.subTest(error=str(error)):
                self.sleep.reset_mock()
                operation = Mock(side_effect=error)
                with self.assertRaises(type(error)):
                    helpers._retry_openml_pull(operation, "pull")
                self.assertEqual(operation.call_count, 1)
                self.sleep.assert_not_called()

    def test_unsupported_format_does_not_retry(self):
        source = dataset()
        source.format = "SPARSE_ARFF"
        with patch.object(helpers.openml.datasets, "get_dataset", return_value=source) as fetch:
            with self.assertRaisesRegex(RuntimeError, "SPARSE_ARFF"):
                helpers.get_random_ds(None, None, preselect_id=26367)
        self.assertEqual(fetch.call_count, 1)
        source.get_data.assert_not_called()
        self.sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
