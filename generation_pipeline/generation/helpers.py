import openml
import time

import re

from random import uniform
from openml.exceptions import OpenMLServerError
from requests.exceptions import ChunkedEncodingError, ConnectionError, HTTPError, Timeout

def after_think(text: str):
    """
    Split reasoning and response returned by Qwen.
    """
    athink = text.split("</think>", 1)[1] if "<think>" in text else text
    think = text.split("</think>", 1)[0] if "<think>" in text else None
    return think, athink

def strip_code_fences(s: str) -> str:
    """
    If a ```python, ```json, or plain ``` code block is present,
    return only the content inside it.
    Otherwise, return the original stripped string.
    """
    s = s.strip()

    match = re.search(
        r"```(?:python|json)?\s*(.*?)\s*```",
        s,
        flags=re.DOTALL,
    )

    if match:
        return match.group(1).strip()

    return s

# DATASET HELPERS

OPENML_PULL_ATTEMPTS = 5
RETRYABLE_HTTP_STATUSES = {408, 429, 500, 502, 503, 504}


def _is_retryable_pull_error(error):
    if isinstance(error, (ConnectionError, Timeout, ChunkedEncodingError)):
        return True

    if isinstance(error, HTTPError):
        response = error.response
        return response is not None and response.status_code in RETRYABLE_HTTP_STATUSES

    if isinstance(error, OpenMLServerError):
        # OpenML wraps non-XML HTTP errors and records their status in the text.
        # OpenMLServerException.code is an API error code, not an HTTP status.
        match = re.search(r"\bStatus\s+code\s*:\s*(\d{3})\b", str(error), re.IGNORECASE)
        return match is not None and int(match.group(1)) in RETRYABLE_HTTP_STATUSES

    return False


def _retry_openml_pull(operation, description):
    for attempt in range(OPENML_PULL_ATTEMPTS):
        try:
            return operation()
        except (ConnectionError, Timeout, ChunkedEncodingError, HTTPError, OpenMLServerError) as error:
            if not _is_retryable_pull_error(error) or attempt == OPENML_PULL_ATTEMPTS - 1:
                raise

            delay = 2 ** (attempt + 1) + uniform(0, 1)
            print(
                f"{description} failed "
                f"(attempt {attempt + 1}/{OPENML_PULL_ATTEMPTS}): {error}. "
                f"Retrying in {delay:.1f}s...",
                flush=True,
            )
            time.sleep(delay)


def openml_list_uci(status="active"):
    """
    Return OpenML catalog metadata sorted by dataset ID for random selection.

    The historical function name does not imply a UCI tag filter.
    """
    ds_dict = _retry_openml_pull(
        lambda: openml.datasets.list_datasets(status=status),
        "OpenML dataset catalog fetch",
    )

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
    
    print("Filtered length:", len(filtered))

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
    def fetch_semantics():
        ds = openml.datasets.get_dataset(int(did))

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

    semantics = _retry_openml_pull(fetch_semantics, f"Dataset {did} metadata fetch")
    if sleep_s and sleep_s > 0:
        time.sleep(sleep_s)
    return semantics

def get_random_ds(d_meta, rng, preselect_id=None):
    """
    Load an ARFF dataset as a pandas DataFrame.

    Random selection requires catalog metadata. A preselected ID is fetched
    directly, so d_meta and rng can be None in that case.
    """

    # Pick dataset id using existing helper
    if preselect_id is None:
        data_id = pick_random_dataset_id(d_meta, rng=rng)
    else:
        data_id = int(preselect_id)

    def fetch_data():
        ds = openml.datasets.get_dataset(data_id)

        fmt = str(ds.format).upper()
        if fmt != "ARFF":
            raise RuntimeError(f"Picked dataset {data_id} is not in ARFF format (got {fmt}).")

        X, y, categorical_indicator, attribute_names = ds.get_data(
            dataset_format="dataframe"
        )

        # Match old behavior: return full data including target column
        if y is not None:
            data = X.copy()
            data[ds.default_target_attribute] = y
            return data
        return X

    data = _retry_openml_pull(fetch_data, f"Dataset {data_id} fetch")
    return data_id, data
