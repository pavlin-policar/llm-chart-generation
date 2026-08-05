import openml
import time

import re

def after_think(text: str):
    """
    Split reasoning and response returned by Qwen.
    """
    athink = text.split("</think>", 1)[1] if "<think>" in text else text
    think = text.split("</think>", 1)[0] if "<think>" in text else None
    return think, athink

def strip_code_fences(s: str) -> str:
    """
    Strip python and json tags from LLM output.
    """
    s = s.strip()
    s = re.sub(r"^```(?:json|python)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    return s.strip()


# DATASET HELPERS

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

def get_random_ds(d_meta, rng, preselect_id=None):
    """
    Pick a random UCI dataset from metadata (list of dicts),
    ensure ARFF format, and load the data as a pandas DataFrame.
    """

    # Pick dataset id using existing helper
    if preselect_id is None:
        data_id = pick_random_dataset_id(d_meta, rng=rng)
    else:
        data_id = preselect_id

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