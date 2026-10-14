import argparse
import json
import os
import time
import uuid
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from calls import (
    determine_dataset_usability_call,
)
from helpers import get_dataset_semantics, get_random_ds, openml_list_uci
from langchain_openai import ChatOpenAI

API_URL = "http://ixb1:8888/v1"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        type=int,
        default=3000,
        help="Number of datasets to select.",
    )
    return parser.parse_args()

if __name__ == "__main__":
    main_dir = Path(__file__).resolve().parent.parent.parent
    datasets_good_file = os.path.join(main_dir, "generation_pipeline", "generation", "configs", "good_datasets.jsonl")

    llm_think = ChatOpenAI(
        model="qwen3.8",
        openai_api_key="EMPTY",
        openai_api_base=API_URL,
        extra_body={
            "chat_template_kwargs": {
                "enable_thinking": True,
                "reasoning_effort": "medium",
            },
            "logit_bias": {
                "248069": 3.0,
            },
        },
    )

    args = parse_args()

    rng = np.random.default_rng()

    datasets_meta = openml_list_uci()

    datasets = []
    id_lookups = []

    if os.path.exists(datasets_good_file):
        with open(datasets_good_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    id_lookups.append(item["id"])

    with open(datasets_good_file, "a+", encoding="utf-8") as f:
        for meta in datasets_meta:
            if len(datasets) >= args.datasets:
                break

            try:
                dataset_id = int(meta["did"])
                if dataset_id <= 43071:
                    continue

                if dataset_id in id_lookups:
                    continue

                n = int(float(meta.get("NumberOfInstances", 0)))
                p = int(float(meta.get("NumberOfFeatures", 0)))
                if n < 200 or not 1 <= p <= 2000:
                    continue
                if str(meta.get("format", "")).upper() != "ARFF":
                    continue

                print(f"Fetching dataset {dataset_id}...")
                dataset_id, df = get_random_ds(
                    [meta], rng=None, preselect_id=dataset_id
                )

                dataset_sem = get_dataset_semantics(dataset_id, sleep_s=1.0)
                if dataset_sem.get("features") is None:
                    dataset_sem["features"] = ""

                usability = determine_dataset_usability_call(
                    llm_think,
                    dataset_sem,
                    df=df,
                    use_tools=True,
                )
                if not usability["useful"]:
                    print(f"Dataset {dataset_id} deemed not useful, skipping...")
                    continue

                print(dataset_id)

                datasets.append({
                    "id": dataset_id
                })
                id_lookups.append(dataset_id)

                f.write(json.dumps(datasets[-1]) + "\n")
                f.flush()

            except Exception as error:
                print(f"Error processing dataset, skipping... {error}")