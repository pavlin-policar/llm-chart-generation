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

API_URL = "http://ixb1:8000/v1"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        type=int,
        default=10,
        help="Number of datasets to generate graphs for.",
    )
    return parser.parse_args()

def select_dataset(datasets_meta, rng, llm, known_ids):
    while True:
        try:
            print("Fetching random dataset...")
            dataset_id, df = get_random_ds(datasets_meta, rng)
            if dataset_id in known_ids:
                continue

            dataset_sem = get_dataset_semantics(dataset_id, sleep_s=1.0)

            if dataset_sem.get("features") is None:
                dataset_sem["features"] = ""

            print("Getting usability...")
            usability = determine_dataset_usability_call(
                llm,
                dataset_sem,
            )

            if not usability["useful"]:
                print(f"Dataset {dataset_id} deemed not useful, picking another...")
                continue

            return dataset_id
        except Exception as error:
            print(f"Error fetching dataset, retrying... {error}")

if __name__ == "__main__":
    main_dir = Path(__file__).resolve().parent.parent.parent
    datasets_good_file = os.path.join(main_dir, "generation_pipeline", "generation", "configs", "good_datasets.jsonl")

    llm_think = ChatOpenAI(
        model="qwen3.5",
        openai_api_key="EMPTY",
        openai_api_base=API_URL,
        extra_body={
            "chat_template_kwargs": {"enable_thinking": True},
            "logit_bias": {
                "248069": 5.0,
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
        while len(datasets) < args.datasets:
            dataset_id = select_dataset(datasets_meta, rng, llm_think, id_lookups)

            print(dataset_id)

            datasets.append({
                "id": dataset_id
            })
            id_lookups.append(dataset_id)

            f.write(json.dumps(datasets[-1]) + "\n")
            f.flush()
            



    
