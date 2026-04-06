import json
import os
from urllib.parse import urlencode
from urllib.request import urlopen, Request
import base64

import time
import uuid
import re
import gc 

from pathlib import Path

import argparse

import glob

from PIL import Image
from io import BytesIO

from tqdm import tqdm

from vllm import LLM, SamplingParams
from transformers import AutoProcessor

def split_think(text:str):
    if "</think>" in text:
        think, athink = text.split("</think>", 1)
        think = think.replace("<think>", "").strip()

        athink = athink.strip()
        
        return think, athink

    return None, text

def format_messages(messages, data_dir):
    msgs_out = []
    images = []

    for i, m in enumerate(messages):
        content = f"Answer this question about the given chart and the description of the data used to make it.\n\nDataset description: {m['dataset_description']}\n\nQuestion:\n{ m['question']['question'] }"
        img = Image.open(os.path.join(data_dir, m["image"]))

        print(i, os.path.getsize(os.path.join(data_dir, m["image"])))

        images.append(img)

        prompt = [{"type": "text", "text": content},
            {"type": "image", "image": img}]

        msgs_out.append([{"role":"system", "content":"You are a researcher that answers questions about charts.\nKeep answers brief and in plain english. Return ONLY the answer and nothing else."}, {"role": "user", "content": prompt}])

    return msgs_out, images

def format_messages_check(messages):
    msgs_out = []

    system_prompt = ("You are a teacher that has to grade students answers about charts.\n"
    "You will be given a question, a REAL answer (the ground truth) and the STUDENT's answer. You will also be given context about the dataset.\n"
    "Compare the two answers and respond with true if the answers are similar enough for it to be considered correct, or false if not.\n"
    "Respond only with ONE word.\n"
    "When dealing with numerical values consider an answer correct if the students value is in the ballpark of the actual value. Especially if it could be easily slightly misread (e.g size of bar is 1005 but student says 1000)")

    for m in messages:
        content = f"Would you say the student answered this question correctly?\n\nDataset description: {m['dataset_description']}\n\n Question: {m['question']['question']}\nREAL answer: {m['question']['answer']}\nSTUDENT answer: {m.get('test_answer')}"
        prompt = [{"type": "text", "text": content}]

        msgs_out.append([{"role":"system", "content":system_prompt}, {"role": "user", "content": prompt}])

    return msgs_out

def batched(seq, batch_size):
    for i in range(0, len(seq), batch_size):
        yield seq[i:i + batch_size]

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--metadata_file", type=str, required=False, default="metadata.jsonl")
    parser.add_argument("--model_path", type=str, required=False, default="/workspace/models")
    parser.add_argument("--model_name", type=str, required=False, default="qwen3.5-9b")
    parser.add_argument("--check_model_name", type=str, required=False, default="qwen3.5-9b")
    parser.add_argument("--thinking", action="store_true", default=False)
    parser.add_argument("--batch_size", type=int, required=False, default=32)

    args = parser.parse_args()

    model_path = os.path.join(args.model_path, args.model_name)
    model_path_check = os.path.join(args.model_path, args.check_model_name)

    MAIN_DIR = Path(__file__).resolve().parent
    DATASET_FOLDER = os.path.join(MAIN_DIR, "dataset")
    IMAGES_FOLDER = os.path.join(DATASET_FOLDER, "images")
    EVAL_FOLDER = os.path.join(MAIN_DIR, "evaluation")

    with open(os.path.join(DATASET_FOLDER, args.metadata_file)) as f:
        
        questions_all = []

        for line in tqdm(f):
            graph = json.loads(line)

            image = graph["images"][-1]
            questions = graph["graph"]["questions"]
            dataset_desc = graph["dataset"]["description"]

            for q in questions:
                questions_all.append({
                    "graph_id": graph["id"],
                    "graph_type": graph["graph"]["type"],
                    "image": image["path"],
                    "question": q,
                    "dataset_description": dataset_desc 
                })

    processor_test = AutoProcessor.from_pretrained(model_path)
    llm_test = LLM(
        model=model_path,
        gpu_memory_utilization=0.92,
        max_num_seqs=32,
        max_model_len=20000,
        max_num_batched_tokens=16384*2,
        tensor_parallel_size=2
    )

    inputs = []

    batch_size = args.batch_size

    for batch_start, batch in enumerate(batched(questions_all, batch_size)):
        formatted, images = format_messages(batch, DATASET_FOLDER)

        inputs = []

        for messages, image in zip(formatted, images):
            prompt = processor_test.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=args.thinking,
            )

            req = {
                "prompt": prompt,
                "multi_modal_data": {
                    "image": [image]
                }
            }

            inputs.append(req)

        outputs = llm_test.generate(
            inputs,
            SamplingParams(temperature=0.6, max_tokens=None)
        )

        for item, out in zip(batch, outputs):
            think, answer = split_think(out.outputs[0].text)
            item["test_answer"] = answer

        print(think)

    del formatted, images, inputs, outputs

    del llm_test
    gc.collect()

    processor_check = AutoProcessor.from_pretrained(model_path_check)
    llm_check = LLM(
        model=model_path_check,
        gpu_memory_utilization=0.92,
        max_num_seqs=32,
        max_model_len=20000,
        max_num_batched_tokens=16384 * 2,
    )

    batch_size = args.batch_size

    for batch in batched(questions_all, batch_size):
        formatted = format_messages_check(batch)

        inputs_check = []

        for messages in formatted:
            prompt = processor_check.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )

            req = {"prompt": prompt}
            inputs_check.append(req)

        outputs = llm_check.generate(
            inputs_check,
            SamplingParams(temperature=0, max_tokens=32)
        )

        for item, out in zip(batch, outputs):
            think, answer = split_think(out.outputs[0].text)
            ans = answer.strip().lower()

            if ans == "true":
                item["correct"] = True
            elif ans == "false":
                item["correct"] = False
            else:
                item["correct"] = None

    del formatted, inputs_check, outputs
    del llm_check
    gc.collect()

    with open(os.path.join(EVAL_FOLDER, f"{args.model_name}{'_reasoning' if args.thinking else ''}.jsonl"), "a+") as f:
        for i, q in enumerate(questions_all):
            f.write(json.dumps(questions_all[i]) + "\n")
            

    stats_dict = {
        "metadata": {
            "correct": 0,
            "incorrect": 0
        },
        "value extraction": {
            "correct": 0,
            "incorrect": 0
        },
        "comparison": {
            "correct": 0,
            "incorrect": 0
        },
        "trends": {
            "correct": 0,
            "incorrect": 0
        },
        "reasoning": {
            "correct": 0,
            "incorrect": 0
        },
        "full": {
            "correct": 0,
            "incorrect": 0
        },
    }


    with open(os.path.join(EVAL_FOLDER, f"{args.model_name}{'_reasoning' if args.thinking else ''}_params.json"), "w+") as f:
        for q in questions_all:
            qtype = q["question"].get("type")
            if q.get("correct") is True:
                if qtype in stats_dict.keys():
                    stats_dict[qtype]["correct"] += 1
                stats_dict["full"]["correct"] += 1
            elif q.get("correct") is False:
                if qtype in stats_dict.keys():
                    stats_dict[qtype]["incorrect"] += 1
                stats_dict["full"]["incorrect"] += 1

        for k in stats_dict.keys():
            c = stats_dict[k]["correct"]
            ic = stats_dict[k]["incorrect"]

            total = c+ic

            stats_dict[k]["accuracy"] = c / total if total > 0 else None

        f.write(json.dumps(stats_dict, indent=2))


        
        



        

