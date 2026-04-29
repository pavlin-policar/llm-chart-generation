import argparse
import base64
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

from PIL import Image
from io import BytesIO
import math


def split_think(text: str):
    """
    Used for initial testing of Qwen style models with reasoning.

    This method is not important anymore as the models we are testing models without reasoning.
    Kept in the code as it doesn't really hurt.
    """
    if "</think>" in text:
        think, athink = text.split("</think>", 1)
        think = think.replace("<think>", "").strip()
        athink = athink.strip()
        return think, athink
    return None, text


def estimate_tokens(text: str):
    """
    Estimate tokens for cost saving purposes
    """
    return len(text) // 4

def estimate_openai_image_tokens(img_bytes: bytes):
    """
    Estimate image tokens for cost saving purposes.
    """

    img = Image.open(BytesIO(img_bytes))
    width, height = img.size

    return 167 + 0.000972 * width * height

def format_messages(messages, data_dir, quest, prev_toks = None):
    """
    Format prompts into proper format to send over an OpenAi style API to the evaluated model.
    """

    msgs_out = []

    static_instruction = (
        "Answer this question about the given chart and the description of the data used to make it.\n"
        "Keep the answer brief and in plain English. Return only the answer."
    )

    for m in messages:
        img_path = os.path.join(data_dir, m["image"])
        ext = img_path.split(".")[-1].lower()
        mime_type = f"image/{ext}" if ext in ["png", "jpg", "jpeg", "gif", "webp"] else "image/jpeg" # Image types in our dataset will generaly be .png

        with open(img_path, "rb") as f:
            img_bytes = f.read()

        base64_img = base64.b64encode(img_bytes).decode()
        img_url = f"data:{mime_type};base64,{base64_img}" # Format image base64 URL for sending over API

        if prev_toks is None:
            toks = estimate_tokens(f"Dataset description:\n{m['dataset_description']}" + static_instruction)
            toks_image = estimate_openai_image_tokens(img_bytes)

            prev_toks = toks + toks_image

        prompt = [
            {
                "role": "system",
                "content": "You are a researcher that answers questions about charts.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": static_instruction},
                    {"type": "text", "text": f"Dataset description:\n{m['dataset_description']}"},
                    {"type": "image_url", "image_url": {"url": img_url, "detail": "high"}},
                ],
            },
        ]

        # This is used for debugging.
        if quest:
            prompt.append(
                {
                    "role": "user",
                    "content": f"Question:\n{m['question']['question']}",
                }
            )

        msgs_out.append(prompt)

    return msgs_out, prev_toks

def format_messages_check(messages):
    """
    Format prompts into proper format to send over an OpenAi style API to the checking model.
    """

    msgs_out = []

    system_prompt = (
        "You are a teacher that has to grade students answers about charts.\n"
        "You will be given a question, a REAL answer (the ground truth) and the STUDENT's answer. "
        "You will also be given context about the dataset.\n"
        "Compare the two answers and respond with true if the answers are similar enough for it to be "
        "considered correct, or false if not.\n"
        "Respond only with ONE word. Either true or false\n"
        "When dealing with numerical values consider an answer correct if the students value is in the "
        "ballpark of the actual value. Especially if it could be easily slightly misread "
        "(e.g size of bar is 1005 but student says 1000)"
    )

    for m in messages:
        content = (
            f"Would you say the student answered this question correctly?\n\n"
            f"Dataset description: {m['dataset_description']}\n\n"
            f"Question: {m['question']['question']}\n"
            f"REAL answer: {m['question']['answer']}\n"
            f"STUDENT answer: {m.get('test_answer')}"
        )

        msgs_out.append(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ]
        )

    return msgs_out


def call_model(client, model_name, messages, temperature, max_tokens, retries=3, sleep_base=2, key="", max_length=1000):
    """
    Call the model
    
    In 'extra_body' of the chat completions you fix the providers for the models to maximize caching as OpenRouter sticky routing sometimes fails.
    """
    
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                prompt_cache_key=key,
                extra_body={
                    "reasoning": {"enabled": False}, 
                    #"provider": {
                    #    "order": ["OpenAI", "alibaba"],
                    #    "allow_fallbacks": False
                    #}
                }
            )
            answer = response.choices[0].message.content

            usage = response.usage

            input_tokens = getattr(usage, "prompt_tokens", 0) if usage is not None else 0
            output_tokens = getattr(usage, "completion_tokens", 0) if usage is not None else 0
            cost = getattr(usage, "cost", 0) if usage is not None else 0

            prompt_details = getattr(usage, "prompt_tokens_details", None)
            cached_tokens = getattr(prompt_details, "cached_tokens", 0) if prompt_details else 0

            print()

            print(input_tokens, cached_tokens, output_tokens, max_length, max_tokens,  flush=True)

            _, answer = split_think(answer)

            return answer.strip(), input_tokens, output_tokens, cached_tokens, cost
        except Exception as e:
            if attempt == retries - 1:
                raise e
            time.sleep(sleep_base * (attempt + 1))


def solve_one(client, model_name, item, dataset_folder, key, quest = True, prev_toks = None):
    """
    Wrapper for one call of the model thats being evaluated.

    Returns 'inp', 'out', 'cache', 'cost', 'prev_toks' are used to debug and monitor usage.
    """
    
    messages, prev_toks = format_messages([item], dataset_folder, quest, prev_toks)

    messages = messages[0]

    answer, inp, out, cache, cost = call_model(
        client=client,
        model_name=model_name,
        messages=messages,
        temperature=0.6,
        max_tokens= (1000 if quest else 0), # Used for debugging
        key = key, # Cache key for more reliable caching of question blocks.
        max_length = (1000 if quest else 0) # Same here
    )
    return answer, inp, out, cache, cost, prev_toks


def check_one(client, model_name, item):
    """
    Wrapper for one call of the checking model.

    Returns 'inp', 'out', 'cache', 'cost' are used to debug and monitor usage.
    """

    messages = format_messages_check([item])[0]
    answer, inp, out, cache, cost = call_model(
        client=client,
        model_name=model_name,
        messages=messages,
        temperature=0,
        max_tokens=10,
    )
    ans = answer.strip().lower()

    if ans == "true":
        return True, inp, out, cache, cost
    elif ans == "false":
        return False, inp, out, cache, cost
    return None, inp, out, cache, cost

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--metadata_file", type=str, default="metadata.jsonl") # Name of the metadata file in the dataset folder.
    parser.add_argument("--model_name", type=str, default="qwen/qwen3-vl-32b-instruct") # Openrouter name of the evaluated model.
    parser.add_argument("--check_model_name", type=str, default="qwen/qwen3.5-9b") # OpenRouter name of the grading
    parser.add_argument("--max_workers", type=int, default=16) # Maximum amount of workers spawned

    args = parser.parse_args()

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])

    # MAIN_DIR is set to the git repository folder
    MAIN_DIR = Path(__file__).resolve().parent
    DATASET_FOLDER = os.path.join(MAIN_DIR, "dataset")
    EVAL_FOLDER = os.path.join(MAIN_DIR, "evaluation")

    questions_all = []

    # Variables for monitoring cumulative usage
    answer_input_tokens_total = 0
    answer_output_tokens_total = 0
    answer_cache_tokens_total = 0
    cost_answer = 0

    grading_input_tokens_total = 0
    grading_output_tokens_total = 0
    grading_cache_tokens_total = 0
    cost_grading = 0

    # Read the metadatafile and store the neccessary fields.
    with open(os.path.join(DATASET_FOLDER, args.metadata_file), encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading metadata"):
            graph = json.loads(line)

            image = graph["images"][-1]
            questions = graph["graph"]["questions"]
            dataset_desc = graph["dataset"]["description"]

            for q in questions:
                questions_all.append(
                    {
                        "graph_id": graph["id"],
                        "dataset_id": graph["dataset"]["id"],
                        "graph_type": graph["graph"]["type"],
                        "image": image["path"],
                        "question": q,
                        "dataset_description": dataset_desc,
                    }
                )

    last_dataset_id = None
    last_image = None

    prev_toks = None

    # First we get all the proposed answers from the evaluated model
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {}

        for idx, item in enumerate(questions_all):
            is_new_prefix = (
                item["dataset_id"] != last_dataset_id
                or item["image"] != last_image
            )

            if is_new_prefix:
                # If question is from a new dataset, we first wait for all the previous questions to finish (just in case)
                for future in tqdm(as_completed(futures), total=len(futures), desc="Answering", ncols=100):
                    f_idx = futures[future]
                    try:
                        test_answer, input_tokens, output_tokens, cached_tokens, cost, _ = future.result()
                        questions_all[f_idx]["test_answer"] = test_answer

                        answer_input_tokens_total += input_tokens
                        answer_output_tokens_total += output_tokens
                        answer_cache_tokens_total += cached_tokens
                        cost_answer += cost

                        print(f"Answer to question {f_idx}: {test_answer}", flush=True)

                        print(
                            f"{f_idx}, answer totals: in={answer_input_tokens_total}, out={answer_output_tokens_total}, cache={answer_cache_tokens_total}, cost={cost_answer}",
                            flush=True,
                        )
                    except Exception as e:
                        print("Exccption" , e, flush = True)
                        questions_all[f_idx]["test_answer"] = None
                        questions_all[f_idx]["answer_error"] = str(e)

                futures = {}

                # Call LLM for the first question of the dataset, and wait for the response. This is done to cache the input prompt.
                # NOTE: Prompts sometimes don't cache because they are too short
                try:
                    test_answer, input_tokens, output_tokens, cached_tokens, cost, prev_toks_new = solve_one(
                        client, args.model_name, item, DATASET_FOLDER, key=str(item["dataset_id"]) + item["image"], quest = True, prev_toks=None
                    )
                    questions_all[idx]["test_answer"] = test_answer

                    answer_input_tokens_total += input_tokens
                    answer_output_tokens_total += output_tokens
                    answer_cache_tokens_total += cached_tokens
                    cost_answer += cost

                    prev_toks = prev_toks_new

                    print(f"Answer to question {idx}: {test_answer}", flush=True)

                    print(
                        f"{idx}, answer totals: in={answer_input_tokens_total}, out={answer_output_tokens_total}, cache={answer_cache_tokens_total}, cost={cost_answer}",
                        flush=True,
                    )
                except Exception as e:
                    print("Exception", e, flush = True)
                    questions_all[idx]["test_answer"] = None
                    questions_all[idx]["answer_error"] = str(e)

                last_dataset_id = item["dataset_id"]
                last_image = item["image"]

            # Submit all other calls
            else:
                futures[executor.submit(solve_one, client, args.model_name, item, DATASET_FOLDER, key=str(item["dataset_id"]) + item["image"], quest=True, prev_toks=prev_toks)] = idx

        # Wait for all other calls to complete
        for future in tqdm(as_completed(futures), total=len(futures), desc="Answering", ncols=100):
            idx = futures[future]
            try:
                test_answer, input_tokens, output_tokens, cached_tokens, cost, _ = future.result()
                questions_all[idx]["test_answer"] = test_answer

                answer_input_tokens_total += input_tokens
                answer_output_tokens_total += output_tokens
                answer_cache_tokens_total += cached_tokens
                cost_answer += cost

                print(f"Answer to question {idx}: {test_answer}", flush=True)

                print(
                    f"{idx}, answer totals: in={answer_input_tokens_total}, out={answer_output_tokens_total}, cache={answer_cache_tokens_total}, cost={cost_answer}",
                    flush=True,
                )

            except Exception as e:
                print("Exception", e, flush = True)
                questions_all[idx]["test_answer"] = None
                questions_all[idx]["answer_error"] = str(e)

    # Here we get the correctness of the proposed answers.
    # NOTE: Here we don't optimize for caching as the cost for Qwen3.5-9b is somewhat low.
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # Submit all requests to the checking model.
        futures = {executor.submit(check_one, client, args.check_model_name, item): idx for idx, item in enumerate(questions_all)}

        # Wait for the responses
        for future in tqdm(as_completed(futures), total=len(futures), desc="Grading", ncols=100):
            idx = futures[future]
            try:
                correct, input_tokens, output_tokens, cached_tokens, cost = future.result()
                questions_all[idx]["correct"] = correct

                grading_input_tokens_total += input_tokens
                grading_output_tokens_total += output_tokens
                grading_cache_tokens_total += cached_tokens
                cost_grading += cost

                print(
                    f"grading totals: in={grading_input_tokens_total}, out={grading_output_tokens_total}, cache={grading_cache_tokens_total}, cost={cost_grading}",
                    flush=True,
                )

            except Exception as e:
                questions_all[idx]["correct"] = None
                questions_all[idx]["check_error"] = str(e)

    # Make eval folder just in case it doesn't exist
    os.makedirs(EVAL_FOLDER, exist_ok=True)

    # Final token counts
    print("answer_input_tokens_total =", answer_input_tokens_total)
    print("answer_output_tokens_total =", answer_output_tokens_total)
    print("grading_input_tokens_total =", grading_input_tokens_total)
    print("grading_output_tokens_total =", grading_output_tokens_total)

    # Save full data
    with open(os.path.join(EVAL_FOLDER, f"{args.model_name.replace('/', '_')}.jsonl"), "w", encoding="utf-8") as f:
        for q in questions_all:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    # Aggregate accuracy by question type.
    stats_dict = {
        "metadata": {"correct": 0, "incorrect": 0},
        "value extraction": {"correct": 0, "incorrect": 0},
        "comparison": {"correct": 0, "incorrect": 0},
        "trends": {"correct": 0, "incorrect": 0},
        "reasoning": {"correct": 0, "incorrect": 0},
        "full": {"correct": 0, "incorrect": 0},
    }

    for q in questions_all:
        qtype = q["question"].get("type")

        if q.get("correct") is True:
            if qtype in stats_dict:
                stats_dict[qtype]["correct"] += 1
            stats_dict["full"]["correct"] += 1
        elif q.get("correct") is False:
            if qtype in stats_dict:
                stats_dict[qtype]["incorrect"] += 1
            stats_dict["full"]["incorrect"] += 1

    for k in stats_dict:
        c = stats_dict[k]["correct"]
        ic = stats_dict[k]["incorrect"]
        total = c + ic
        stats_dict[k]["accuracy"] = c / total if total > 0 else None

    with open(os.path.join(EVAL_FOLDER, f"{args.model_name.replace('/', '_')}_params.json"), "w", encoding="utf-8") as f:
        f.write(json.dumps(stats_dict, indent=2, ensure_ascii=False))
