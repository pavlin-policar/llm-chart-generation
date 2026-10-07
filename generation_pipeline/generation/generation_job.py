import argparse
import copy
import json
import os
import re
import time
import uuid
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from call_collector import LLMCallCollector
from calls import (
    correct_graph_questions,
    describe_graph_png,
    determine_dataset_usability_call,
    format_dataset_description_call,
    generate_graph_question_one,
    generate_graph_questions,
    give_question_types,
    graph_call,
    graph_error_call,
    graph_evaluation_call,
    graphs_call,
    judge_graph_question_data,
    judge_graph_questions,
    plan_call,
    recode_call,
    replace_vars_call,
    sanitize_dataset_description_call,
)
from helpers import get_dataset_semantics, get_random_ds, openml_list_uci
from vllm_openai import VLLMChatOpenAI

MAX_GRAPH_RETRIES = 3
MAX_GRAPH_TYPE_RETRIES = 3
ERROR_PATH = None
CURRENT_STAGE = None
CURRENT_GRAPH_ID = None


LLM_CALL_COLLECTOR = LLMCallCollector()


def log_error(stage, error):

    if getattr(error, "_already_logged", False):
        return

    if ERROR_PATH is not None:
        with open(ERROR_PATH, "a", encoding="utf-8") as file:
            file.write(
                json.dumps(
                    {
                        "graph_id": CURRENT_GRAPH_ID,
                        "stage": stage,
                        "error": f"{type(error).__name__}: {error}",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    error._already_logged = True


def define_llm_clients(api_url, model_name):
    llm = VLLMChatOpenAI(
        model=model_name,
        openai_api_key="EMPTY",
        openai_api_base=api_url,
        extra_body={
            "chat_template_kwargs": {
                "enable_thinking": False,
            }
        },
        callbacks=[LLM_CALL_COLLECTOR],
    )

    llm_think = VLLMChatOpenAI(
        model=model_name,
        openai_api_key="EMPTY",
        openai_api_base=api_url,
        extra_body={
            "chat_template_kwargs": {
                "enable_thinking": True,
                "reasoning_effort": "xhigh",
            },
        },
        callbacks=[LLM_CALL_COLLECTOR],
    )

    return llm, llm_think


def parse_args(default_seed):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata_file",
        type=str,
        default="",
        help="Output filename inside the dataset folder.",
    )
    parser.add_argument(
        "--parameters_file",
        type=str,
        default="default_parameters.json",
        help="Pipeline parameters file inside generation/configs.",
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="Starting image index. Use -1 to continue after existing images.",
    )
    parser.add_argument(
        "--datasets",
        type=int,
        default=10,
        help="Number of datasets to generate graphs for.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=default_seed,
        help="Seed for random dataset selection.",
    )
    parser.add_argument(
        "--run_id",
        type=int,
        required=True,
        help="Run ID used to avoid filename collisions in parallel jobs.",
    )
    parser.add_argument(
        "--fixed_datasets",
        action="store_true",
        help="Use datasets from the configs/good_datasets.jsonl file",
    )
    parser.add_argument(
        "--rating_threshold",
        type=int,
        default=3,
        help="Minimum rating for which the graph is accepted.",
    )
    parser.add_argument(
        "--api_url",
        type=str,
        default="http://0.0.0.0:8888/v1",
        help="URL of the vLLM OpenAI API endpoint.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="qwen3.8",
        help="Name of the model hosted on the vLLM server",
    )
    return parser.parse_args()


def load_pipeline(file_name):
    with open(file_name, "r", encoding="utf-8") as file:
        return json.load(file)


def stage_is_active(stages, stage_name):
    return stages[stage_name].get("active", True)


def stage_parameter(stages, stage_name, parameter_name):
    return stages[stage_name].get("parameters", {})[parameter_name]


def stage_uses_tools(stages, stage_name):
    return stages[stage_name].get("parameters", {}).get("tools", False)


def select_llm(stages, stage_name, llm, llm_think):
    if stage_parameter(stages, stage_name, "reasoning"):
        return llm_think
    return llm


def replace_variables(llm, dataset_sem, df):
    print("Replacing variables...")

    old_names = list(df.columns)
    original_sem = copy.deepcopy(dataset_sem)

    try:
        new_names = replace_vars_call(
            llm,
            dataset_sem.get("features"),
            dataset_sem["description"],
            call_metadata={"stage_name": "variable_replacement"},
        )

        if len(new_names) != len(old_names):
            raise ValueError("Replacement feature count does not match the dataset")

        if not all(isinstance(name, str) and name.strip() for name in new_names):
            raise ValueError("Replacement feature names must be non-empty strings")

        new_names = [name.strip() for name in new_names]

        if len(set(new_names)) != len(new_names):
            raise ValueError("Replacement feature names must be unique")

        rename_map = dict(zip(old_names, new_names))

        df.columns = new_names
        for feature in dataset_sem.get("features", []):
            old_name = feature.get("name")
            if old_name in rename_map:
                feature["name"] = rename_map[old_name]

        pattern = re.compile(r"(?<!\w)(" + "|".join(re.escape(name) for name in sorted(old_names, key=len, reverse=True)) + r")(?!\w)")
        dataset_sem["description"] = pattern.sub(
            lambda match: rename_map[match.group(0)],
            dataset_sem["description"],
        )

    except Exception as error:
        df.columns = old_names
        dataset_sem.clear()
        dataset_sem.update(original_sem)
        log_error("variable_replacement", error)
        print(f"Couldn't replace variable names... {error}")

        return None, old_names

    return old_names, new_names


def get_start_index(start_index, images_folder):
    if start_index != -1:
        return start_index

    return sum(1 for file_name in os.listdir(images_folder) if file_name.endswith("_it0.png"))


def select_dataset(datasets_meta, rng, stages, llm, llm_think, preselect_id=None):
    while True:
        try:
            print("Fetching random dataset...")

            dataset_id, df = get_random_ds(datasets_meta, rng, preselect_id)

            dataset_sem = get_dataset_semantics(dataset_id, sleep_s=1.0)

            if dataset_sem.get("features") is None:
                dataset_sem["features"] = ""

            usable = True

            if stage_is_active(stages, "dataset_usability") and preselect_id is None:
                print("Getting usability...")
                usability_llm = select_llm(
                    stages,
                    "dataset_usability",
                    llm,
                    llm_think,
                )

                usability = determine_dataset_usability_call(
                    usability_llm,
                    dataset_sem,
                    call_metadata={"stage_name": "dataset_usability"},
                )

                usable = usability["useful"]

                if not usable:
                    print(f"Dataset {dataset_id} deemed not useful, picking another...")
                    continue

            else:
                print("Skipping usability check.")

            if stage_is_active(stages, "format_description"):
                description_llm = select_llm(
                    stages,
                    "format_description",
                    llm,
                    llm_think,
                )

                description = format_dataset_description_call(
                    description_llm,
                    dataset_sem,
                    call_metadata={"stage_name": "format_description"},
                )["description"]

                dataset_sem["description"] = description

            else:
                print("Skipping description formatting.")

            return dataset_id, df, dataset_sem
        except Exception as error:
            log_error("dataset_selection", error)
            print(f"Error fetching dataset... {error}")

            # If we are fixing the dataset go to the next one if it can't be found
            if preselect_id is not None:
                return None, None, None

            print("Retrying with another random dataset...")


def generate_graph_types(
    dataset_id,
    df,
    dataset_sem,
    rng,
    stages,
    llm,
    llm_think,
):
    print(f"Generating graph types for dataset {dataset_id}...")

    for retry in range(1, MAX_GRAPH_TYPE_RETRIES + 1):
        try:
            graph_types_llm = select_llm(
                stages,
                "graph_types_generation",
                llm,
                llm_think,
            )
            num_graphs = stage_parameter(
                stages,
                "graph_types_generation",
                "num_graphs",
            )

            creativity = stage_parameter(stages, "graph_types_generation", "creativity")

            alpha, beta = 2, 4

            if creativity == "random" or type(creativity) not in (int, float):
                alpha = stage_parameter(stages, "graph_types_generation", "alpha")

                beta = stage_parameter(stages, "graph_types_generation", "beta")

                creativity = None

            graph_types = graphs_call(
                graph_types_llm,
                dataset_sem["features"],
                dataset_sem["description"],
                num_graphs,
                creativity=creativity,
                alpha=alpha,
                beta=beta,
                call_metadata={"stage_name": "graph_types_generation"},
                df=df,
                use_tools=stage_uses_tools(stages, "graph_types_generation"),
                final_llm=llm,
            )

            for graph_type in graph_types:
                graph_type["style"] = rng.choice(plt.style.available)
            return graph_types
        except Exception as error:
            log_error("graph_types_generation", error)
            print(f"Error generating graph types, retrying ({retry}) with dataset {dataset_id}... {error}")

    return None


def execute_graph_code(code, df, selected_plot, graph_file_path):

    if os.path.exists(graph_file_path):
        os.remove(graph_file_path)

    exec_namespace = {
        "df": df,
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

    return graph_data, graph_df


def review_and_regenerate(
    code,
    df,
    selected_plot,
    dataset_sem,
    graph_file_path,
    image_prefix,
    dataset_folder,
    stages,
    llm,
    llm_think,
    graph_number,
    graph_count,
    time_start,
    rating_threshold,
):
    images = []
    feedback_llm = select_llm(stages, "feedback", llm, llm_think)
    feedback_type = stage_parameter(stages, "feedback", "feedback_type")
    if feedback_type not in ("rating", "per_error", "per_error_exhaustive"):
        raise ValueError(f"Unsupported feedback_type: {feedback_type}")
    regeneration_active = stage_is_active(stages, "code_regeneration")

    max_iterations = stage_parameter(stages, "code_regeneration", "iterations") if regeneration_active else 0

    num_code_error_regen = stage_parameter(stages, "code_generation", "error_iterations")

    graph_data = None
    graph_df = None
    iteration = 0
    feedback_text = None

    skip_evaluation = False
    last_valid_code = code
    last_accepted = None

    while True:
        previous_graph_file_path = graph_file_path

        if not skip_evaluation:
            try:
                feedback = graph_evaluation_call(
                    feedback_llm,
                    graph_file_path,
                    code,
                    call_metadata={
                        "stage_name": "feedback",
                        "regeneration_iteration": iteration,
                    },
                    feedback_type=feedback_type,
                )

                if feedback_type == "rating":
                    feedback_text = feedback["feedback"]
                    rating = int(feedback["rating"])
                    accepted = rating >= rating_threshold
                    error_types = feedback["error_type"]
                else:
                    errors = feedback["errors"]
                    feedback_text = "\n".join(
                        f"{error['type']} (severity {error['severity']}): "
                        f"{error['description']} Fix: {error['feedback']}"
                        for error in errors
                    )
                    rating = None
                    accepted = not any(error["severity"] >= 3 for error in errors)
                    error_types = [error["type"] for error in errors]

                image = {
                    "path": os.path.relpath(
                        graph_file_path,
                        dataset_folder,
                    ),
                    "rating": rating,
                    "feedback": feedback_text,
                    "accept": accepted,
                    "error_type": error_types,
                    "code": code,
                }
                if feedback_type in ("per_error", "per_error_exhaustive"):
                    image["errors"] = errors
                images.append(image)
                if feedback_type == "per_error_exhaustive" and accepted:
                    # Keep the accepted image's code and plotting data together.
                    last_accepted = (
                        len(images) - 1,
                        code,
                        copy.deepcopy(graph_data),
                        graph_df.copy(deep=True) if graph_df is not None else None,
                        graph_file_path,
                    )

                feedback_resolved = (
                    not errors
                    if feedback_type == "per_error_exhaustive"
                    else accepted
                )
                if feedback_resolved or iteration >= max_iterations:
                    break

            except Exception as error:
                log_error("graph_evaluation", error)
                feedback_text = f"Graph evaluation failed with {type(error).__name__}: {error}"

                images.append(
                    {
                        "path": os.path.relpath(
                            graph_file_path,
                            dataset_folder,
                        ),
                        "rating": None,
                        "feedback": feedback_text,
                        "accept": False,
                        "error_type": ["generation_error"],
                        "code": code,
                    }
                )

                if iteration >= max_iterations:
                    break
        else:
            skip_evaluation = False

        iteration += 1

        print(
            f"Graph {graph_number}/{graph_count} needs correction, "
            f"regenerating ({iteration})... "
            f"Time: {(time.perf_counter() - time_start):.04f}"
        )

        graph_file_path = f"{image_prefix}_it{iteration}.png"

        try:
            recode_llm = select_llm(
                stages,
                "code_regeneration",
                llm,
                llm_think,
            )

            code = recode_call(
                recode_llm,
                dataset_sem.get("features"),
                selected_plot,
                code,
                feedback_text,
                df=df,
                use_tools=stage_uses_tools(
                    stages,
                    "code_regeneration",
                ),
                call_metadata={
                    "stage_name": "code_regeneration",
                    "regeneration_iteration": iteration,
                },
            )

            plt.close("all")

            iterations_error = 0
            exec_error = None

            while True:
                try:
                    graph_data, graph_df = execute_graph_code(
                        code,
                        df,
                        selected_plot,
                        graph_file_path,
                    )

                    exec_error = None
                    last_valid_code = code

                    break

                except Exception as e:
                    exec_error = e
                    log_error("code_regeneration", exec_error)

                    if iterations_error >= num_code_error_regen:
                        break

                    print(
                        f"Error executing regenerated code (error iteration {iterations_error}/{num_code_error_regen}): ", str(exec_error)
                    )

                    code = graph_error_call(
                        recode_llm,
                        dataset_sem.get("features"),
                        selected_plot,
                        json.loads(
                            df.head(5).to_json(orient="records", date_format="iso")
                        ),
                        previous_code=code,
                        execution_error=str(exec_error),
                        df=df,
                        use_tools=stage_uses_tools(stages, "code_regeneration"),
                        call_metadata={
                            "stage_name": "code_regeneration",
                            "regeneration_iteration": iteration,
                            "error_iteration": iterations_error,
                        },
                    )

                iterations_error += 1

            if exec_error is not None:
                raise exec_error

        except Exception as error:
            plt.close("all")
            log_error("code_regeneration", error)

            feedback_text = f"The regenerated graph code failed during execution. Fix the following error:\n{type(error).__name__}: {error}"

            print(f"Graph {graph_number}/{graph_count} regeneration {iteration} failed: {error}")

            # No new image was produced, so record this iteration using the
            # last successfully created image and its matching code.
            graph_file_path = previous_graph_file_path
            images.append(
                {
                    "path": os.path.relpath(
                        graph_file_path,
                        dataset_folder,
                    ),
                    "rating": None,
                    "feedback": feedback_text,
                    "accept": False,
                    "error_type": ["generation_error"],
                    "code": last_valid_code,
                }
            )

            if iteration >= max_iterations:
                code = last_valid_code
                break

            skip_evaluation = True
            continue

    selected_index = len(images) - 1
    if last_accepted is not None:
        selected_index, code, graph_data, graph_df, graph_file_path = last_accepted
    # Preserve attempt order while identifying the final graph for consumers.
    images[selected_index]["selected"] = True
    return code, graph_data, graph_df, graph_file_path, images


def label_questions(questions, stages, llm, llm_think):
    if not stage_is_active(stages, "question_labeling"):
        return

    print("Labeling questions")
    labeling_llm = llm
    parameters = stages["question_labeling"].get("parameters", {})
    if parameters.get("reasoning", False):
        labeling_llm = llm_think

    try:
        labels = []
        rerun_labels = 0
        while len(labels) != len(questions):
            labels = give_question_types(
                labeling_llm,
                questions,
                call_metadata={"stage_name": "question_labeling"},
            )
            rerun_labels += 1
            if rerun_labels > 30:
                for question in questions:
                    question["type"] = None
                return

        for question, label in zip(questions, labels):
            question["type"] = label
    except Exception as error:
        log_error("question_labeling", error)
        print(f"Couldn't generate question types... {error}")


def build_metadata(
    dataset_id,
    dataset_sem,
    old_names,
    new_names,
    selected_plot,
    description,
    code,
    graph_data,
    questions,
    images,
    image_id,
    llm_calls,
    graph_id=None,
):
    final_image = next((image for image in images if image.get("selected")), images[-1])
    return {
        "id": graph_id or str(uuid.uuid4()),
        "prefix_id": image_id,
        "accepted": final_image["accept"],
        "dataset": {
            "id": dataset_id,
            "description": dataset_sem["description"],
            "sanitized_description": dataset_sem["sanitized_description"],
            "old_feature_names": old_names,
            "feature_names": new_names,
        },
        "graph": {
            "type": selected_plot["type"],
            "style": selected_plot["style"],
            "full_description": description,
            "short_description": selected_plot["description"],
            "code": code,
            "structured_data": graph_data,
            "questions": questions,
        },
        "images": images,
        "llm_calls": llm_calls,
    }


def verify_batched_questions(
    questions, llm, png_path, dataset_sem, graph_data, graph_df,
    plot_code, stages, final_llm, *, correction_round=0,
):
    """Verify candidates once in groups of five and retain their judgments."""
    global CURRENT_STAGE

    review_phase = "Initial question review" if correction_round == 0 else "Corrected question review"
    batch_count = (len(questions) + 4) // 5
    for start in range(0, len(questions), 5):
        batch = questions[start:start + 5]
        batch_label = f"{review_phase}, batch {start // 5 + 1}/{batch_count} ({len(batch)} questions)"
        metadata = {"question_batch_start": start, "question_correction_round": correction_round}
        if stages["questions"].get("parameters", {}).get("grounding_judge", True):
            CURRENT_STAGE = "question_judge"
            validator_start = time.perf_counter()
            print(f"{batch_label}: running visual validator...", flush=True)
            judgments = judge_graph_questions(
                llm, png_path, dataset_sem["sanitized_description"], batch,
                call_metadata={"stage_name": CURRENT_STAGE, **metadata},
                final_llm=final_llm, return_reasons=True,
            )
            for question, judgment in zip(batch, judgments):
                question["vlisual_valid"] = judgment["vlisual_valid"]
                question["visual_reason"] = judgment["reason"]
            rejected_count = sum(not judgment["vlisual_valid"] for judgment in judgments)
            print(
                f"{batch_label}: visual validator finished. "
                f"Elapsed: {time.perf_counter() - validator_start:.04f}s; "
                f"rejected {rejected_count}/{len(batch)}.",
                flush=True,
            )

        CURRENT_STAGE = "question_data_judge"
        validator_start = time.perf_counter()
        print(f"{batch_label}: running data validator...", flush=True)
        judgments = judge_graph_question_data(
            llm, batch, graph_df, graph_data, plot_code,
            call_metadata={"stage_name": CURRENT_STAGE, **metadata},
            final_llm=final_llm, return_reasons=True,
        )
        for question, judgment in zip(batch, judgments):
            question["data_valid"] = judgment["data_valid"]
            question["data_reason"] = judgment["reason"]
        rejected_count = sum(not judgment["data_valid"] for judgment in judgments)
        print(
            f"{batch_label}: data validator finished. "
            f"Elapsed: {time.perf_counter() - validator_start:.04f}s; "
            f"rejected {rejected_count}/{len(batch)}.",
            flush=True,
        )


def verify_and_correct_batched_questions(
    questions, llm, png_path, dataset_sem, description, graph_data, graph_df,
    plot_code, stages, final_llm,
):
    """Validate, correct all failures in one batch, then validate corrections once."""
    global CURRENT_STAGE

    review_start = time.perf_counter()
    print(f"Reviewing {len(questions)} initial questions...", flush=True)
    verify_batched_questions(
        questions, llm, png_path, dataset_sem, graph_data, graph_df,
        plot_code, stages, final_llm,
    )
    rejected = [(index, question) for index, question in enumerate(questions)
                if question.get("vlisual_valid") is False or question.get("data_valid") is False]
    print(
        f"Initial question review finished. Elapsed: {time.perf_counter() - review_start:.04f}s; "
        f"rejected {len(rejected)}/{len(questions)}.",
        flush=True,
    )
    if not rejected:
        print("Skipping question correction: all initial questions passed.", flush=True)
        return

    retained = [question for question in questions
                if question.get("vlisual_valid") is not False and question.get("data_valid") is not False]
    CURRENT_STAGE = "question_correction"
    correction_start = time.perf_counter()
    print(f"Correcting {len(rejected)} rejected questions together (round 1/1)...", flush=True)
    try:
        corrected = correct_graph_questions(
            llm, png_path, dataset_sem["description"], description, graph_data,
            plot_code, [question for _, question in rejected], retained,
            graph_df=graph_df, use_tools=stage_uses_tools(stages, "questions"),
            call_metadata={"stage_name": CURRENT_STAGE, "question_correction_round": 1},
            final_llm=final_llm,
            sanitized_dataset_desc=dataset_sem["sanitized_description"],
        )
    except Exception as error:
        log_error(CURRENT_STAGE, error)
        print(
            f"Question correction failed after {time.perf_counter() - correction_start:.04f}s; "
            f"keeping original candidates: {error}",
            flush=True,
        )
        return
    print(
        f"Question correction finished ({len(corrected)} questions). "
        f"Elapsed: {time.perf_counter() - correction_start:.04f}s.",
        flush=True,
    )

    for (index, original), correction in zip(rejected, corrected):
        # Keep originals and link each correction to its one-based saved index.
        correction["replaces_question"] = index + 1
        if "difficulty" in original:
            correction["difficulty"] = original["difficulty"]
    questions.extend(corrected)
    review_start = time.perf_counter()
    print(f"Reviewing {len(corrected)} corrected questions...", flush=True)
    verify_batched_questions(
        corrected, llm, png_path, dataset_sem, graph_data, graph_df,
        plot_code, stages, final_llm, correction_round=1,
    )
    rejected_count = sum(
        question.get("vlisual_valid") is False or question.get("data_valid") is False
        for question in corrected
    )
    print(
        f"Corrected question review finished. Elapsed: {time.perf_counter() - review_start:.04f}s; "
        f"rejected {rejected_count}/{len(corrected)}. Stopping after one correction round.",
        flush=True,
    )


def generate_graph(
    graph_index,
    graph_types,
    dataset_id,
    df,
    dataset_sem,
    old_names,
    new_names,
    image_index,
    job_id,
    dataset_folder,
    images_folder,
    stages,
    llm,
    llm_think,
    rating_threshold,
    initial_llm_calls,
    graph_id=None,
):
    global CURRENT_STAGE, CURRENT_GRAPH_ID

    CURRENT_GRAPH_ID = graph_id or str(uuid.uuid4())

    LLM_CALL_COLLECTOR.start(initial_llm_calls)
    time_start = time.perf_counter()
    selected_plot = graph_types[graph_index]
    image_id = f"{job_id}_{image_index}"
    image_prefix = os.path.join(images_folder, image_id)
    graph_file_path = f"{image_prefix}_it0.png"

    print(f"Generating graph {graph_index + 1}/{len(graph_types)} for dataset {dataset_id}..., image id: {job_id}_{image_index}")

    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    plt.style.use("default")

    plan = None
    if stage_is_active(stages, "plan_code_generation"):
        CURRENT_STAGE = "plan_code_generation"
        plan_llm = select_llm(
            stages,
            "plan_code_generation",
            llm,
            llm_think,
        )
        plan = plan_call(
            plan_llm,
            dataset_sem.get("features"),
            selected_plot,
            df=df,
            use_tools=stage_uses_tools(stages, "plan_code_generation"),
            call_metadata={"stage_name": "plan_code_generation"},
            final_llm=llm,
        )

    CURRENT_STAGE = "code_generation"
    code_llm = select_llm(stages, "code_generation", llm, llm_think)

    num_code_error_regen = stage_parameter(stages, "code_generation", "error_iterations")
    iterations_error = 0
    exec_error = None

    code = graph_call(
        code_llm,
        dataset_sem.get("features"),
        selected_plot,
        json.loads(
            df.head(5).to_json(orient="records", date_format="iso")
        ),
        plan,
        df=df,
        use_tools=stage_uses_tools(stages, "code_generation"),
        call_metadata={"stage_name": "code_generation"},
    )

    plt.style.use(selected_plot["style"])

    while True:
        try:
            graph_data, graph_df = execute_graph_code(
                code,
                df,
                selected_plot,
                graph_file_path,
            )
            exec_error = None

            break

        except Exception as e:
            exec_error = e

            log_error("code_generation", exec_error)

            if iterations_error >= num_code_error_regen:
                break

            print(f"Error executing generated code (error iteration {iterations_error}/{num_code_error_regen}):", str(exec_error))

            code = graph_error_call(
                code_llm,
                dataset_sem.get("features"),
                selected_plot,
                json.loads(
                    df.head(5).to_json(orient="records", date_format="iso")
                ),
                previous_code=code,
                execution_error=str(exec_error),
                df=df,
                use_tools=stage_uses_tools(stages, "code_generation"),
                call_metadata={
                    "stage_name": "code_generation",
                    "error_iteration": iterations_error,
                },
            )

        iterations_error += 1

    if exec_error is not None:
        CURRENT_STAGE = "code_generation"

        raise exec_error

    CURRENT_STAGE = "feedback"
    (
        code,
        regenerated_data,
        regenerated_df,
        graph_file_path,
        images,
    ) = review_and_regenerate(
        code,
        df,
        selected_plot,
        dataset_sem,
        graph_file_path,
        image_prefix,
        dataset_folder,
        stages,
        llm,
        llm_think,
        graph_index + 1,
        len(graph_types),
        time_start,
        rating_threshold,
    )

    graph_data = regenerated_data if regenerated_data is not None else graph_data
    graph_df = regenerated_df if regenerated_df is not None else graph_df

    final_image = next((image for image in images if image.get("selected")), images[-1])
    final_img_path = os.path.join(dataset_folder, final_image["path"])

    description = None
    questions = None
    if final_image["accept"]:
        CURRENT_STAGE = "description"
        print(f"Generating description... Time: {(time.perf_counter() - time_start):.04f}")
        description_llm = select_llm(
            stages,
            "description",
            llm,
            llm_think,
        )
        description = describe_graph_png(
            description_llm,
            final_img_path,
            code,
            graph_data,
            graph_df,
            dataset_sem["description"],
            selected_plot["description"],
            use_tools=stage_uses_tools(stages, "description"),
            call_metadata={"stage_name": "description"},
            final_llm=llm,
        )

        CURRENT_STAGE = "questions"
        print(f"Generating questions... Time: {(time.perf_counter() - time_start):.04f}", flush=True)
        questions_llm = select_llm(stages, "questions", llm, llm_think)
        num_questions = stage_parameter(stages, "questions", "num_questions")
        one_by_one = stages["questions"].get("parameters", {}).get("one", False)

        if one_by_one:
            questions = []
            valid_questions = []
            easy = round(num_questions * 0.35)
            medium = round(num_questions * 0.30)
            difficulties = (
                ["easy"] * easy
                + ["medium"] * medium
                + ["hard"] * (num_questions - easy - medium)
            )
            while len(valid_questions) < num_questions:
                CURRENT_STAGE = "questions"
                difficulty = difficulties[len(valid_questions)]
                quest = generate_graph_question_one(
                    questions_llm,
                    final_img_path,
                    dataset_sem["description"],
                    description,
                    graph_data,
                    valid_questions,
                    sanitized_dataset_desc=dataset_sem["sanitized_description"],
                    difficulty=difficulty,
                    graph_df=graph_df,
                    use_tools=stage_uses_tools(stages, "questions"),
                    call_metadata={"stage_name": "questions"},
                    final_llm=llm,
                )
                quest["difficulty"] = difficulty

                CURRENT_STAGE = "question_judge"
                quest["vlisual_valid"] = judge_graph_questions(
                    questions_llm,
                    final_img_path,
                    dataset_sem["sanitized_description"],
                    [quest],
                    call_metadata={"stage_name": "question_judge"},
                    final_llm=llm,
                )[0]

                CURRENT_STAGE = "question_data_judge"
                quest["data_valid"] = judge_graph_question_data(
                    questions_llm,
                    [quest],
                    graph_df,
                    graph_data,
                    code,
                    call_metadata={"stage_name": "question_data_judge"},
                    final_llm=llm,
                )[0]

                # Keep every candidate for review, but only count passing pairs.
                questions.append(quest)
                if quest["vlisual_valid"] and quest["data_valid"]:
                    valid_questions.append(quest)

        else:
            qa_generation_start = time.perf_counter()
            questions = generate_graph_questions(
                questions_llm,
                final_img_path,
                dataset_sem["description"],
                description,
                graph_data,
                num_questions,
                sanitized_dataset_desc=dataset_sem["sanitized_description"],
                graph_df=graph_df,
                use_tools=stage_uses_tools(stages, "questions"),
                call_metadata={"stage_name": "questions"},
                final_llm=llm,
            )
            print(
                f"Initial QA generation finished ({len(questions)} questions). "
                f"Elapsed: {time.perf_counter() - qa_generation_start:.04f}s.",
                flush=True,
            )

        if not one_by_one:
            verify_and_correct_batched_questions(
                questions, questions_llm, final_img_path, dataset_sem, description,
                graph_data, graph_df, code, stages, llm,
            )

        CURRENT_STAGE = "question_labeling"
        label_questions(questions, stages, llm, llm_think)
    else:
        print(f"Rejected graph {graph_file_path}... Time: {(time.perf_counter() - time_start):.04f}")

    print(f"Finished graph... Time: {(time.perf_counter() - time_start):.04f}")
    CURRENT_STAGE = "metadata"
    llm_calls = LLM_CALL_COLLECTOR.stop()
    return build_metadata(
        dataset_id,
        dataset_sem,
        old_names,
        new_names,
        selected_plot,
        description,
        code,
        graph_data,
        questions,
        images,
        image_id,
        llm_calls,
        graph_id=CURRENT_GRAPH_ID,
    )


def append_metadata(metadata_path, metadata):
    with open(metadata_path, "a", encoding="utf-8") as file:
        file.write(json.dumps(metadata, ensure_ascii=False) + "\n")


def run_generation(args, job_id, stages, llm, llm_think, dataset_ids):
    global ERROR_PATH, CURRENT_GRAPH_ID

    CURRENT_GRAPH_ID = None

    main_dir = Path(__file__).resolve().parent.parent.parent
    dataset_folder = os.path.join(main_dir, "dataset")
    images_folder = os.path.join(dataset_folder, "images")

    config_name = Path(args.parameters_file).stem
    dataset_folder = os.path.join(main_dir, "dataset", config_name)
    images_folder = os.path.join(dataset_folder, "images")

    os.makedirs(dataset_folder, exist_ok=True)
    os.makedirs(images_folder, exist_ok=True)

    metadata_path = os.path.join(dataset_folder, args.metadata_file)
    error_file = "error_" + args.metadata_file
    ERROR_PATH = os.path.join(dataset_folder, error_file)
    start_index = get_start_index(args.start_index, images_folder)
    image_index = start_index
    graphs_per_dataset = stage_parameter(
        stages,
        "graph_types_generation",
        "num_graphs",
    )
    target_index = start_index + args.datasets * graphs_per_dataset
    rng = np.random.default_rng(args.seed)
    datasets_meta = openml_list_uci()

    datasets_iter = iter(dataset_ids or [])

    print("Start generation...")
    while image_index < target_index:
        LLM_CALL_COLLECTOR.start()
        ds_id = None
        if args.fixed_datasets:
            try:
                ds_id = next(datasets_iter)
            except StopIteration:
                print("No more fixed datasets available.")
                break

        (dataset_id, df, dataset_sem) = select_dataset(datasets_meta, rng, stages, llm, llm_think, ds_id)

        if dataset_id is None and args.fixed_datasets:
            print(f"Skipping fixed dataset {ds_id}.")
            continue

        old_names = None
        new_names = list(df.columns)
        if stage_is_active(stages, "variable_replacement"):
            replacement_llm = select_llm(
                stages,
                "variable_replacement",
                llm,
                llm_think,
            )

            old_names, new_names = replace_variables(
                replacement_llm,
                dataset_sem,
                df,
            )

        print("Sanitizing dataset description...")
        try:
            dataset_sem["sanitized_description"] = sanitize_dataset_description_call(
                llm,
                dataset_sem["description"],
                call_metadata={"stage_name": "sanitize_description"},
            )["description"]
        except Exception as error:
            log_error("sanitize_description", error)
            print(f"Couldn't sanitize description for dataset {dataset_id}, skipping... {error}")
            continue

        graph_types = generate_graph_types(
            dataset_id,
            df,
            dataset_sem,
            rng,
            stages,
            llm,
            llm_think,
        )
        initial_llm_calls = LLM_CALL_COLLECTOR.stop()
        if graph_types is None:
            continue

        for graph_index in range(len(graph_types)):
            if image_index >= target_index:
                break

            graph_id = str(uuid.uuid4())
            for retry in range(1, MAX_GRAPH_RETRIES + 1):
                try:
                    metadata = generate_graph(
                        graph_index,
                        graph_types,
                        dataset_id,
                        df,
                        dataset_sem,
                        old_names,
                        new_names,
                        image_index,
                        job_id,
                        dataset_folder,
                        images_folder,
                        stages,
                        llm,
                        llm_think,
                        args.rating_threshold,
                        initial_llm_calls,
                        graph_id=graph_id,
                    )
                    append_metadata(metadata_path, metadata)
                    image_index += 1
                    break

                except Exception as error:
                    log_error(CURRENT_STAGE or "graph_generation", error)
                    print(f"Error generating graph, retrying ({retry}/{MAX_GRAPH_RETRIES})... {error}")

            CURRENT_GRAPH_ID = None


def main():
    slurm_job_id = int(os.getenv("SLURM_JOB_ID", 1))
    pid = os.getpid()
    args = parse_args(default_seed=(slurm_job_id * pid) % 60000)
    job_id = f"{slurm_job_id}_{args.run_id}"

    print(f"JOB ID: {job_id}, PID: {pid}, RUN ID: {args.run_id}")
    if not args.metadata_file:
        args.metadata_file = f"metadata{job_id}.jsonl"

    main_dir = Path(__file__).resolve().parent.parent.parent
    parameters_path = os.path.join(
        main_dir,
        "generation_pipeline",
        "generation",
        "configs",
        args.parameters_file,
    )
    stages = load_pipeline(parameters_path)["stages"]
    llm, llm_think = define_llm_clients(args.api_url, args.model_name)

    datasets_good_file = os.path.join(main_dir, "generation_pipeline", "generation", "configs", "good_datasets.jsonl")
    dataset_ids = None
    if args.fixed_datasets:
        dataset_ids = []
        with open(datasets_good_file, "r", encoding="utf-8") as f:
            for line in f:
                dataset_ids.append(json.loads(line)["id"])

        start = args.run_id * args.datasets
        end = start + args.datasets

        dataset_ids = dataset_ids[start:end]

    run_generation(args, job_id, stages, llm, llm_think, dataset_ids)


if __name__ == "__main__":
    main()
