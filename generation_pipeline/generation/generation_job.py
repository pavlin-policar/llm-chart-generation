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
    describe_graph_png,
    determine_dataset_usability_call,
    format_dataset_description_call,
    generate_graph_question_one,
    generate_graph_questions,
    give_question_types,
    graph_call,
    graphs_call,
    plan_call,
    recode_call,
    replace_vars_call,
    graph_evaluation_call,
)
from helpers import get_dataset_semantics, get_random_ds, openml_list_uci
from langchain_openai import ChatOpenAI

API_URL = "http://0.0.0.0:8888/v1"
MAX_GRAPH_RETRIES = 3
MAX_GRAPH_TYPE_RETRIES = 3
ERROR_PATH = None
CURRENT_STAGE = None


def log_error(stage, error):
    if ERROR_PATH is not None:
        with open(ERROR_PATH, "a", encoding="utf-8") as file:
            file.write(json.dumps({
                "stage": stage,
                "error": f"{type(error).__name__}: {error}",
            }, ensure_ascii=False) + "\n")


def define_llm_clients():
    llm = ChatOpenAI(
        model="qwen3.5",
        openai_api_key="EMPTY",
        openai_api_base=API_URL,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )

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
    try:
        new_names = replace_vars_call(
            llm,
            dataset_sem.get("features"),
            dataset_sem["description"],
        )

        if len(new_names) != len(old_names):
            raise ValueError("Replacement feature count does not match the dataset")

        df.columns = new_names
        for old, new in zip(old_names, new_names):
            dataset_sem["description"] = dataset_sem["description"].replace(old, new)
            features_json = json.dumps(dataset_sem["features"]).replace(old, new)
            dataset_sem["features"] = json.loads(features_json)
    except Exception as error:
        df.columns = old_names
        log_error("variable_replacement", error)
        print(f"Couldn't replace variable names... {error}")
        return None, old_names

    return old_names, new_names


def get_start_index(start_index, images_folder):
    if start_index != -1:
        return start_index

    return sum(
        1
        for file_name in os.listdir(images_folder)
        if file_name.endswith("_it0.png")
    )


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
    head_json = df.head(5).to_dict(orient="records")

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

            creativity = stage_parameter(
                stages,
                "graph_types_generation",
                "creativity"
            )

            alpha, beta = 2, 4

            if creativity == "random" or type(creativity) not in (int, float):
                alpha = stage_parameter(
                    stages,
                    "graph_types_generation",
                    "alpha"
                )

                beta = stage_parameter(
                    stages,
                    "graph_types_generation",
                    "beta"
                )

                creativity = None
                                
            graph_types = graphs_call(
                graph_types_llm,
                json.dumps(head_json),
                dataset_sem["description"],
                num_graphs,
                creativity=creativity,
                alpha=alpha,
                beta=beta
            )

            for graph_type in graph_types:
                graph_type["style"] = rng.choice(plt.style.available)
            return graph_types
        except Exception as error:
            log_error("graph_types_generation", error)
            print(f"Error generating graph types, retrying ({retry}) with dataset {dataset_id}... {error}")

    return None


def execute_graph_code(code, df, selected_plot, graph_file_path):
    exec_namespace = {
        "df": df,
        "selected_plot": selected_plot,
        "graph_file_path": graph_file_path,
        "__builtins__": __builtins__,
    }
    exec(code, exec_namespace, exec_namespace)

    if not os.path.exists(graph_file_path):
        raise ValueError("Generated code did not save image")

    return (
        exec_namespace.get("graph_data"),
        exec_namespace.get("graph_df"),
    )


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
    rating_threshold
):
    images = []
    feedback_llm = select_llm(stages, "feedback", llm, llm_think)
    regeneration_active = stage_is_active(stages, "code_regeneration")
    max_iterations = (
        stage_parameter(stages, "code_regeneration", "iterations")
        if regeneration_active
        else 0
    )

    graph_data = None
    graph_df = None
    iteration = 0
    feedback_text = None

    skip_evaluation = False
    last_valid_code = code


    while True:
        previous_graph_file_path = graph_file_path
        
        if not skip_evaluation:
            try:
                feedback = graph_evaluation_call(
                    feedback_llm,
                    graph_file_path,
                    code,
                )

                feedback_text = feedback["feedback"]

                rating = int(feedback["rating"])
                accepted = rating >= rating_threshold

                images.append(
                    {
                        "path": os.path.relpath(
                            graph_file_path,
                            dataset_folder,
                        ),
                        "rating": rating,
                        "feedback": feedback_text,
                        "accept": accepted,
                        "error_type": feedback["error_type"],
                        "code": code,
                    }
                )

                if accepted or iteration >= max_iterations:
                    break

            except Exception as error:
                log_error("graph_evaluation", error)
                feedback_text = (
                    f"Graph evaluation failed with "
                    f"{type(error).__name__}: {error}"
                )

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
            )

            plt.close("all")

            graph_data, graph_df = execute_graph_code(
                code,
                df,
                selected_plot,
                graph_file_path,
            )

            last_valid_code = code

        except Exception as error:
            plt.close("all")
            log_error("code_regeneration", error)

            feedback_text = (
                "The regenerated graph code failed during execution. "
                "Fix the following error:\n"
                f"{type(error).__name__}: {error}"
            )

            print(
                f"Graph {graph_number}/{graph_count} regeneration "
                f"{iteration} failed: {error}"
            )

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
                # Do not return failed code paired with a valid older image.
                code = last_valid_code
                break

            # The failed regeneration attempt has already been recorded.
            # Skip evaluating the unchanged previous image again.
            skip_evaluation = True
            continue

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
            labels = give_question_types(labeling_llm, questions)
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
):
    return {
        "id": str(uuid.uuid4()),
        "prefix_id": image_id,
        "accepted": images[-1]["accept"],
        "dataset": {
            "id": dataset_id,
            "description": dataset_sem["description"],
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
    }


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
    rating_threshold
):
    global CURRENT_STAGE

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
        )

    CURRENT_STAGE = "code_generation"
    code_llm = select_llm(stages, "code_generation", llm, llm_think)
    code = graph_call(
        code_llm,
        dataset_sem.get("features"),
        selected_plot,
        df.head(5).to_dict(orient="records"),
        plan,
        df=df,
        use_tools=stage_uses_tools(stages, "code_generation"),
    )

    plt.style.use(selected_plot["style"])
    CURRENT_STAGE = "code_execution"
    graph_data, graph_df = execute_graph_code(
        code,
        df,
        selected_plot,
        graph_file_path,
    )

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

    final_img_path = os.path.join(dataset_folder, images[-1]["path"])

    description = None
    questions = None
    if images[-1]["accept"]:
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
        )

        CURRENT_STAGE = "questions"
        print(f"Generating questions... Time: {(time.perf_counter() - time_start):.04f}")
        questions_llm = select_llm(stages, "questions", llm, llm_think)
        num_questions = stage_parameter(stages, "questions", "num_questions")
        one_by_one = stages["questions"].get("parameters", {}).get("one", False)

        if one_by_one:
            questions = []
            for _ in range(num_questions):
                quest = generate_graph_question_one(
                    questions_llm,
                    final_img_path,
                    dataset_sem["description"],
                    description,
                    graph_data,
                    questions,
                    graph_df=graph_df,
                    use_tools=stage_uses_tools(stages, "questions"),
                )

                questions.append(quest)

        else:
            questions = generate_graph_questions(
                questions_llm,
                final_img_path,
                dataset_sem["description"],
                description,
                graph_data,
                num_questions,
                graph_df=graph_df,
                use_tools=stage_uses_tools(stages, "questions"),
            )

        CURRENT_STAGE = "question_labeling"
        label_questions(questions, stages, llm, llm_think)
    else:
        print(f"Rejected graph {graph_file_path}... Time: {(time.perf_counter() - time_start):.04f}")

    print(f"Finished graph... Time: {(time.perf_counter() - time_start):.04f}")
    CURRENT_STAGE = "metadata"
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
        image_id
    )


def append_metadata(metadata_path, metadata):
    with open(metadata_path, "a", encoding="utf-8") as file:
        file.write(json.dumps(metadata, ensure_ascii=False) + "\n")


def run_generation(args, job_id, stages, llm, llm_think, dataset_ids):
    global ERROR_PATH

    main_dir = Path(__file__).resolve().parent.parent.parent
    dataset_folder = os.path.join(main_dir, "dataset")
    images_folder = os.path.join(dataset_folder, "images")

    config_name = Path(args.parameters_file).stem
    dataset_folder = os.path.join(main_dir, "dataset", config_name)
    images_folder = os.path.join(dataset_folder, "images")

    os.makedirs(dataset_folder, exist_ok=True)
    os.makedirs(images_folder, exist_ok=True)

    metadata_path = os.path.join(dataset_folder, args.metadata_file)
    error_file = args.metadata_file.replace("metadata", "error", 1)
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
        ds_id = None
        if args.fixed_datasets:
            try:
                ds_id = next(datasets_iter)
            except StopIteration:
                print("No more fixed datasets available.")
                break

        (
            dataset_id,
            df,
            dataset_sem
        ) = select_dataset(datasets_meta, rng, stages, llm, llm_think, ds_id)

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

        graph_types = generate_graph_types(
            dataset_id,
            df,
            dataset_sem,
            rng,
            stages,
            llm,
            llm_think,
        )
        if graph_types is None:
            continue

        for graph_index in range(len(graph_types)):
            if image_index >= target_index:
                break

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
                    )
                    append_metadata(metadata_path, metadata)
                    image_index += 1
                    break
                except Exception as error:
                    log_error(CURRENT_STAGE or "graph_generation", error)
                    print(f"Error generating graph, retrying ({retry}/{MAX_GRAPH_RETRIES})... {error}")


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
    llm, llm_think = define_llm_clients()

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
