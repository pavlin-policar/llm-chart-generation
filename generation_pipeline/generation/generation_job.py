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
    check_call,
    describe_graph_png,
    determine_dataset_call,
    generate_graph_question_one,
    generate_graph_questions,
    give_question_types,
    graph_call,
    graphs_call,
    plan_call,
    recode_call,
    rejection_call,
    replace_vars_call,
)
from helpers import get_dataset_semantics, get_random_ds, openml_list_uci
from langchain_openai import ChatOpenAI

API_URL = "http://localhost:8888/v1"
MAX_GRAPH_RETRIES = 3
MAX_GRAPH_TYPE_RETRIES = 3


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
        default="parameters.json",
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
    new_names = replace_vars_call(
        llm,
        dataset_sem.get("features"),
        dataset_sem["description"],
    )

    try:
        if len(new_names) != len(old_names):
            raise ValueError("Replacement feature count does not match the dataset")

        df.columns = new_names
        for old, new in zip(old_names, new_names):
            dataset_sem["description"] = dataset_sem["description"].replace(old, new)
            features_json = json.dumps(dataset_sem["features"]).replace(old, new)
            dataset_sem["features"] = json.loads(features_json)
    except Exception as error:
        df.columns = old_names
        print(f"Couldn't replace variable names... {error}")
        return None, old_names

    return old_names, new_names


def get_start_index(start_index, images_folder):
    if start_index != -1:
        return start_index

    return sum(1 for file_name in os.listdir(images_folder) if os.path.isfile(os.path.join(images_folder, file_name)))


def select_dataset(datasets_meta, rng, stages, llm, llm_think):
    while True:
        try:
            print("Fetching random dataset...")
            dataset_id, df = get_random_ds(datasets_meta, rng)
            dataset_sem = get_dataset_semantics(dataset_id, sleep_s=1.0)

            if dataset_sem.get("features") is None:
                dataset_sem["features"] = ""

            if stage_is_active(stages, "dataset_usability"):
                print("Getting usability...")
                usability_llm = select_llm(
                    stages,
                    "dataset_usability",
                    llm,
                    llm_think,
                )
                dataset_description = determine_dataset_call(
                    usability_llm,
                    dataset_sem,
                )
                if not dataset_description["useful"]:
                    print(f"Dataset {dataset_id} deemed not useful, picking another...")
                    continue
            else:
                print("Skipping usability check")
                dataset_description = {
                    "useful": True,
                    "description": dataset_sem.get("description", ""),
                }

            return dataset_id, df, dataset_sem, dataset_description
        except Exception as error:
            print(f"Error fetching dataset, retrying... {error}")


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
            graph_types = graphs_call(
                graph_types_llm,
                json.dumps(head_json),
                dataset_sem["description"],
                num_graphs,
            )

            for graph_type in graph_types:
                graph_type["style"] = rng.choice(plt.style.available)
            return graph_types
        except Exception as error:
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
):
    images = []
    feedback_llm = select_llm(stages, "feedback", llm, llm_think)
    regeneration_active = stage_is_active(stages, "code_regeneration")
    max_iterations = stage_parameter(stages, "code_regeneration", "iterations") if regeneration_active else 0

    graph_data = None
    graph_df = None
    iteration = 0

    while True:
        feedback = check_call(feedback_llm, graph_file_path, code)
        images.append(
            {
                "path": os.path.relpath(graph_file_path, dataset_folder),
                "feedback": feedback["feedback"],
                "code": code,
            }
        )

        if not feedback["correction"] or iteration >= max_iterations:
            break

        iteration += 1
        print(
            f"Graph {graph_number}/{graph_count} needs correction, "
            f"regenerating ({iteration})... "
            f"Time: {(time.perf_counter() - time_start):.04f}"
        )

        graph_file_path = f"{image_prefix}_it{iteration}.png"
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
            feedback["feedback"],
            df=df,
            use_tools=stage_uses_tools(stages, "code_regeneration"),
        )

        plt.close("all")
        graph_data, graph_df = execute_graph_code(
            code,
            df,
            selected_plot,
            graph_file_path,
        )

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
        print(f"Couldn't generate question types... {error}")


def build_metadata(
    dataset_id,
    dataset_description,
    old_names,
    new_names,
    selected_plot,
    description,
    code,
    graph_data,
    questions,
    rejection_response,
    images,
):
    return {
        "id": str(uuid.uuid4()),
        "dataset": {
            "id": dataset_id,
            "description": dataset_description["description"],
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
            "accepted": rejection_response["accept"],
            "reason": (None if rejection_response["accept"] else rejection_response.get("reason", "")),
        },
        "images": images,
    }


def generate_graph(
    graph_index,
    graph_types,
    dataset_id,
    df,
    dataset_sem,
    dataset_description,
    old_names,
    new_names,
    image_index,
    job_id,
    dataset_folder,
    images_folder,
    stages,
    llm,
    llm_think,
):
    time_start = time.perf_counter()
    selected_plot = graph_types[graph_index]
    image_prefix = os.path.join(images_folder, f"{job_id}_{image_index}")
    graph_file_path = f"{image_prefix}_it0.png"

    print(f"Generating graph {graph_index + 1}/{len(graph_types)} for dataset {dataset_id}..., image id: {job_id}_{image_index}")

    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    plt.style.use("default")

    plan = None
    if stage_is_active(stages, "plan_code_generation"):
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
    graph_data, graph_df = execute_graph_code(
        code,
        df,
        selected_plot,
        graph_file_path,
    )

    if stage_is_active(stages, "feedback"):
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
        )
        graph_data = regenerated_data or graph_data
        graph_df = regenerated_df if regenerated_df is not None else graph_df
    else:
        images = [
            {
                "path": os.path.relpath(graph_file_path, dataset_folder),
                "feedback": "",
                "code": code,
            }
        ]

    final_img_path = os.path.join(dataset_folder, images[-1]["path"])
    try:
        rejection_response = rejection_call(llm, final_img_path)
    except Exception as error:
        print(f"Error during rejection step... {error}")
        rejection_response = {"accept": True, "reason": ""}

    description = None
    questions = None
    if rejection_response.get("accept"):
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
            dataset_description["description"],
            selected_plot["description"],
            use_tools=stage_uses_tools(stages, "description"),
        )

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
                    dataset_description["description"],
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
                dataset_description["description"],
                description,
                graph_data,
                num_questions,
                graph_df=graph_df,
                use_tools=stage_uses_tools(stages, "questions"),
            )

        label_questions(questions, stages, llm, llm_think)
    else:
        print(f"Rejected graph {graph_file_path}... Time: {(time.perf_counter() - time_start):.04f}")

    print(f"Finished graph... Time: {(time.perf_counter() - time_start):.04f}")
    return build_metadata(
        dataset_id,
        dataset_description,
        old_names,
        new_names,
        selected_plot,
        description,
        code,
        graph_data,
        questions,
        rejection_response,
        images,
    )


def append_metadata(metadata_path, metadata):
    with open(metadata_path, "a", encoding="utf-8") as file:
        file.write(json.dumps(metadata, ensure_ascii=False) + "\n")


def run_generation(args, job_id, stages, llm, llm_think):
    main_dir = Path(__file__).resolve().parent.parent.parent
    dataset_folder = os.path.join(main_dir, "dataset")
    images_folder = os.path.join(dataset_folder, "images")

    os.makedirs(dataset_folder, exist_ok=True)
    os.makedirs(images_folder, exist_ok=True)

    metadata_path = os.path.join(dataset_folder, args.metadata_file)
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

    print("Start generation...")
    while image_index < target_index:
        (
            dataset_id,
            df,
            dataset_sem,
            dataset_description,
        ) = select_dataset(datasets_meta, rng, stages, llm, llm_think)

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
                        dataset_description,
                        old_names,
                        new_names,
                        image_index,
                        job_id,
                        dataset_folder,
                        images_folder,
                        stages,
                        llm,
                        llm_think,
                    )
                    append_metadata(metadata_path, metadata)
                    image_index += 1
                    break
                except Exception as error:
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
    run_generation(args, job_id, stages, llm, llm_think)


if __name__ == "__main__":
    main()
