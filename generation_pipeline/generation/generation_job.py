from langchain_openai import ChatOpenAI

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import os
import json
import time
import uuid
from pathlib import Path
import argparse

from calls import (generate_graph_questions, give_question_types, graph_call, 
                   graphs_call, describe_graph_png, check_call, recode_call, 
                   replace_vars_call, determine_dataset_call, rejection_call, 
                   plan_call)

from helpers import (get_random_ds, openml_list_uci, get_dataset_semantics)

OPENML_LIST_URL = "https://www.openml.org/api/v1/json/data"

API_URL = "http://localhost:8888/v1"

def define_llm_clients():
    # Non-reasoning client used for easier tasks
    llm = ChatOpenAI( 
        model="qwen3.5",   
        openai_api_key="EMPTY",  # required but ignored by vLLM
        openai_api_base=API_URL,
        extra_body= {
            "chat_template_kwargs": {"enable_thinking": False}
        }
    )

    # Reasoning model used for harder tasks
    llm_think = ChatOpenAI( 
        model="qwen3.5",  
        openai_api_key="EMPTY",
        openai_api_base=API_URL,
        extra_body= {
            "chat_template_kwargs": {"enable_thinking": True},
            "logit_bias": {
                "248069": 5.0,   # make </think> more likely to discourage reasoning loops
            }
        }
    )

    return llm, llm_think 

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--metadata_file", type=str, required=False, default="") # Name of the output file in the dataset folder. If you're parallelizing should be left blank.
    parser.add_argument("--parameters_file", type=str, default="parameters.json") # Parameters for run
    parser.add_argument("--start_index",  type=int, required=False, default=0) # Starting index of the output image.
    parser.add_argument("--datasets", type=int, required=False, default=10) # For how many datasets to generate the image.
    parser.add_argument("--seed", type=int, required=False, default=((job_id * pid) % 60000)) # The seed for random dataset selection
    parser.add_argument("--run_id", type=int, required=True, default=0) # ID of the run, this is used if youre generating many graphs in parallel, so images don't overwrite themselves.
    args = parser.parse_args()

    return args

def set_pipeline(file_name):
    with open(file_name, "r", encoding="utf-8") as f:
        parameters = json.load(f)

    return parameters

def replace_variables(llm, dataset_sem, df):
    print("Replacing variables...")

    new_names = replace_vars_call(llm, dataset_sem.get("features"), dataset_sem["description"])

    try:
        old_names = list(df.columns)
        df.columns = new_names

        for old, new in zip(old_names, new_names):
            dataset_sem["description"] = dataset_sem["description"].replace(old, new)
            dataset_sem["features"] = json.loads(json.dumps(dataset_sem["features"]).replace(old, new))
        
    except Exception as e:
        # Reset names to default if it fails
        new_names = old_names
        old_names = None

        print(f"Couldn't replace variable names... {e}")

    return old_names, new_names



if __name__ == "__main__": 

    llm, llm_think = define_llm_clients()

    # Get ID of slurm job and the PID to determine the seed, not necessarly unique in every case but good for our case.
    job_id = int(os.getenv("SLURM_JOB_ID", 1)) 
    pid = os.getpid()
    
    args = parse_args()

    job_id = f"{job_id}_{args.run_id}"

    print(f"JOB ID: {job_id}, PID: {pid}, RUN ID: {args.run_id}")

    if args.metadata_file == "":
        args.metadata_file = f"metadata{job_id}.jsonl"

    # MAIN_DIR is set to the git repository folder
    MAIN_DIR = Path(__file__).resolve().parent.parent.parent

    DATASET_FOLDER = os.path.join(MAIN_DIR, "dataset")
    IMAGES_FOLDER = os.path.join(DATASET_FOLDER, "images")
    GENERATE_DS_IMAGES = args.datasets
    FEEDBACK = True

    os.makedirs(DATASET_FOLDER, exist_ok=True)
    os.makedirs(IMAGES_FOLDER, exist_ok=True)

    pipeline_params = set_pipeline(os.path.join(MAIN_DIR, "generation_pipeline", "generation", "configs", args.parameters_file))
    stages = pipeline_params["stages"]

    # If start_index is -1 count files in folder and index from there.
    # Probably better to use UUID for saving but impractical when reviewing the dataset.
    if args.start_index == -1:
        n_files = sum(
            1 for f in os.listdir(IMAGES_FOLDER)
            if os.path.isfile(os.path.join(IMAGES_FOLDER, f))
        )
    else:
        n_files = args.start_index

    index = n_files
    rng = np.random.default_rng(args.seed)

    datasets_meta = openml_list_uci()
   
    print("Start generation...")

    while index < GENERATE_DS_IMAGES*10 + n_files:

        # Randomly select a dataset
        print(f"Fetching random dataset...")
        while True:
            try:
                ds_id, df = get_random_ds(datasets_meta, rng) # Fetch random UCI dataset
                dataset_sem = get_dataset_semantics(ds_id, sleep_s=1.0) # Get dataset semantics

                if dataset_sem.get("features") == None:
                    dataset_sem["features"] = ""

                if stages["dataset_usability"]["active"]:
                    print("Getting usability...")

                    llm_usable = llm_think if stages["dataset_usability"]["parameters"]["reasoning"] else llm
                    desc_dataset = determine_dataset_call(llm_usable, dataset_sem) # Determine if dataset is useful and format description

                    if not desc_dataset["useful"]:
                        print(f"Dataset {ds_id} deemed not useful, picking another...")
                        continue

                else:
                    print("Skipping usability check")

                break

            except Exception as e:
                print(f"Error fetching dataset, retrying... {e}")
                continue


        # Replace the variables with more meaningful names. If it fails, keep previous names.
        if stages["variable_replacement"]["active"]:
            llm_replace = llm_think if stages["dataset_usability"]["parameters"]["reasoning"] else llm
            replace_variables(llm_replace, dataset_sem=dataset_sem, df=df)

        head_json = df.head(5).to_dict(orient="records")

        # Generate graph specifications
        print(f"Generating graph types for dataset {ds_id}...")
        retr = 0
        while retr < 3:
            try:
                llm_types = llm_think if stages["graph_types_generation"]["parameters"]["reasoning"] else llm
                n_graphs = stages["graph_types_generation"]["parameters"]["num_graphs"]
                graph_types = graphs_call(llm_types, json.dumps(head_json), dataset_sem["description"], n_graphs)

                templates = plt.style.available
                for i, t in enumerate(graph_types):
                    graph_types[i]["style"] = rng.choice(templates)

                break
            
            except Exception as e:
                retr+=1
                print(f"Error generating graph types, retrying ({retr}) with dataset {ds_id}... {e}")
                continue

        if retr >= 3:
            continue

        # Generate and render graphs
        i = 0
        rerun = 0

        # Iterate over all the generated specifications
        while i < len(graph_types):
            time_start = time.perf_counter()

            # If errors occur too many times we skip the graph.
            if rerun > 2:
                rerun = 0
                i += 1
                continue

            # Reset matplotlib params just in case.
            matplotlib.rcParams.update(matplotlib.rcParamsDefault)
            plt.style.use("default")

            print(f"Generating graph {i+1}/{len(graph_types)} for dataset {ds_id}..., image id: {job_id}_{index}")
            graph_file_path = os.path.join(IMAGES_FOLDER, f"{job_id}_{index}_it0.png")
            selected_plot = graph_types[i]
            try:
                imgs = []

                plan = None
                if stages["plan_code_generation"]["active"]:
                    plan_llm = llm_think if stages["plan_code_generation"]["parameters"]["reasoning"] else llm
                    plan = plan_call(plan_llm, dataset_sem.get("features"), selected_plot)

                code_llm = llm_think if stages["code_generation"]["parameters"]["reasoning"] else llm
                code = graph_call(code_llm, dataset_sem.get("features"), selected_plot, plan)

                # Execute plotting code.
                plt.style.use(selected_plot["style"])

                exec_ns = {
                    "df": df,
                    "selected_plot": selected_plot,
                    "graph_file_path": graph_file_path,
                    "__builtins__": __builtins__,
                }   
                exec(code, exec_ns, exec_ns)

                # Store the variables created in the plotting code
                graph_data = exec_ns.get("graph_data", None)
                graph_df   = exec_ns.get("graph_df", None)

                if not os.path.exists(graph_file_path):
                    raise ValueError("Generated code did not save image")
                
                # check the graph, give feedback and regenerate if needed
                feedback_llm = llm_think if stages["feedback"]["parameters"]["reasoning"] else llm

                if stages["plan_code_generation"]["active"]:
                    regen_count = 0
                    img_count = 0
                    while True:
                        try:
                            try:
                                feedback = check_call(feedback_llm, graph_file_path, code)
                            except:
                                # Stop checking just in case image doesn't exist. This will lead to error downstream and start the graph from scratch.
                                print("Couldn't find image")
                                break
                            
                            # Append current image
                            imgs.append({ 
                                "path": os.path.relpath(graph_file_path, DATASET_FOLDER), 
                                "feedback": feedback["feedback"], 
                                "code": code
                            })

                            # Regenerate the image
                            if feedback["correction"] and regen_count < stages["plan_code_generation"]["parameters"]["iterations"]:
                                print(f"Graph {i+1} needs correction, regenerating ({regen_count})... Time: {(time.perf_counter() - time_start):.04f}")
                                
                                graph_file_path = os.path.join(IMAGES_FOLDER, f"{job_id}_{index}_it{img_count + 1}.png")

                                recode_llm = llm_think if stages["code_regeneration"]["parameters"]["reasoning"] else llm
                                code_new = recode_call(recode_call, dataset_sem.get("features"), selected_plot, code, feedback["feedback"])

                                # Close previous plots just in case
                                plt.close("all")

                                # Execute code, same as before
                                exec_ns = {
                                    "df": df,
                                    "selected_plot": selected_plot,
                                    "graph_file_path": graph_file_path,
                                    "__builtins__": __builtins__,
                                }

                                exec(code_new, exec_ns, exec_ns)

                                graph_data = exec_ns.get("graph_data", None)
                                graph_df   = exec_ns.get("graph_df", None)

                                if not os.path.exists(graph_file_path):
                                    raise ValueError("Generated code did not save image")
                                
                                code = code_new

                                regen_count += 1
                                img_count += 1

                        except Exception as e:
                            print(f"Error during graph checking, retrying... {e}")
                            graph_file_path = os.path.join(IMAGES_FOLDER, f"{job_id}_{index}_it{img_count}.png") # We set this to the latest available image to use in description generation and questions
                            
                            regen_count += 1
                            continue
                else:
                    # If we don't want regeneration, we just store some feedback and push image to array.
                    feedback = check_call(feedback_llm, graph_file_path, code)

                    imgs.append({ 
                        "path":os.path.relpath(graph_file_path, DATASET_FOLDER), 
                        "feedback": feedback["feedback"],
                        "code": code
                    })

                # Generate graph description ang question and answer pairs.
                graph_data = graph_data  # from executed code   
                graph_df = graph_df  # from executed code
                final_img_path = os.path.join(DATASET_FOLDER, imgs[-1]["path"])

                try:
                    rejection_response = rejection_call(llm, final_img_path)

                except:
                    print(f"Error during rejection step... {e}")

                    rejection_response = {
                        "accept": True,
                        "reason": ""
                    }

                description = None
                questions = None

                if rejection_response.get("accept"):

                    print(f"Generating description... Time: {(time.perf_counter() - time_start):.04f}")
                    desc_llm = llm_think if stages["description"]["parameters"]["reasoning"] else llm
                    description = describe_graph_png(desc_llm, final_img_path, code, graph_data, graph_df, desc_dataset["description"], graph_types[i]["description"])

                    print(f"Generating questions... Time: {(time.perf_counter() - time_start):.04f}")
                    quest_llm = llm_think if stages["questions"]["parameters"]["reasoning"] else llm
                    n_questions = stages["questions"]["parameters"]["questions"]
                    questions = generate_graph_questions(quest_llm, final_img_path, desc_dataset["description"], description, graph_data, n_questions)

                    # Label the generated questions.
                    print(f"Labeling questions")
                    try:
                        rerun_labels = 0
                        labels = []
                        while len(labels) != len(questions):
                            labels = give_question_types(llm, questions)
                            rerun_labels += 1
                            if rerun_labels > 30:
                                for l, q in enumerate(questions):
                                    questions[l]["type"] = None
                                break

                        for l, q in enumerate(questions):
                            questions[l]["type"] = labels[l]

                    except Exception as e:
                        print("Couldnt generate question types", e)
                
                else:
                    print(f"Rejected graph {graph_file_path}... Time: {(time.perf_counter() - time_start):.04f}")

                # Save the graph to the metadata.jsonl file.
                with open(os.path.join(DATASET_FOLDER, args.metadata_file), "a", encoding="utf-8") as f:
                    obj = {
                        "id": str(uuid.uuid4()),
                        "dataset": {
                            "id": ds_id,
                            "description": desc_dataset["description"],
                            "old_feature_names": old_names,
                            "feature_names": new_names
                        },
                        "graph":{
                            "type": graph_types[i]["type"],
                            "style": graph_types[i]["style"],
                            "full_description": description,
                            "short_description": graph_types[i]["description"],
                            "code": code,
                            "structured_data": graph_data,
                            "questions": questions,
                            "accepted": rejection_response["accept"],
                            "reason": rejection_response["reason"] if rejection_response["accept"] else None
                        },
                        "images": imgs
                    }
                    f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    
                index += 1
                i += 1
                rerun = 0

                print(f"Finished graph... Time: {(time.perf_counter() - time_start):.04f}")

            except Exception as e:
                print(f"Error generating graph, retrying... {e}")
                rerun += 1
                continue
