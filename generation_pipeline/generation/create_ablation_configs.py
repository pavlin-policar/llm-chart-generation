import json
from pathlib import Path
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configs_subfolder",
        type=str,
        required=True,
        help="Name of the subfolder folder inside of the generation/configs folder",
    )
    parser.add_argument(
        "--fixed_datasets",
        action="store_true",
        help="Ignores the dataset usability stages of the pipeline while generating combinations for ablation",
    )
    return parser.parse_args()

if __name__ == "__main__":
    main_dir = Path(__file__).resolve().parent.parent.parent

    args = parse_args()

    configs_dir = os.path.join(main_dir, "generation_pipeline", "generation", "configs")
    ablation_configs_dir = os.path.join(configs_dir, args.configs_subfolder)

    os.makedirs(ablation_configs_dir, exist_ok=True)

    default_config = os.path.join(configs_dir, "default_parameters.json")

    with open(default_config, "r", encoding="utf-8") as f:
        default_config = json.load(f)

    with open(os.path.join(ablation_configs_dir, f"config_base.json"), "w+", encoding="utf-8") as f:
        f.write(json.dumps(default_config, indent=4))

    stages = default_config["stages"]

    # Set all to false for complete baseline
    for k in stages.keys():
        if (k == "dataset_usability" and args.fixed_datasets) or k == "feedback":
            continue
        if "parameters" in stages[k].keys():
            stages[k]["parameters"]["reasoning"] = False

    with open(os.path.join(ablation_configs_dir, f"config_no_think.json"), "w+", encoding="utf-8") as f:
        f.write(json.dumps(default_config, indent=4))

    for k in stages.keys():
        if (k == "dataset_usability" and args.fixed_datasets) or k != "feedback":
            continue
         
        if "parameters" in stages[k].keys():
            stages[k]["parameters"]["reasoning"] = True

            with open(os.path.join(ablation_configs_dir, f"config_think_{k}.json"), "w+", encoding="utf-8") as f:
                f.write(json.dumps(default_config, indent=4))

            stages[k]["parameters"]["reasoning"] = False

    for k in stages.keys():
        if k == "dataset_usability" and args.fixed_datasets:
            continue

        if "active" in stages[k].keys():
            stages[k]["active"] = False

            with open(os.path.join(ablation_configs_dir, f"config_no_{k}.json"), "w+", encoding="utf-8") as f:
                f.write(json.dumps(default_config, indent=4))

            stages[k]["active"] = True
    
    # Only during planning task
    stages["plan_code_generation"]["parameters"]["tools"] = True
    with open(os.path.join(ablation_configs_dir, f"config_tools_planning.json"), "w+", encoding="utf-8") as f:
        f.write(json.dumps(default_config, indent=4))

    # Both during coding tasks and planning
    stages["code_generation"]["parameters"]["tools"] = True
    stages["code_regeneration"]["parameters"]["tools"] = True
    with open(os.path.join(ablation_configs_dir, f"config_tools_generation_planning.json"), "w+", encoding="utf-8") as f:
        f.write(json.dumps(default_config, indent=4))

    # Only during coding tasks
    stages["plan_code_generation"]["parameters"]["tools"] = False
    with open(os.path.join(ablation_configs_dir, f"config_tools_generation.json"), "w+", encoding="utf-8") as f:
        f.write(json.dumps(default_config, indent=4))

    
    
    


    