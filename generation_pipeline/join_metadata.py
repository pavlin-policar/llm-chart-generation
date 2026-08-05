import argparse
import glob
import json
import os
import re


def join_files(current_directory, prefix):
    output_filepath = os.path.join(current_directory, f"{prefix}.jsonl")

    if os.path.exists(output_filepath):
        print(f"Skipping {output_filepath}")
        return

    pattern = os.path.join(current_directory, f"{prefix}*.jsonl")
    regex = re.compile(rf"^{prefix}\d+(?:_\d+)*\.jsonl$")

    matching_files = []
    for filepath in glob.glob(pattern):
        current_filename = os.path.basename(filepath)

        if regex.match(current_filename):
            matching_files.append(filepath)

    if not matching_files:
        return

    with open(output_filepath, "w", encoding="utf-8") as f_full:
        for filepath in sorted(matching_files):
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    j = json.loads(line)
                    f_full.write(json.dumps(j, ensure_ascii=False) + "\n")

    print(f"Created {output_filepath}")


def resave_dataset(directory="./dataset"):
    for current_directory, _, _ in os.walk(directory):
        join_files(current_directory, "metadata")
        join_files(current_directory, "error")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Join metadata and error JSONL files in all subdirectories."
    )
    parser.add_argument(
        "--directory",
        type=str,
        default="./dataset",
        help="Directory containing dataset subdirectories.",
    )

    args = parser.parse_args()
    resave_dataset(directory=args.directory)