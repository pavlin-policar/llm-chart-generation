import argparse
import glob
import json
import os
import re


def resave_dataset(min_number=0, directory="./dataset", filename="metadata.jsonl"):
    join_filepath = os.path.join(directory, filename)

    pattern = os.path.join(directory, "metadata*.jsonl")
    regex = re.compile(r"metadata(\d{1}).jsonl$")

    matching_files = []
    for filepath in glob.glob(pattern):
        filename = os.path.basename(filepath)
        match = regex.match(filename)

        if match and int(match.group(1)) >= min_number:
            matching_files.append(filepath)

    with open(join_filepath, "a", encoding="utf-8") as f_full:
        for filepath in sorted(matching_files):
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    j = json.loads(line)
                    f_full.write(json.dumps(j, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Join multiple metadata JSONL files into a single metadata.jsonl file.")
    parser.add_argument(
        "--min-number",
        type=int,
        default=0,
        help="Minimum numeric suffix of metadata files to include (e.g. 0 for metadata0.jsonl and above).",
    )
    parser.add_argument(
        "--directory",
        type=str,
        default="./dataset",
        help="Directory containing metadata files.",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="metadata.jsonl",
        help="Output filename for the joined metadata.",
    )

    args = parser.parse_args()
    resave_dataset(min_number=args.min_number, directory=args.directory, filename=args.filename)
