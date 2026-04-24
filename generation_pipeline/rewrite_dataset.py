from pathlib import Path
import json
import glob
import tempfile
import shutil

JSONL_GLOB = "/workspace/projects/generate-graphs/dataset/*.jsonl"

OLD_PREFIX = "/workspace/projects/generate-graphs/dataset/"
# result: dataset/images/...
# if you want only images/... then use OLD_PREFIX = "/workspace/projects/generate-graphs/dataset/"

for filepath in glob.glob(JSONL_GLOB):
    filepath = Path(filepath)

    if "metadata.jsonl" not in str(filepath):
        continue

    with open(filepath, "r", encoding="utf-8") as f, \
         tempfile.NamedTemporaryFile("w", delete=False, dir=filepath.parent, encoding="utf-8") as tmp:



        for line in f:
            obj = json.loads(line)

            if "images" in obj and isinstance(obj["images"], list):
                for img in obj["images"]:
                    if isinstance(img, dict) and "path" in img and isinstance(img["path"], str):
                        if img["path"].startswith(OLD_PREFIX):
                            img["path"] = img["path"][len(OLD_PREFIX):]

            tmp.write(json.dumps(obj, ensure_ascii=False) + "\n")

    shutil.move(tmp.name, filepath)

print("Done.")