from pathlib import Path
import json
import glob
import tempfile
import shutil

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from urllib.parse import urlencode
from urllib.request import urlopen, Request
from tqdm import tqdm

JSONL_GLOB = "/workspace/projects/generate-graphs/dataset/*.jsonl"

llm = ChatOpenAI(
    model="qwen3.5",   
    openai_api_key="EMPTY",  # required but ignored by vLLM
    openai_api_base="http://localhost:8888/v1",
    extra_body= {
        "chat_template_kwargs": {"enable_thinking": False}
    }
)

def after_think(text: str):
    athink = text.split("</think>", 1)[1] if "<think>" in text else text
    think = text.split("</think>", 1)[0] if "<think>" in text else None
    return think, athink

def give_question_types(questions, prev):
    prompt = (
        "You are a question categorizer.\n"
        "YOu will be given questions and you have to assing types to them.\n"
        "You can assign questions only these labels:\n"
        "  - 'metadata': The question asks for chart text or styling directly visible in the image, such as the title, axis labels, legend entries, colors, units, or tick values.\n"
        "  - 'value extraction': The question asks for the value of a specific plotted element or local visual quantity, such as a bar height, point value, coordinate, count, or frequency.\n"
        "  - 'comparison': The question asks to compare two or more visual elements, categories, series, or values.\n"
        "  - 'trends': The question asks about the overall pattern, structure, or distribution in the plot, such as increase/decrease, skewness, clustering, gaps, outliers, seasonality, or general shape.\n"
        "  - 'reasoning': The question requires combining visual evidence from the image with arithmetic, dataset context, or external information to infer the answer. Overall requires a more complex reasoning process.\n\n"
        "Rules:"
        "   - Respond only with a valid array of string values from the previous list.\n"
        "   - You may NOT add or remove questions.\n"
        "   - You must give each question EXACTLY ONE label.\n"
        f"  - Length of output array MUST BE EXACTLY {len(questions)}, in previous iteration it was {prev}\n\n"
        "Questions:\n"
        f"{json.dumps(questions, indent=2)}"
    
    )

    out = llm.invoke(prompt).content
    _, out = after_think(out)

    out = json.loads(out.replace("```json", "").replace("```", ""))

    return out

for filepath in glob.glob(JSONL_GLOB):
    filepath = Path(filepath)

    if "metadata2.jsonl" not in str(filepath):
        continue

    with open(filepath, "r", encoding="utf-8") as f, \
        tempfile.NamedTemporaryFile("w", delete=False, dir=filepath.parent, encoding="utf-8") as tmp:

        for i, line in enumerate(f):
            print(i)
            obj = json.loads(line)

            questions = obj["graph"]["questions"]

            try:
                rerun_labels = 0
                labels = []
                
                while len(labels) != len(questions):
                    labels = give_question_types(questions, prev=len(labels))
                    print(len(labels), len(questions))

                for l, q in enumerate(questions):
                    questions[l]["type"] = labels[l]

            except Exception as e:
                print("Couldnt generate question types", e)

            tmp.write(json.dumps(obj, ensure_ascii=False) + "\n")

    shutil.move(tmp.name, "metadata3.jsonl")

print("Done.")