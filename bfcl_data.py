import argparse
import hashlib
import json
import os
import urllib.request

from datasets import Dataset

BFCL_DIR = os.environ.get("BFCL_DIR", "/tmp/bfcl")
BASE_URL = "https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/resolve/main"
CATEGORIES = ("simple", "multiple", "parallel", "parallel_multiple", "live_simple", "live_multiple")
EVAL_PCT = 10

SYSTEM_TEMPLATE = """You are a function calling assistant.

You are given these functions inside <tools></tools>:
<tools>
{tools}
</tools>

For each function call needed, return a JSON object with the function name and arguments inside <tool_call></tool_call>:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>
If several calls are needed, output one <tool_call> block per call. Output nothing else."""

def fetch(path):
    local = f"{BFCL_DIR}/{path}"
    if not os.path.exists(local):
        os.makedirs(os.path.dirname(local), exist_ok=True)
        urllib.request.urlretrieve(f"{BASE_URL}/{path}", local)
    with open(local) as f:
        return [json.loads(line) for line in f if line.strip()]


def to_json_schema(node):
    if isinstance(node, dict):
        return {k: ("object" if k == "type" and v == "dict" else to_json_schema(v)) for k, v in node.items()}
    if isinstance(node, list):
        return [to_json_schema(v) for v in node]
    return node


def build_messages(question, functions):
    tools = "\n".join(json.dumps({"type": "function", "function": to_json_schema(f)}) for f in functions)
    system = SYSTEM_TEMPLATE.format(tools=tools)
    turn = question[0]
    extra_system = [m["content"] for m in turn if m["role"] == "system"]
    if extra_system:
        system = "\n\n".join(extra_system) + "\n\n" + system
    return [{"role": "system", "content": system}] + [m for m in turn if m["role"] != "system"]


def is_eval(row_id):
    return int(hashlib.md5(row_id.encode()).hexdigest(), 16) % 100 < EVAL_PCT


def load_rows(categories=CATEGORIES):
    rows = []
    for cat in categories:
        answers = {a["id"]: a["ground_truth"] for a in fetch(f"possible_answer/BFCL_v3_{cat}.json")}
        for q in fetch(f"BFCL_v3_{cat}.json"):
            if q["id"] not in answers:
                continue
            rows.append({
                "id": q["id"],
                "category": cat,
                "prompt": build_messages(q["question"], q["function"]),
                "ground_truth": json.dumps(answers[q["id"]]),
            })
    return rows


def load_split(split, categories=CATEGORIES):
    rows = [r for r in load_rows(categories) if is_eval(r["id"]) == (split == "eval")]
    return Dataset.from_list(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="train", choices=["train", "eval"])
    p.add_argument("--show", type=int, default=1)
    args = p.parse_args()
    ds = load_split(args.split)
    print(f"{args.split}: {len(ds)} rows")
    for row in ds.select(range(args.show)):
        print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
