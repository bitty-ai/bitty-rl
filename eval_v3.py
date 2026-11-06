import json
import os
import sys

import torch

from inference import DEVICE, load

EVAL_DIR = os.environ.get("EVAL_DIR", "/tmp/bitty-rl-eval")
sys.path.insert(0, EVAL_DIR)
from reward import extract_answer, reward  # noqa: E402

DATA = f"{EVAL_DIR}/cipher_rl_v2.jsonl"
LEVELS = {1, 2, 3}
SAMPLES = 16
SAMPLING = dict(do_sample=True, temperature=0.9, top_p=1.0, top_k=0)

WORKED_EXAMPLE = """Here is one solved example.
c b u → b a t
e p h → d o g
Decode: g p y
Solution: c→b is −1, b→a is −1, u→t is −1. Rule: each letter −1.
Check: e→d, p→o, h→g. Correct.
Decode: g→f, p→o, y→x.
<answer>f o x</answer>
---
Now solve this one the same short way.

"""


REVERSE_EXAMPLE = """Here is one solved example.
t a c → c a t
g o d → d o g
Decode: x o f
Solution: t a c backwards is c a t. Rule: reverse.
Check: g o d backwards is d o g. Correct.
Decode: x o f backwards is f o x.
<answer>f o x</answer>
Now solve this one the same short way.

"""


def chat(tokenizer, prompt, thinking):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=thinking,
    )


def generate(tokenizer, model, texts, max_new_tokens, stop=None):
    inputs = tokenizer(texts, return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens, stop_strings=stop, tokenizer=tokenizer, **SAMPLING
        )
    return tokenizer.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def run_no_thinking(tokenizer, model, prompt):
    return generate(tokenizer, model, [chat(tokenizer, prompt, False)] * SAMPLES, NO_THINK_TOKENS)


def run_budget_forcing(tokenizer, model, prompt):
    base = chat(tokenizer, prompt, True)
    thoughts = [t.split("</think>")[0] for t in generate(tokenizer, model, [base] * SAMPLES, 512, "</think>")]
    forced = [f"{t}</think>\n<answer>" for t in thoughts]
    tails = generate(tokenizer, model, [base + f for f in forced], 20, "</answer>")
    outputs = []
    for f, tail in zip(forced, tails):
        tail = tail.split("</answer>")[0]
        outputs.append(f + tail + "</answer>")
    return outputs


RUNS = {
    "A": lambda tok, m, p: run_no_thinking(tok, m, p),
    "B": run_budget_forcing,
    "C": lambda tok, m, p: run_no_thinking(tok, m, WORKED_EXAMPLE + p),
    "D": lambda tok, m, p: run_no_thinking(tok, m, REVERSE_EXAMPLE + p),
}


NO_THINK_TOKENS = 64


def main():
    global NO_THINK_TOKENS
    if len(sys.argv) > 2:
        NO_THINK_TOKENS = int(sys.argv[2])
    name = sys.argv[1]
    tag = sys.argv[3] if len(sys.argv) > 3 else ""
    out_path = f"{EVAL_DIR}/results_{name}{tag}.jsonl"
    rows = [r for r in map(json.loads, open(DATA)) if r["level"] in LEVELS]
    done = {json.loads(line)["id"] for line in open(out_path)} if os.path.exists(out_path) else set()

    tokenizer, model = load()
    tokenizer.padding_side = "left"
    with open(out_path, "a") as f:
        for row in rows:
            if row["id"] in done:
                continue
            for i, output in enumerate(RUNS[name](tokenizer, model, row["prompt"])):
                rec = {"id": row["id"], "level": row["level"], "sample_idx": i, "output": output,
                       "score": reward(output, row["answer"])}
                f.write(json.dumps(rec) + "\n")
            f.flush()

    results = [json.loads(line) for line in open(out_path)]
    for row in rows:
        rs = [r for r in results if r["id"] == row["id"]]
        mean = sum(r["score"] for r in rs) / len(rs)
        perfect = sum(r["score"] == 1.0 for r in rs)
        no_tag = sum(extract_answer(r["output"]) is None for r in rs)
        nonzero = sum(r["score"] > 0 for r in rs)
        above_bonus = sum(r["score"] > 0.1 for r in rs)
        print(f"RESULT {name}{tag} | {row['level']} | {mean:.3f} | {perfect}/{len(rs)} | {no_tag}/{len(rs)} "
              f"| nonzero={nonzero} | above0.1={above_bonus}/{len(rs)}")


if __name__ == "__main__":
    main()
