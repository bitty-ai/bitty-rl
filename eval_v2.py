import json
import os
import sys

import torch

from inference import DEVICE, load

EVAL_DIR = "/tmp/bitty-rl-eval"
sys.path.insert(0, EVAL_DIR)
from reward import extract_answer, reward  # noqa: E402

DATA = f"{EVAL_DIR}/cipher_rl_v2.jsonl"
OUT = f"{EVAL_DIR}/results_v2.jsonl"
LEVELS = {1, 2}
SAMPLES = 16
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.9


def sample(tokenizer, model, prompt):
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=1.0,
            top_k=0,
            num_return_sequences=SAMPLES,
        )
    return tokenizer.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def main():
    rows = [r for r in map(json.loads, open(DATA)) if r["level"] in LEVELS]
    done = set()
    if os.path.exists(OUT):
        done = {json.loads(line)["id"] for line in open(OUT)}

    tokenizer, model = load()
    with open(OUT, "a") as f:
        for row in rows:
            if row["id"] in done:
                continue
            for i, output in enumerate(sample(tokenizer, model, row["prompt"])):
                rec = {"id": row["id"], "level": row["level"], "sample_idx": i, "output": output,
                       "score": reward(output, row["answer"])}
                f.write(json.dumps(rec) + "\n")
            f.flush()

    results = [json.loads(line) for line in open(OUT)]
    print("level | mean score | perfect/16 | no <answer> tag")
    for row in rows:
        rs = [r for r in results if r["id"] == row["id"]]
        mean = sum(r["score"] for r in rs) / len(rs)
        perfect = sum(r["score"] == 1.0 for r in rs)
        no_tag = sum(extract_answer(r["output"]) is None for r in rs)
        nonzero = sum(r["score"] > 0 for r in rs)
        print(f"{row['level']} | {mean:.3f} | {perfect}/{len(rs)} | {no_tag} | nonzero={nonzero}")


if __name__ == "__main__":
    main()
