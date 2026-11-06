import os
import sys

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

MODEL_DIR = os.environ.get("MODEL_DIR", "/tmp/Qwen3.5-0.8B")
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def load():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForImageTextToText.from_pretrained(MODEL_DIR, dtype=torch.bfloat16).to(DEVICE).eval()
    return tokenizer, model


def generate(tokenizer, model, prompt, max_new_tokens=64, thinking=False):
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=thinking,
    )
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    raw = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    reasoning, sep, answer = raw.rpartition("</think>")
    if not sep:
        # no closing tag: thinking was cut off, or thinking was off
        return ("", raw.strip()) if not thinking else (raw.strip(), "")
    return reasoning.replace("<think>", "").strip(), answer.strip()


if __name__ == "__main__":
    tokenizer, model = load()
    print(generate(tokenizer, model, " ".join(sys.argv[1:]) or "Write one sentence about the ocean.")[1])
