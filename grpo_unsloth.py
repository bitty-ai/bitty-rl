# ruff: noqa: I001
import unsloth  # noqa: F401  must be imported before trl/transformers so its patches apply
from unsloth import FastLanguageModel

import argparse
import json
import os
import random

from datasets import Dataset
from transformers.trainer_utils import get_last_checkpoint
from trl import GRPOConfig, GRPOTrainer

from eval_v3 import EVAL_DIR, extract_answer, reward
from grpo_cipher import make_puzzle

MODEL_NAME = os.environ.get("MODEL_NAME", "unsloth/Qwen2.5-1.5B-Instruct")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--puzzles", type=int, default=2000)
    p.add_argument("--group", type=int, default=8)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--max-seq", type=int, default=1024)
    p.add_argument("--max-prompt", type=int, default=512)
    p.add_argument("--max-new", type=int, default=256)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--beta", type=float, default=0.04)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--load-4bit", action="store_true")
    p.add_argument("--vllm", action="store_true")
    p.add_argument("--gpu-mem", type=float, default=0.6)
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=f"{EVAL_DIR}/grpo_unsloth_run1")
    return p.parse_args()


def build_dataset(n, seed):
    rows = []
    for i in range(n):
        puzzle = make_puzzle(random.Random(seed * 1_000_003 + i))
        rows.append({
            "prompt": [{"role": "user", "content": puzzle["prompt"]}],
            "answer": puzzle["answer"],
            "shift": puzzle["shift"],
        })
    return Dataset.from_list(rows)


def make_reward_fn(log_path):
    def cipher_reward(prompts, completions, answer, **kwargs):
        texts = [c[0]["content"] for c in completions]
        scores = [reward(t, a) for t, a in zip(texts, answer)]
        with open(log_path, "a") as f:
            f.writelines(
                json.dumps({"answer": a, "score": s, "has_tag": bool(extract_answer(t)), "output": t}) + "\n"
                for t, a, s in zip(texts, answer, scores)
            )
        return scores

    return cipher_reward


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    with open(f"{args.out}/config.json", "w") as f:
        json.dump({**vars(args), "model": MODEL_NAME}, f, indent=2)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=args.max_seq,
        load_in_4bit=args.load_4bit,
        fast_inference=args.vllm,
        max_lora_rank=args.lora_rank,
        gpu_memory_utilization=args.gpu_mem,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        lora_alpha=args.lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    config = GRPOConfig(
        output_dir=args.out,
        learning_rate=args.lr,
        beta=args.beta,
        temperature=args.temperature,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        num_generations=args.group,
        max_prompt_length=args.max_prompt,
        max_completion_length=args.max_new,
        max_steps=args.steps,
        save_steps=args.save_every,
        logging_steps=1,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        max_grad_norm=1.0,
        optim="adamw_8bit",
        use_vllm=args.vllm,
        seed=args.seed,
        report_to="none",
    )
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[make_reward_fn(f"{args.out}/completions.jsonl")],
        args=config,
        train_dataset=build_dataset(args.puzzles, args.seed),
    )
    trainer.train(resume_from_checkpoint=get_last_checkpoint(args.out))
    model.save_pretrained(f"{args.out}/lora_final")
    tokenizer.save_pretrained(f"{args.out}/lora_final")


if __name__ == "__main__":
    main()
