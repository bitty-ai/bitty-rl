import argparse
import contextlib
import glob
import json
import os
import random
import time

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

from eval_v3 import DATA, EVAL_DIR, REVERSE_EXAMPLE, chat, extract_answer, generate, reward
from inference import DEVICE, MODEL_DIR

PROMPT_TEMPLATE = (
    "You will see examples of a secret code. Letters are separated by spaces.\n"
    "Find the rule from the examples, then decode the last line.\n\n"
    "Examples:\n{examples}\n\n"
    "Decode: {query}\n\n"
    "Think briefly. Check your rule on every example before answering.\n"
    "Put the final answer inside <answer></answer>, with letters separated by spaces."
)

# dataset answers (bed, hi, red) are left out so eval puzzles never appear in training
WORDS = """at up in on it go no we me us be do to so am an as by he if is my of or ox
cat dog sun map pen hat top cup box fox pig cow hen egg ink jam key leg lip mud net nut owl pot rat
sea sky toy van web yes zoo ant arm bag bat bus car day ear eye fan fig fly gum hut ice jar kid lid
log man mix oak pan pie rug sad six tap ten tub wax wig yak fish frog bird book tree star moon ship
cake milk rain snow wind fire lamp door hand foot ring coin gold king lake leaf nest rock sand rose
wolf duck goat bear lion seed bell drum kite boat home park road""".split()
SHIFTS = (1, 2)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--group", type=int, default=16)
    p.add_argument("--max-new", type=int, default=256)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--beta", type=float, default=0.05)
    p.add_argument("--eps", type=float, default=0.2)
    p.add_argument("--micro-batch", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=f"{EVAL_DIR}/grpo_run1")
    return p.parse_args()


def encode(word, shift):
    return " ".join(chr((ord(c) - 97 + shift) % 26 + 97) for c in word)


def make_puzzle(rng):
    shift = rng.choice(SHIFTS)
    target, *example_words = rng.sample(WORDS, 5)
    examples = "\n".join(f"{encode(w, shift)} → {' '.join(w)}" for w in example_words)
    prompt = PROMPT_TEMPLATE.format(examples=examples, query=encode(target, shift))
    return {"shift": shift, "target": target, "prompt": REVERSE_EXAMPLE + prompt, "answer": " ".join(target)}


def autocast():
    if DEVICE == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def token_logprobs(model, ids, prompt_len):
    logits = model(ids, logits_to_keep=ids.shape[1] - prompt_len + 1).logits[:, :-1].float()
    return torch.log_softmax(logits, -1).gather(-1, ids[:, prompt_len:, None]).squeeze(-1)


def batched_logprobs(model, ids, prompt_len, micro_batch):
    with torch.no_grad(), autocast():
        return torch.cat([token_logprobs(model, ids[i:i + micro_batch], prompt_len)
                          for i in range(0, len(ids), micro_batch)])


def completion_mask(completion, end_ids):
    is_end = torch.isin(completion, end_ids)
    return (is_end.cumsum(-1) - is_end.long()) == 0  # keep tokens up to and including the first end token


def evaluate(policy, tokenizer, rows, args, step):
    policy.eval()
    with open(f"{args.out}/eval_log.jsonl", "a") as log, open(f"{args.out}/eval_samples.jsonl", "a") as samples:
        for row in rows:
            with autocast():
                outs = generate(tokenizer, policy, [chat(tokenizer, REVERSE_EXAMPLE + row["prompt"], False)] * args.group,
                                args.max_new)
            scores = [reward(o, row["answer"]) for o in outs]
            for i, (o, s) in enumerate(zip(outs, scores)):
                samples.write(json.dumps({"step": step, "level": row["level"], "sample_idx": i, "output": o, "score": s}) + "\n")
            rec = {
                "step": step,
                "level": row["level"],
                "mean_score": sum(scores) / len(scores),
                "perfect": sum(s == 1.0 for s in scores),
                "no_tag": sum(not extract_answer(o) for o in outs),
                "above_0.1": sum(s > 0.1 for s in scores),
            }
            log.write(json.dumps(rec) + "\n")
            log.flush()
            samples.flush()
            print(f"EVAL step={step} {rec}", flush=True)


def save_checkpoint(policy, tokenizer, args, step):
    path = f"{args.out}/step_{step}"
    policy.save_pretrained(path, state_dict={k: v.to(torch.bfloat16) for k, v in policy.state_dict().items()})
    tokenizer.save_pretrained(path)


def latest_checkpoint(out):
    ckpts = sorted(glob.glob(f"{out}/step_*"), key=lambda p: int(p.rsplit("_", 1)[1]))
    return (ckpts[-1], int(ckpts[-1].rsplit("_", 1)[1])) if ckpts else (None, 0)


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    with open(f"{args.out}/config.json", "w") as f:
        json.dump({**vars(args), "model": MODEL_DIR, "words": len(WORDS), "shifts": SHIFTS}, f, indent=2)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    tokenizer.padding_side = "left"
    ckpt, start = latest_checkpoint(args.out)
    # policy keeps fp32 master weights; bf16 would round away updates at lr ~1e-6
    policy = AutoModelForImageTextToText.from_pretrained(ckpt or MODEL_DIR, dtype=torch.float32).to(DEVICE)
    reference = AutoModelForImageTextToText.from_pretrained(MODEL_DIR, dtype=torch.bfloat16).to(DEVICE).eval()
    reference.requires_grad_(False)
    for name, p in policy.named_parameters():
        p.requires_grad_(not any(k in name for k in ("visual", "embed_tokens", "lm_head")))
    policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    trainable = [p for p in policy.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr)
    print(f"start={start} trainable={sum(p.numel() for p in trainable) / 1e6:.0f}M resumed_from={ckpt}", flush=True)

    eos = policy.generation_config.eos_token_id
    end_ids = torch.tensor(sorted({*(eos if isinstance(eos, list) else [eos]), tokenizer.pad_token_id}), device=DEVICE)
    eval_rows = [r for r in map(json.loads, open(DATA)) if r["level"] in {1, 2, 3}]
    eval_log = f"{args.out}/eval_log.jsonl"
    evaluated = {json.loads(line)["step"] for line in open(eval_log)} if os.path.exists(eval_log) else set()

    for step in range(start, args.steps + 1):
        if step % args.eval_every == 0 or step == args.steps:
            if step not in evaluated:
                evaluate(policy, tokenizer, eval_rows, args, step)
            if step > start:
                save_checkpoint(policy, tokenizer, args, step)
        if step == args.steps:
            break

        t0 = time.time()
        puzzle = make_puzzle(random.Random(args.seed * 1_000_003 + step))
        prompt_ids = tokenizer(chat(tokenizer, puzzle["prompt"], False), return_tensors="pt").input_ids.to(DEVICE)
        prompt_len = prompt_ids.shape[1]

        policy.eval()
        with torch.no_grad(), autocast():
            ids = policy.generate(
                prompt_ids,
                max_new_tokens=args.max_new,
                do_sample=True,
                temperature=0.9,
                top_p=1.0,
                top_k=0,
                num_return_sequences=args.group,
                pad_token_id=tokenizer.pad_token_id,
            )
        completion = ids[:, prompt_len:]
        mask = completion_mask(completion, end_ids)
        texts = tokenizer.batch_decode(completion, skip_special_tokens=True)
        R = torch.tensor([reward(t, puzzle["answer"]) for t in texts], device=DEVICE)
        A = R - R.mean()

        loss_total, kl_total, grad_norm = 0.0, 0.0, 0.0
        if A.abs().sum() > 0:
            policy.train()
            ref_lp = batched_logprobs(reference, ids, prompt_len, args.micro_batch)
            old_lp = batched_logprobs(policy, ids, prompt_len, args.micro_batch)
            opt.zero_grad()
            for i in range(0, args.group, args.micro_batch):
                sl = slice(i, i + args.micro_batch)
                with autocast():
                    new_lp = token_logprobs(policy, ids[sl], prompt_len)
                ratio = torch.exp(new_lp - old_lp[sl])
                clipped = torch.clamp(ratio, 1 - args.eps, 1 + args.eps)
                pg = torch.min(ratio * A[sl, None], clipped * A[sl, None])
                kl = torch.exp(ref_lp[sl] - new_lp) - (ref_lp[sl] - new_lp) - 1
                loss = (-(pg - args.beta * kl) * mask[sl]).sum() / (args.group * args.max_new)
                loss.backward()
                loss_total += loss.item()
                kl_total += (kl * mask[sl]).sum().item()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0).item()
            opt.step()

        rec = {
            "step": step,
            "shift": puzzle["shift"],
            "target": puzzle["target"],
            "mean_R": R.mean().item(),
            "perfect": int((R == 1.0).sum()),
            "no_tag": sum(not extract_answer(t) for t in texts),
            "loss": loss_total,
            "kl": kl_total / max(mask.sum().item(), 1),
            "grad_norm": grad_norm,
            "mean_len": mask.sum(-1).float().mean().item(),
            "sec": round(time.time() - t0, 1),
            "sample": texts[0],
        }
        with open(f"{args.out}/train_log.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"step {step:4d} R={rec['mean_R']:.3f} perfect={rec['perfect']}/{args.group} "
              f"no_tag={rec['no_tag']} kl={rec['kl']:.4f} len={rec['mean_len']:.0f} {rec['sec']}s", flush=True)


if __name__ == "__main__":
    main()
