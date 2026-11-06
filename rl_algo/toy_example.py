import copy
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "gpt2"
PROMPT = "The capital of France is"
TARGET = "Paris"
G, STEPS, MAX_NEW, EPS, BETA, LR = 8, 100, 8, 0.2, 0.05, 1e-4

tok = AutoTokenizer.from_pretrained(MODEL)
tok.pad_token = tok.eos_token
policy = AutoModelForCausalLM.from_pretrained(MODEL) #model
ref = copy.deepcopy(policy).eval()
for p in ref.parameters():
    p.requires_grad_(False)
opt = torch.optim.AdamW(policy.parameters(), lr=LR)


def reward(text):
    return 1.0 if TARGET.lower() in text.lower() else 0.0


def token_logprobs(model, ids, prompt_len):
    logits = model(ids).logits[:, :-1]
    lp = torch.log_softmax(logits, -1).gather(-1, ids[:, 1:, None]).squeeze(-1)
    return lp[:, prompt_len - 1:]


for step in range(STEPS):
    prompt_ids = tok(PROMPT, return_tensors="pt").input_ids.repeat(G, 1)
    with torch.no_grad():
        ids = policy.generate(prompt_ids, max_new_tokens=MAX_NEW, do_sample=True,
                              top_k=0, pad_token_id=tok.eos_token_id)
        old_lp = token_logprobs(policy, ids, prompt_ids.shape[1])
        ref_lp = token_logprobs(ref, ids, prompt_ids.shape[1])

    texts = tok.batch_decode(ids[:, prompt_ids.shape[1]:])
    R = torch.tensor([reward(t) for t in texts])
    A = (R - R.mean()) / (R.std() + 1e-6)

    new_lp = token_logprobs(policy, ids, prompt_ids.shape[1])
    
    ratio = torch.exp(new_lp - old_lp)
    clipped = torch.clamp(ratio, 1 - EPS, 1 + EPS)
    pg = torch.min(ratio * A[:, None], clipped * A[:, None])
    kl = torch.exp(ref_lp - new_lp) - (ref_lp - new_lp) - 1
    loss = -(pg - BETA * kl).mean()

    opt.zero_grad()
    loss.backward()
    opt.step()

    if step % 10 == 0:
        print(f"step {step:3d}  mean_R={R.mean():.2f}  kl={kl.mean():.3f}  sample={texts[0]!r}")