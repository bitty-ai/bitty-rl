# DPO algo 

# PG Loss = -R * log (policy (output | prompt ))

# We start with pairs here that is : correct answer, incorrect answer
# simply increase the prob of getting the correct answer and decrease the prob of getting bad answer 

# loss = [new(correct | prompt) - ref(correct | prompt)] - [new(incorrect | prompt) - ref(incorrect | prompt)]

# scaled_loss =  Beta * loss
# sigmoid : squashes no. into (0,1)
# BCE : -log sigmoid(scaled_loss)

# ---x---
# maintain : policy, ref_model

import torch 
from base import model 
from base import tokenizer
import torch.nn.functional as F
import copy 
import sys
import re


policy = model
reference = copy.deepcopy(model)

reference.eval()
for p in reference.parameters():
    p.requires_grad_(False)

PROMPT = "The capital of France is"
CORRECT_ANSWER = " Paris." # adding space for toy implementation
INCORRECT_ANSWER = " Paris"
 
G, STEPS, MAX_NEW, EPS, BETA, LR,K = 1, 100, 1, 0.2, 0.05, 1e-6, 2
opt = torch.optim.AdamW(policy.parameters(), lr=LR)


def token_logprobs(model, ids, prompt_len):
    logits = model(ids).logits[:, :-1].float()
    lp = torch.log_softmax(logits, -1).gather(-1, ids[:, 1:, None]).squeeze(-1)
    return lp[:, prompt_len-1:]


device = next(policy.parameters()).device
for step in range(STEPS):
    prompt_ids = tokenizer(PROMPT, return_tensors="pt").input_ids

    prompt_correct_ans_ids = tokenizer(PROMPT + CORRECT_ANSWER, return_tensors='pt').input_ids.repeat(G, 1)
    prompt_correct_ans_ids = prompt_correct_ans_ids.to(device)
    
    prompt_incorrect_ans_ids = tokenizer(PROMPT + INCORRECT_ANSWER, return_tensors='pt').input_ids.repeat(G, 1)
    prompt_incorrect_ans_ids = prompt_incorrect_ans_ids.to(device)
    
    # Policy-Loss
    with torch.no_grad():
        prompt_length = prompt_ids.shape[-1]
        ref_corr_lp = token_logprobs(reference, prompt_correct_ans_ids, prompt_length).sum(-1)
        ref_incorr_lp = token_logprobs(reference, prompt_incorrect_ans_ids, prompt_length).sum(-1)
        
    new_corr_lp = token_logprobs(policy, prompt_correct_ans_ids, prompt_length).sum(-1)
    new_incorr_lp = token_logprobs(policy, prompt_incorrect_ans_ids, prompt_length).sum(-1)
    
    margin = BETA * ((new_corr_lp - ref_corr_lp) - (new_incorr_lp - ref_incorr_lp))
    loss = -F.logsigmoid(margin)
    loss = loss.mean()
        
    opt.zero_grad()
    loss.backward()

    # gradient norm
    total_norm = 0.0
    for p in policy.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2

    total_norm = total_norm**0.5
    print(f"Total Gradient Norm: {total_norm}")
    
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    opt.step()

    if step % 10 == 0:
        print(f"step {step:3d}  loss : {loss.detach().item()}")