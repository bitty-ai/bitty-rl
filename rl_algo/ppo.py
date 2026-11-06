# PPO algo 

# REINFORCE ALGO 

## Start out with rollouts, and we need to see how good is this policy 
## So we start out by taking all the possible trajectories , weighted by probabilities

# J(Theta) = SUM(P(t ; Theta) * R(t))  -> prob of taking this trajectory and reward from that trajectory 

# So we now want to increase this value of J(theta) , to make a better model but we dont know here to nudge my params in ? 

# We take gradient wrt Theta
# d(J(Theta)) = SUM [d(P(T ; Theta)) * R(t)]

# calculus trick : d[P(T | Theta)] = P(T | Theta) * d[log(P(T | Theta))]
# d(J(Theta)) = SUM [P(T | Theta) * d[log(P(T | Theta))] * R(t)]

# Expectation = E ( d [log [P(T;Theta)]] * R(t)] )

# P(T;Theta) = starting state * MUL [POLICY( a_t | s_t )] * P(T ; Theta) -> initial state * prob of taking that trajectory * prob of next state in world 

# log [P(T;Theta)] = 0 + SUM (log (POLICY (a_t | s_t))) + 0 -> 0 cause those are constants 
# d [log [P(T;Theta)]] = d[SUM (log (POLICY (a_t ; s_t)))]

# d(J(Theta)) =  Expectation{ d[log(POLICY (a_t ; s_t))] * R(t)]} --> PG LOSS DERIVATION 

# This is a surrogate loss here its not something that we get directly from expecation maximization

# --x--- 

# REINFORCE / PG Loss = -R * log (policy (output | prompt )) # a long derivation to come to this point 
# Baselining : loss = -(R - R_avg) * log (POLICY (output | prompt))
# Advantage : loss = -A * log (POLICY (output | prompt))
# importance sampling : So we sample ones and want to make over it 4-8 gradient steps as inference is also expensive in case of LMs (this is the main problem)

# So the PG loss assumes that we are updating the same policy from which rollouts have been made, so this is expensive for RL and at each iteration model actually becomes a bit better so to balance that out we are maintaining a ratio between old and new model

# ratio = log[policy(output | prompt) / prev_step_policy(output | prompt )]
# importance ratio = exp(ratio) = policy(output | prompt) / prev_step_policy(output | prompt) = new/old

## ratio == 1 : first time iterating model 

## ratio > 1 : model is more likely to output that , so (wrong / right) penalise on either side

# So integrate this importance sampling in this PG loss we use a new surrogate loss here = - Adv * ratio ( how ? , do proof by verification here), we 

### policy : new , prev_step : old
# l = -A * r 
# dl = -A * dr ... (1)
# dr = d(new/old)
# dr = (1/old) * d(new) ... (2)
# using this property : d(log(x)) = (1/x) * dx , we get
# d(new) = new * d(log(new)) , using this in (2)
# dr = (1/old) * new * d(log(new))
# dr = (new/old) * d(log(new))
# dr = Ratio * d (log(new))
# Putting this in (1)

# dl = -A * Ratio * d(log(new))
# d(pg_loss) = -A * d(log(new)), for ratio = 1

# --x-- SAME PATTERN FOR PG-LOSS AND FOR IMPT. SAMPLING LOSS , aka surrogate loss 

# here at step 7, 8 the ratio can become higher like 5, 10 and multiplying with advantage can lead to exploding gradient so we use this minimizing term here 
# policy_loss = -min(A * Ratio , A * min(ratio , 1-e, 1+e)) 

# KLD (dont diverge from the reference model)
# KL[ POLICY( output | prompt ) || REFERENCE( output | prompt )]

# kld_loss = Beta * (log(POLICY( output | prompt )) - log( REFERENCE( output | prompt )))

# loss = policy_loss + kld_loss
# length normalizing : (1/Length) * loss , here length is the masked length

# ---x---
# maintain : policy, ref_model

import torch 
from base import model 
from base import tokenizer
import copy 
import sys
import re


policy = model
reference = copy.deepcopy(model)

reference.eval()
for p in reference.parameters():
    p.requires_grad_(False)

PROMPT = "The capital of France is"
TARGET = "Paris"
G, STEPS, MAX_NEW, EPS, BETA, LR,K = 8, 100, 8, 0.2, 0.05, 1e-6, 2
opt = torch.optim.AdamW(policy.parameters(), lr=LR)


def reward(text):
    words = re.findall(r"[a-z]+", text.lower())
    return float(sum(w == TARGET.lower() for w in words))
        
def token_logprobs(model, ids, prompt_len):
    logits = model(ids).logits[:, :-1].float()
    lp = torch.log_softmax(logits, -1).gather(-1, ids[:, 1:, None]).squeeze(-1)
    return lp[:, prompt_len - 1:] # log prob of each generated token 


device = next(policy.parameters()).device
for step in range(STEPS):

    prompt_ids = tokenizer(PROMPT, return_tensors='pt').input_ids.repeat(G, 1)
    prompt_ids = prompt_ids.to(device)
    
    # Policy-Loss
    with torch.no_grad():
        ids = policy.generate(
            prompt_ids,
            max_new_tokens=MAX_NEW,
            do_sample=True,
            top_k=0,
            pad_token_id=tokenizer.eos_token_id
        )
        prompt_length = prompt_ids.shape[-1]
        mask = (ids[:, prompt_length:] == tokenizer.eos_token_id).cumsum(-1) <= 1
        ref_lp = token_logprobs(reference, ids, prompt_length)
        old_lp = token_logprobs(policy, ids, prompt_length) # prev policy, but for each step this is also getting updated so why do we need importance sampling here ?  

    texts = tokenizer.batch_decode(ids[:, prompt_ids.shape[1]:])
    R = torch.tensor([reward(t) for t in texts], device=device)
    A = (R-R.mean()) # baselining
    
    for _ in range(K):
        new_lp = token_logprobs(policy, ids, prompt_length)
        ratio = torch.exp(new_lp - old_lp)
        clipped = torch.clamp(ratio , 1-EPS, 1+EPS)
        pg = torch.min(ratio * A[:, None] , A[:,None] * clipped)
        
        # KLD-Loss
        # kl = (new_lp - ref_lp) ( faulty one)
        kl = torch.exp(ref_lp - new_lp) - (ref_lp - new_lp) - 1
        
        # loss 
        loss = -(pg - BETA * kl)
        loss = (loss * mask).sum() / mask.sum()
        
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
        print(f"step {step:3d}  mean_R={R.mean():.2f}  sample={texts}")