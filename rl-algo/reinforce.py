# REINFORCE in simple terms 
# loss = - Reward * SUM[log(POLICY(answer | prompt))]

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


from base import model as policy
from base import tokenizer
import torch 

PROMPT = "The capital of Delhi is"
TARGET = "India"
G, STEPS, MAX_NEW, EPS, BETA, LR = 1, 100, 1, 0.2, 0.05, 1e-4
opt = torch.optim.AdamW(policy.parameters(), lr=LR)

def reward(output):
    return 1.0 if TARGET in output else 0.0


def token_logprobs(model, ids, prompt_len):
    logits = model(ids).logits[:, :-1]
    lp = torch.log_softmax(logits, -1).gather(-1, ids[:, 1:, None]).squeeze(-1)
    return lp[:, prompt_len - 1:] # log prob of each generated token 



device = next(policy.parameters()).device
for step in range(STEPS):

    prompt_ids = tokenizer(PROMPT, return_tensors='pt').input_ids.repeat(G, 1)
    prompt_ids = prompt_ids.to(device)

    ids = policy.generate(
        prompt_ids,
        max_new_tokens=MAX_NEW,
        do_sample=True,
        top_k=0,
        pad_token_id=tokenizer.eos_token_id
    )

    lp = token_logprobs(policy, ids, prompt_ids.shape[1])
    texts = tokenizer.batch_decode(ids[:, prompt_ids.shape[1]:])
    R = torch.tensor([reward(t) for t in texts], device=device)

    loss = (-R * lp)
    
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
    
    opt.step()

    if step % 10 == 0:
        print(f"step {step:3d}  mean_R={R.mean():.2f}  sample={texts[0]!r}")