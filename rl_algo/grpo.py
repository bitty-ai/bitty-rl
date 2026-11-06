# ruff: noqa
# GRPO algo 
# GRPO Done Right (DR)

# PG Loss = -R * log (policy (output | prompt ))
# Baselining : loss = -(R - R_avg) * log (POLICY (output | prompt))
# Advantage : loss = -A * log (POLICY (output | prompt))
# importance sampling : so we sample ones and want to make over it 4-8 gradient steps as inference is also expensive in case of LMs 

# ratio = log[policy(output | prompt) / prev_step_policy(output | prompt )]
# importance ratio = exp(ratio) = policy(output | prompt) / prev_step_policy(output | prompt) = new/old

## ratio > 1 : model is more likely to output that , so (wrong / right) penalise on either side
 
# loss = - Adv * ratio ( how ? , do proof by verification here), we 

### policy : new , prev_step : old
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
# d(pg_loss) = -R * d(log(new))

# --x-- SAME PATTERN FOR PG-LOSS AND FOR IMPT. SAMPLING LOSS , aka surrogate loss 

# policy_loss = -min (A * Ratio , A * min(ratio , 1-e, 1+e)) 

# KLD (dont diverge from the reference model)
# KL[ POLICY( output | prompt ) || REFERENCE( output | prompt )]

# kld_loss = Beta * (log(POLICY( output | prompt )) - log( REFERENCE( output | prompt )))

# loss = policy_loss + kld_loss
# length normalizing : (1/(G * MAX_NEW)) * loss 

# ---x---
# maintain : policy, ref_model

# ---x---
# remove std from original implementation : so for and easy or hard question where variance is very small this advantage term gets amplified
# remove (1/length) multiplication from the implementation : reward hacking, model learns to output more tokens to bring down average loss here l


import torch
import copy
import torch.nn as nn
from data.io import extract_answer, SYSTEM_PROMPT
from logger import logger

class GRPO(nn.Module):
    def __init__(self, model, tokenizer, dataloader):
        super().__init__()

        G, STEPS, MAX_NEW, EPS, BETA, LR,K = 8, 100, 256, 0.2, 0.05, 1e-6, 2

        self.G = G
        self.STEPS = STEPS
        self.MAX_NEW = MAX_NEW
        self.EPS = EPS
        self.BETA = BETA
        self.K = K

        self.policy = model 
        self.reference = copy.deepcopy(model)

        self.reference.eval()
        for p in self.reference.parameters():
            p.requires_grad_(False)

        self.dataloader = dataloader
        
        self.opt = torch.optim.AdamW(self.policy.parameters(), lr=LR)    
        self.tokenizer = tokenizer    

    def reward(self, text, target):
        return float(extract_answer(text) == target)
            
    def token_logprobs(self, model, ids, prompt_len):
        logits = model(ids).logits[:, prompt_len - 1:-1].float()
        lp = torch.log_softmax(logits, -1).gather(-1, ids[:, prompt_len:, None]).squeeze(-1)
        return lp # log prob of each generated token


    def train(self):
        device = next(self.policy.parameters()).device
        
        for step in range(self.STEPS):
            for qa in self.dataloader:
                question = qa['question'][0]
                answer = qa['answer'][0]
                chat = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}]
                text = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True) + "<think>"
                prompt_ids = self.tokenizer(text, return_tensors='pt', add_special_tokens=False).input_ids.repeat(self.G, 1)
                prompt_ids = prompt_ids.to(device)
                
                # Policy-Loss
                with torch.no_grad():
                    ids = self.policy.generate(
                        prompt_ids,
                        max_new_tokens=self.MAX_NEW,
                        do_sample=True,
                        top_k=0,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                    prompt_length = prompt_ids.shape[-1]
                    mask = (ids[:, prompt_length:] == self.tokenizer.eos_token_id).cumsum(-1) <= 1
                    ref_lp = self.token_logprobs(self.reference, ids, prompt_length)
                    old_lp = self.token_logprobs(self.policy, ids, prompt_length) # prev policy, but for each step this is also getting updated so why do we need importance sampling here ?

                texts = self.tokenizer.batch_decode(ids[:, prompt_ids.shape[1]:])
                R = torch.tensor([self.reward(t, answer) for t in texts], device=device)
                A = (R-R.mean()) # baselining
                logger.info(f"gold={answer} preds={[extract_answer(t) for t in texts]} R={R.tolist()}")
                if R.max() == R.min():
                    continue
                
                for _ in range(self.K):
                    new_lp = self.token_logprobs(self.policy, ids, prompt_length)
                    ratio = torch.exp(new_lp - old_lp)
                    clipped = torch.clamp(ratio , 1-self.EPS, 1+self.EPS)
                    pg = torch.min(ratio * A[:, None] , A[:,None] * clipped)
                    
                    # KLD-Loss
                    # kl = (new_lp - ref_lp) ( faulty one)
                    kl = torch.exp(ref_lp - new_lp) - (ref_lp - new_lp) - 1
                    
                    # loss 
                    loss = -(pg - self.BETA * kl)
                    loss = (loss * mask).sum() / (self.G * self.MAX_NEW)
                    
                    self.opt.zero_grad()
                    loss.backward()

                    # gradient norm
                    total_norm = 0.0
                    for p in self.policy.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2

                    total_norm = total_norm**0.5
                    logger.debug(f"grad_norm={total_norm:.4f} loss={loss.item():.4f}")
                    
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
                    self.opt.step()

                if step % 10 == 0:
                    logger.info(f"step {step:3d}  mean_R={R.mean():.2f}  sample={texts[0]!r}")


