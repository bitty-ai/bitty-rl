#ruff :noqa
import torch 
from base import model 
from base import tokenizer
import copy 
import sys
import re
import torch.nn as nn 
from dataclasses import dataclass


class RL(nn.Module):
    @dataclass
    def __init__(self, model, tokenizer, prompt= "The capital of France is", target="Paris"):
        super().__init__()
        
        G, STEPS, MAX_NEW, EPS, BETA, LR,K = 8, 100, 8, 0.2, 0.05, 1e-6, 2

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

        PROMPT = self.prompt
        TARGET = self.target
        
        self.opt = torch.optim.AdamW(self.policy.parameters(), lr=LR)    
        self.tokenizer = tokenizer    

    @dataclass
    def reward(self, text):
        pass
                
    def token_logprobs(self, model, ids, prompt_len):
        logits = model(ids).logits[:, :-1].float()
        lp = torch.log_softmax(logits, -1).gather(-1, ids[:, 1:, None]).squeeze(-1)
        return lp[:, prompt_len - 1:] # log prob of each generated token 


    def train(self, ):
        device = next(self.policy.parameters()).device
        for step in range(self.STEPS):

            prompt_ids = self.tokenizer(self.PROMPT, return_tensors='pt').input_ids.repeat(G, 1)
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
            R = torch.tensor([self.reward(t) for t in texts], device=device)
            A = (R-R.mean()) # baselining
            
            for _ in range(self.K):
                new_lp = token_logprobs(self.policy, ids, prompt_length)
                ratio = torch.exp(new_lp - old_lp)
                clipped = torch.clamp(ratio , 1-EPS, 1+EPS)
                pg = torch.min(ratio * A[:, None] , A[:,None] * clipped)
                
                # KLD-Loss
                # kl = (new_lp - ref_lp) ( faulty one)
                kl = torch.exp(ref_lp - new_lp) - (ref_lp - new_lp) - 1
                
                # loss 
                loss = -(pg - BETA * kl)
                loss = (loss * mask).sum() / (G * MAX_NEW) 
                
                self.opt.zero_grad()
                loss.backward()

                # gradient norm
                total_norm = 0.0
                for p in self.policy.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2

                total_norm = total_norm**0.5
                print(f"Total Gradient Norm: {total_norm}")
                
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
                self.opt.step()

            if step % 10 == 0:
                print(f"step {step:3d}  mean_R={R.mean():.2f}  sample={texts}")


