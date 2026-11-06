#ruff:noqa
import json
import torch
import torch.nn as nn

from rl_algo.grpo import GRPO
from data.io import get_loader, single_example, extract_answer, SYSTEM_PROMPT
from transformers import AutoTokenizer, AutoModelForCausalLM
from inference import generate
from logger import logger

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, device_map="mps")

# first check if RL can be used or not here
qa = single_example()
question, answer = qa['question'][0] , qa['answer'][0]
k, K = 0, 32
for i in range(K):
    reasoning , output = generate(tokenizer, model, question, SYSTEM_PROMPT, thinking=True)
    logger.info(f'input: {question}')
    logger.info(f"output: {output}")

    if extract_answer(reasoning + output) == answer:
       k+=1

if k > 0:
    logger.info(f"base model solves it sometimes: {k/K}")
    # grpo implementation and trainer
    trainer = GRPO(model = model , tokenizer = tokenizer, dataloader = get_loader())
    trainer.train()
else:
    logger.info(f"The base model cant predict the answer : {k/K} , so we first need to do SFT over it first")

