#ruff : noqa
# https://huggingface.co/datasets/openai/gsm8k

import re

from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

DATA_CARD = 'openai/gsm8k'


class GSM(Dataset):
    def __init__(self, split='train', data_card=DATA_CARD):
        self.data = load_dataset(data_card, 'main', split=split)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        answer = row['answer'].split('####')[-1].strip().replace(',', '')
        return {'question': row['question'], 'answer': answer}


def get_loader(split='train', batch_size=1, shuffle=True):
    return DataLoader(GSM(split), batch_size=batch_size, shuffle=shuffle)


def single_example(split = 'test', batch_size =1):
    return next(iter(get_loader(split, batch_size=batch_size, shuffle=False)))

def extract_answer(text):
    m = re.search(r'<answer>(.*?)</answer>', text, re.S)
    if not m:
        return None
    nums = re.findall(r'-?\d+(?:\.\d+)?', m.group(1).replace(',', ''))
    return nums[-1] if nums else None


SYSTEM_PROMPT = """You are a helpful assistant that solves grade school math word problems.
Think step by step inside <think></think>, then give only the final numeric answer inside <answer></answer>.
Do not include units, commas or extra text in the answer.

Example:
<think>Tom has 3 apples and buys 2 more. 3 + 2 = 5.</think>
<answer>5</answer>"""

if __name__ == '__main__':
    batch = next(iter(get_loader(batch_size=2)))
    qa = single_example()
    print(qa['question'][0])
    print(qa['answer'][0])
    # print(batch)
