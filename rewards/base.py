import json
import re
from abc import ABC, abstractmethod

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)

NO_MATCH, NAME_OK, KEYS_OK, VALUES_OK = 0, 1, 2, 3


def parse_tool_calls(text):
    calls = []
    for blob in TOOL_CALL_RE.findall(text):
        try:
            call = json.loads(blob)
        except json.JSONDecodeError:
            return None
        if not isinstance(call, dict) or "name" not in call or not isinstance(call.get("arguments", {}), dict):
            return None
        calls.append((call["name"], call.get("arguments", {})))
    return calls


def normalize(v):
    if isinstance(v, str):
        return re.sub(r"[ ,./\-_*^]", "", v.lower())
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, list):
        return [normalize(x) for x in v]
    if isinstance(v, dict):
        return {k: normalize(x) for k, x in v.items()}
    return v


def call_stage(name, args, expected):
    exp_name, exp_args = next(iter(expected.items()))
    if name != exp_name:
        return NO_MATCH
    required = {k for k, allowed in exp_args.items() if "" not in allowed}
    if set(args) - set(exp_args) or required - set(args):
        return NAME_OK
    for param, value in args.items():
        if normalize(value) not in [normalize(a) for a in exp_args[param] if a != ""]:
            return KEYS_OK
    return VALUES_OK


def match_stages(calls, expected):
    remaining = list(expected)
    stages = []
    for name, args in calls:
        best = max(range(len(remaining)), key=lambda i: call_stage(name, args, remaining[i]), default=None)
        if best is None:
            stages.append(NO_MATCH)
            continue
        stages.append(call_stage(name, args, remaining[best]))
        if stages[-1] > NO_MATCH:
            remaining.pop(best)
    return stages + [NO_MATCH] * len(remaining)


class Reward(ABC):
    def __init__(self, log_path=None):
        self.log_path = log_path
        self.__name__ = type(self).__name__

    def __call__(self, prompts, completions, ground_truth, id, **kwargs):
        texts = [c[0]["content"] if isinstance(c, list) else c for c in completions]
        scores = [self.score(t, json.loads(g)) for t, g in zip(texts, ground_truth)]
        self.log(id, texts, scores)
        return scores

    def score(self, text, expected):
        calls = parse_tool_calls(text)
        if not calls:
            return 0.0
        return self.score_stages(match_stages(calls, expected))

    def log(self, ids, texts, scores):
        if not self.log_path:
            return
        with open(self.log_path, "a") as f:
            f.writelines(
                json.dumps({"reward": self.__name__, "id": i, "score": s, "output": t}) + "\n"
                for i, t, s in zip(ids, texts, scores)
            )

    @abstractmethod
    def score_stages(self, stages):
        ...
