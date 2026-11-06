import json
import os
import sys

from inference import generate, load

THINKING = "--thinking" in sys.argv
OUT = f"/tmp/bitty-rl-eval/results{'_thinking' if THINKING else ''}.jsonl"
MAX_NEW_TOKENS = 8192 if THINKING else 64
INSTRUCTION = "Reply with only the decoded text, nothing else."

EXAMPLES = [
    {"level": 1, "prompt": "hi→ij, ok→pl. Decode: ifmmp", "answer": "hello"},
    {"level": 2, "prompt": "cat→fdw, dog→grj. Decode: elug", "answer": "bird"},
    {"level": 3, "prompt": "sun→xzs, map→rfu. Decode: kwtl", "answer": "frog"},
    {"level": 4, "prompt": "cat→tac, moon→noom. Decode: elppa", "answer": "apple"},
    {"level": 5, "prompt": "cat→ubd, dog→hpe. Decode: lppc", "answer": "book"},
    {"level": 6, "prompt": "cat→mqz, her→wxb. Decode: mqbx", "answer": "care"},
    {"level": 7, "prompt": "abc→zyx, hi→sr, fox→ulc. Decode: dliow", "answer": "world"},
    {"level": 8, "prompt": "blarn zif→dog runs, pom zif→cat runs, pom kesh→cat eats. Decode: blarn kesh", "answer": "dog eats"},
    {"level": 9, "prompt": "zif blarn→dog runs, kesh pom→cat eats, zif pom→cat runs. Decode: kesh blarn", "answer": "dog eats"},
    {"level": 10, "prompt": "aaa→bcd, dog→eqj, cat→dcw. Decode: igopt", "answer": "hello"},
]


def normalize(s):
    return s.strip().strip(".!\"'`").strip().lower()


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    done = set()
    if os.path.exists(OUT):
        with open(OUT) as f:
            done = {json.loads(line)["level"] for line in f}

    tokenizer, model = load()
    with open(OUT, "a") as f:
        for ex in EXAMPLES:
            if ex["level"] in done:
                continue
            reasoning, output = generate(
                tokenizer, model, f"{ex['prompt']}\n{INSTRUCTION}", MAX_NEW_TOKENS, THINKING
            )
            reward = int(normalize(output) == ex["answer"])
            f.write(json.dumps({**ex, "reasoning": reasoning, "output": output, "reward": reward}) + "\n")
            f.flush()

    with open(OUT) as f:
        rows = sorted((json.loads(line) for line in f), key=lambda r: r["level"])
    for r in rows:
        print(f"L{r['level']:>2} {'PASS' if r['reward'] else 'FAIL'}  expected={r['answer']!r}  got={r['output']!r}")
    print(f"pass rate: {sum(r['reward'] for r in rows)}/{len(rows)}")


if __name__ == "__main__":
    main()
