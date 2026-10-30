"""Build the four public prompt samples for first-hour check 6. Runs on the rented machine.

    python data/build_first_hour_public.py <output dir>

All four sources are public, so no Hugging Face token is needed. Each sample is 100
prompts chosen with a fixed seed. Output: one JSONL file per kind of text, one
OpenAI-style request per line ({"id", "messages", "thinking"}).

The fifth kind, tool calls, comes from a gated dataset and is built on the laptop
(build_first_hour_toolcalls.py).
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

from datasets import load_dataset

SAMPLES = 100
SEED = 0


def sample(rows: list, count: int = SAMPLES) -> list:
    return random.Random(SEED).sample(rows, count)


def write(path: Path, requests: list[dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as out:
        for request in requests:
            out.write(json.dumps(request, ensure_ascii=False) + "\n")
    print(f"{path.name}: {len(requests)} prompts")


def user(content: str) -> list[dict]:
    return [{"role": "user", "content": content}]


def main() -> None:
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Code: HumanEval function stubs. The model writes the body.
    rows = list(load_dataset("openai/openai_humaneval", split="test"))
    write(out_dir / "code.jsonl", [
        {"id": r["task_id"], "thinking": False,
         "messages": user(f"Complete the following Python function.\n\n```python\n{r['prompt']}```")}
        for r in sample(rows)
    ])

    # General chat: Alpaca instructions that need no extra input.
    rows = [r for r in load_dataset("tatsu-lab/alpaca", split="train") if not r["input"].strip()]
    write(out_dir / "chat.jsonl", [
        {"id": f"alpaca-{i}", "thinking": False, "messages": user(r["instruction"])}
        for i, r in enumerate(sample(rows))
    ])

    # Long reasoning: GSM8K word problems with the model's thinking mode switched on,
    # so the output is a long chain of reasoning before the answer.
    rows = list(load_dataset("openai/gsm8k", "main", split="test"))
    write(out_dir / "reasoning.jsonl", [
        {"id": f"gsm8k-{i}", "thinking": True, "messages": user(r["question"])}
        for i, r in enumerate(sample(rows))
    ])

    # Non-English: Chinese instructions. Two candidate sources, first that loads wins,
    # and the one used is recorded in the ids.
    for repo, field in [("shibing624/alpaca-zh", "instruction"),
                        ("silk-road/alpaca-data-gpt4-chinese", "instruction_zh")]:
        try:
            rows = [r for r in load_dataset(repo, split="train") if r.get(field) and not (r.get("input") or r.get("input_zh") or "").strip()]
        except Exception as err:  # noqa: BLE001 - any load failure means try the next source
            print(f"could not load {repo}: {type(err).__name__}")
            continue
        write(out_dir / "chinese.jsonl", [
            {"id": f"{repo}-{i}", "thinking": False, "messages": user(r[field])}
            for i, r in enumerate(sample(rows))
        ])
        break
    else:
        sys.exit("no Chinese source could be loaded")


if __name__ == "__main__":
    main()
