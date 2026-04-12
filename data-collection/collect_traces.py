"""
Collect reasoning traces from DeepSeek-R1 on benchmark datasets.
Saves both raw reasoning (CoT) and final answer for each problem.

Setup:
    export TOGETHER_API_KEY="your-key-here"

Usage:
    python collect_traces.py --benchmark math500 --n 5       # first 5 MATH-500 problems
    python collect_traces.py --benchmark aime24 --n 30       # all 30 AIME 2024 problems
    python collect_traces.py --benchmark gpqa_diamond --n 10  # first 10 GPQA Diamond
"""

import argparse
import json
import os
import time
from pathlib import Path

from datasets import load_dataset
from together import Together


MODEL = "deepseek-ai/DeepSeek-R1"
MAX_TOKENS = 16384  # reasoning models can produce long CoT
OUTPUT_DIR = Path(__file__).parent / "traces"
REQUEST_TIMEOUT = 600.0  # 10 min — AIME problems can take a while
MAX_RETRIES = 2


def load_benchmark(name: str):
    """Load benchmark dataset, return list of {id, problem, answer}."""
    if name == "math500":
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        return [
            {
                "id": row["unique_id"],
                "problem": row["problem"],
                "answer": row["answer"],
                "subject": row["subject"],
                "level": row["level"],
            }
            for row in ds
        ]
    elif name == "aime24":
        ds = load_dataset("HuggingFaceH4/aime_2024", split="train")
        return [
            {
                "id": row["id"],
                "problem": row["problem"],
                "answer": row["answer"],
            }
            for row in ds
        ]
    elif name == "gpqa_diamond":
        ds = load_dataset("fingertap/GPQA-Diamond", split="test")
        return [
            {
                "id": f"gpqa_{i}",
                "problem": row["question"],
                "answer": row["answer"],
            }
            for i, row in enumerate(ds)
        ]
    else:
        raise ValueError(f"Unknown benchmark: {name}")


def collect_trace(client: Together, problem: str) -> dict:
    """Send a problem to DeepSeek-R1, return reasoning + answer + usage."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": f"Solve the following problem. Show your work.\n\n{problem}",
            }
        ],
        max_tokens=MAX_TOKENS,
    )
    msg = response.choices[0].message
    usage = response.usage
    return {
        "reasoning": msg.reasoning or "",
        "content": msg.content or "",
        "completion_tokens": usage.completion_tokens,
        "prompt_tokens": usage.prompt_tokens,
        "total_tokens": usage.total_tokens,
    }


def main():
    parser = argparse.ArgumentParser(description="Collect reasoning traces")
    parser.add_argument(
        "--benchmark",
        required=True,
        choices=["math500", "aime24", "gpqa_diamond"],
    )
    parser.add_argument("--n", type=int, default=None, help="Number of problems (default: all)")
    parser.add_argument("--offset", type=int, default=0, help="Start from this index")
    args = parser.parse_args()

    client = Together(timeout=REQUEST_TIMEOUT)
    problems = load_benchmark(args.benchmark)
    if args.n is not None:
        problems = problems[args.offset : args.offset + args.n]
    else:
        problems = problems[args.offset :]

    out_dir = OUTPUT_DIR / args.benchmark
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Collecting {len(problems)} traces for {args.benchmark} using {MODEL}")
    print(f"Output: {out_dir}/")
    print()

    for i, prob in enumerate(problems):
        clean_id = str(prob["id"]).replace("/", "_").removesuffix(".json")
        out_file = out_dir / f"{clean_id}.json"

        # Skip if already collected
        if out_file.exists():
            print(f"[{i+1}/{len(problems)}] {prob['id']} — already exists, skipping")
            continue

        print(f"[{i+1}/{len(problems)}] {prob['id']}...", end=" ", flush=True)
        t0 = time.time()

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                trace = collect_trace(client, prob["problem"])
                result = {**prob, **trace, "model": MODEL}

                with open(out_file, "w") as f:
                    json.dump(result, f, indent=2)

                elapsed = time.time() - t0
                reasoning_len = len(trace["reasoning"])
                print(
                    f"done ({elapsed:.1f}s, "
                    f"{trace['completion_tokens']} tokens, "
                    f"{reasoning_len} chars reasoning)"
                )
                break
            except Exception as e:
                if attempt < MAX_RETRIES:
                    print(f"retry {attempt}/{MAX_RETRIES} ({e})...", end=" ", flush=True)
                else:
                    print(f"FAILED after {MAX_RETRIES} attempts: {e}")

    print("\nDone!")


if __name__ == "__main__":
    main()
