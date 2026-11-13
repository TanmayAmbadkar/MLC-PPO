#!/usr/bin/env python3
"""Download and serialize the GSM8K training split for exact-match RLVR."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--dataset-name", default="openai/gsm8k")
    args = parser.parse_args()

    rows = load_dataset(args.dataset_name, "main", split=args.split)
    if args.max_examples:
        rows = rows.select(range(min(args.max_examples, len(rows))))
    output = Path(args.output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for row in rows:
            prompt = (
                "Solve the following math problem. Show your reasoning, then put the final "
                "answer in \\boxed{...}.\n\n"
                f"{row['question']}"
            )
            f.write(json.dumps({"prompt": prompt, "answer": row["answer"]}, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} examples to {output}")


if __name__ == "__main__":
    main()
