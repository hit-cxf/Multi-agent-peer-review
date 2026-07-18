#!/usr/bin/env python3
"""Mark provider-rejected samples in Figure 4 result copies only."""

import argparse
import json
import os
import tempfile
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import sample_filter  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-root",
        type=Path,
        default=PROJECT_DIR / "result_agent_num_review_rounds",
    )
    parser.add_argument("--task", default="StrategyQA")
    parser.add_argument("--index", type=int, default=385, help="1-based dataset index")
    parser.add_argument("--time-flag", help="Only modify files with this timestamp suffix")
    parser.add_argument("--reason", default="DashScope data_inspection_failed")
    return parser.parse_args()


def atomic_write(path, data):
    descriptor, temporary_name = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False)
        os.replace(temporary_name, path)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def main():
    args = parse_args()
    if args.index < 1:
        raise ValueError("--index must be positive")

    pattern = f"{args.task}_*_{args.time_flag}.json" if args.time_flag else "*.json"
    paths = sorted(
        path for path in args.result_root.rglob(pattern)
        if path.parent.name == args.task
    )
    if not paths:
        raise FileNotFoundError(f"No {args.task} result files found under {args.result_root}")

    changed = 0
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        if len(data) < args.index:
            print(f"SKIP incomplete ({len(data)} rows): {path}")
            continue
        original = data[args.index - 1]
        data[args.index - 1] = sample_filter.excluded_record(
            original, args.task, args.index
        )
        data[args.index - 1]["exclusion_reason"] = args.reason
        atomic_write(path, data)
        changed += 1
        print(f"MARKED: {path}")
    print(f"Marked {changed} result file(s) under {args.result_root}.")


if __name__ == "__main__":
    main()
