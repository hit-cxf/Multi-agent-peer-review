#!/usr/bin/env python3
"""Compute the exact metrics reported in paper Table 5."""

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from eval import compute_accuracy, parse_pred_answer  # noqa: E402
import sample_filter  # noqa: E402


DEFAULT_PAIRS = [
    ("qwen3-8b", "qwen3-14b"),
    ("qwen3-8b", "llama3-8b"),
    ("qwen3-14b", "llama3-8b"),
]
DEFAULT_TASKS = ["GSM8K", "StrategyQA"]


def slug(value):
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate heterogeneous Table 5 runs")
    parser.add_argument(
        "--result-dir", type=Path, default=PROJECT_DIR / "result_heterogeneous_llm"
    )
    parser.add_argument("--time-flag", required=True)
    parser.add_argument("--max-example-num", type=int, default=500)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def result_path(args, task, pair):
    pair_slug = "__".join(slug(model) for model in pair)
    return (
        args.result_dir
        / task
        / pair_slug
        / f"{task}_heterogeneous_peer_review_{args.max_example_num}_{args.time_flag}.json"
    )


def percentage(values):
    return sum(values) / len(values) * 100


def evaluate(args):
    table_rows = []

    for task in DEFAULT_TASKS:
        task_args = SimpleNamespace(task=task)
        model_initials = {}
        for pair in DEFAULT_PAIRS:
            path = result_path(args, task, pair)
            if not path.exists():
                raise FileNotFoundError(path)
            raw_rows = json.loads(path.read_text(encoding="utf-8"))
            if len(raw_rows) != args.max_example_num:
                raise ValueError(
                    f"{path} has {len(raw_rows)} rows, expected {args.max_example_num}"
                )
            rows, excluded = sample_filter.filter_evaluation_rows(raw_rows)
            if not rows:
                raise ValueError(f"no evaluable rows in {path}")

            initial_predictions = [[], []]
            initial_correct = [[], []]
            updated_correct = [[], []]
            for row in rows:
                if row.get("agent_models") != list(pair):
                    raise ValueError(
                        f"model order mismatch at {path}, row {row.get('dataset_index')}"
                    )
                for agent_index, model in enumerate(pair):
                    key = (model, row["dataset_index"])
                    initial_response = row["initial_responses"][agent_index]
                    previous = model_initials.setdefault(key, initial_response)
                    if previous != initial_response:
                        raise ValueError(
                            f"shared initial response mismatch: model={model}, "
                            f"sample={row['dataset_index']}"
                        )
                    initial_predictions[agent_index].append(
                        parse_pred_answer(initial_response, task_args)
                    )
                    initial_correct[agent_index].append(
                        compute_accuracy(row["answer"], initial_response, task_args)
                    )
                    updated_correct[agent_index].append(
                        compute_accuracy(
                            row["answer"],
                            row["revised_responses"][agent_index],
                            task_args,
                        )
                    )

            diversity = percentage(
                [
                    prediction_a != prediction_b
                    for prediction_a, prediction_b in zip(*initial_predictions)
                ]
            )
            initial_accuracies = [percentage(values) for values in initial_correct]
            updated_accuracies = [percentage(values) for values in updated_correct]
            capability_gap = abs(initial_accuracies[0] - initial_accuracies[1])

            for index, model in enumerate(pair):
                table_rows.append(
                    {
                        "peer_review": f"{pair[0]} & {pair[1]}",
                        "task": task,
                        "capability_gap": capability_gap,
                        "diversity_incon": diversity,
                        "llm": model,
                        "initial_acc": initial_accuracies[index],
                        "updated_acc": updated_accuracies[index],
                        "delta": updated_accuracies[index] - initial_accuracies[index],
                        "samples": len(rows),
                        "excluded": len(excluded),
                        "source": str(path),
                    }
                )
    return table_rows


def print_table(rows):
    header = (
        f"{'Task':<11} {'Peer Review':<34} {'Gap':>6} {'INCON':>7} {'LLM':<16} "
        f"{'Initial':>8} {'Updated':>8} {'Delta':>8}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['task']:<11} {row['peer_review']:<34} {row['capability_gap']:>6.1f} "
            f"{row['diversity_incon']:>7.1f} {row['llm']:<16} "
            f"{row['initial_acc']:>8.1f} {row['updated_acc']:>8.1f} "
            f"{row['delta']:>+8.1f}"
        )


def main():
    args = parse_args()
    rows = evaluate(args)
    print_table(rows)
    output = args.output or (
        args.result_dir / f"table5_heterogeneous_{args.time_flag}.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    output.with_suffix(".json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"CSV: {output}")
    print(f"JSON: {output.with_suffix('.json')}")


if __name__ == "__main__":
    main()
