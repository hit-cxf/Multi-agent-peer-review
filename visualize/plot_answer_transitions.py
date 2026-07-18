#!/usr/bin/env python3
"""Draw answer-change pie charts from existing MAPR result files."""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np

PROJECT_DIR = Path(
    os.getenv("MAPR_PROJECT_DIR", "/Users/xinfanchen/Project/Multi-agent-peer-review")
).resolve()
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from eval import compute_accuracy, parse_pred_answer
from params import EXAMPLE_NUM


METHODS = {
    "self_correction": "Self-correct",
    "peer_review": "Peer review",
}
DEFAULT_TASKS = ("GSM8K", "StrategyQA")
CATEGORIES = (
    "No Change",
    "Correct → Incorrect",
    "Incorrect → Incorrect",
    "Incorrect → Correct",
)
# Match the supplied reference figure's Matplotlib categorical palette exactly.
COLORS = ("#1f77b4", "#d62728", "#2ca02c", "#ff7f0e")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, default=PROJECT_DIR / "result_qwen3_8b")
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--methods", nargs="+", choices=list(METHODS), default=list(METHODS))
    parser.add_argument("--time-flag", default="0713")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "pics")
    return parser.parse_args()


def answers_equal(left, right, task):
    if left is None or right is None:
        return left is right
    if task in {"GSM8K", "SVAMP", "MultiArith", "AddSub", "SingleEq"}:
        try:
            return float(left) == float(right)
        except (TypeError, ValueError):
            pass
    return str(left).strip().upper() == str(right).strip().upper()


def transition_counts(path, task):
    data = json.loads(path.read_text(encoding="utf-8"))
    eval_args = SimpleNamespace(task=task)
    counts = Counter()
    excluded = 0

    for row in data:
        contexts = row.get("agent_contexts")
        if not contexts:
            excluded += 1
            continue
        for context in contexts:
            if len(context) < 2:
                excluded += 1
                continue
            initial_text = context[1]["content"]
            final_text = context[-1]["content"]
            initial = parse_pred_answer(initial_text, eval_args)
            final = parse_pred_answer(final_text, eval_args)
            initial_correct = bool(compute_accuracy(row["answer"], initial_text, eval_args))
            final_correct = bool(compute_accuracy(row["answer"], final_text, eval_args))

            if answers_equal(initial, final, task):
                category = "No Change"
            elif initial_correct and not final_correct:
                category = "Correct → Incorrect"
            elif not initial_correct and final_correct:
                category = "Incorrect → Correct"
            elif not initial_correct and not final_correct:
                category = "Incorrect → Incorrect"
            else:
                category = "No Change"
            counts[category] += 1
    return counts, excluded


def plot_pie(counts, output):
    values = [counts[category] for category in CATEGORIES]
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    wedges, _ = ax.pie(
        values,
        colors=COLORS,
        startangle=22.7,
        counterclock=False,
        wedgeprops={"linewidth": 0},
    )

    total = sum(values)
    label_points = []
    small_percent_points = []
    for category, value, wedge in zip(CATEGORIES, values, wedges):
        if value == 0:
            continue
        share = value / total * 100 if total else 0
        angle = np.deg2rad((wedge.theta1 + wedge.theta2) / 2)

        # Keep the supplied layout while preventing tiny adjacent slices from
        # placing their percentages on top of one another.
        if share < 4:
            radius = 0.64
            small_percent_points.append(
                [f"{share:.1f}%", radius * np.cos(angle), radius * np.sin(angle)]
            )
        else:
            radius = 0.62
            ax.text(
                radius * np.cos(angle), radius * np.sin(angle), f"{share:.1f}%",
                ha="center", va="center", fontsize=10,
            )

        label_points.append([category, 1.08 * np.cos(angle), 1.08 * np.sin(angle)])

    small_percent_points.sort(key=lambda point: point[2])
    for index in range(1, len(small_percent_points)):
        small_percent_points[index][2] = max(
            small_percent_points[index][2], small_percent_points[index - 1][2] + 0.085
        )
    for value, x, y in small_percent_points:
        ax.text(x, y, value, ha="center", va="center", fontsize=10)

    # Resolve collisions among exterior labels independently on each side.
    for side in (-1, 1):
        points = [point for point in label_points if np.sign(point[1]) == side]
        points.sort(key=lambda point: point[2])
        for index in range(1, len(points)):
            points[index][2] = max(points[index][2], points[index - 1][2] + 0.105)
        for category, x, y in points:
            ax.text(x, y, category, ha="left" if side > 0 else "right", va="center", fontsize=10)

    ax.axis("equal")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def print_plot_data(task, method, counts, excluded):
    total = sum(counts.values())
    print(f"\n[PLOT DATA] task={task}\tmethod={METHODS[method]}\ttotal={total}\texcluded={excluded}")
    print("category\tcount\tshare_pct")
    for category in CATEGORIES:
        count = counts[category]
        share = count / total * 100 if total else 0
        print(f"{category}\t{count}\t{share:.4f}")


def main():
    args = parse_args()
    for task in args.tasks:
        if task not in EXAMPLE_NUM:
            raise ValueError(f"Unknown task: {task}")
        for method in args.methods:
            source = (
                args.result_dir / task
                / f"{task}_{method}_{EXAMPLE_NUM[task]}_{args.time_flag}.json"
            )
            if not source.exists():
                raise FileNotFoundError(source)
            counts, excluded = transition_counts(source, task)
            print_plot_data(task, method, counts, excluded)
            output = args.output_dir / f"answer_change_{task}_{method}.png"
            plot_pie(counts, output)


if __name__ == "__main__":
    main()
