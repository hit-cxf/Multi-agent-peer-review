#!/usr/bin/env python3
"""Plot initial-to-final answer transitions from existing MAPR results."""

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


METHODS = [
    ("self_correction", "Self-correct"),
    ("peer_review", "Ours"),
    ("feedback", "Ours (w/o confidence)"),
    ("ablation_solution", "Ours (w/o solution)"),
]
CATEGORIES = [
    "No Change",
    "Correct → Incorrect",
    "Incorrect → Correct",
    "Incorrect → Incorrect",
]
COLORS = {
    "No Change": "#CBD5E1",
    "Correct → Incorrect": "#D95F59",
    "Incorrect → Correct": "#2A9D6F",
    "Incorrect → Incorrect": "#E9A23B",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, default=PROJECT_DIR / "result_qwen3_8b")
    parser.add_argument("--task", default="GSM8K", choices=list(EXAMPLE_NUM))
    parser.add_argument("--time-flag", default="0713")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def answers_equal(left, right, task):
    """Compare parsed answers semantically instead of comparing raw text."""
    if left is None or right is None:
        return left is right
    if task in {"GSM8K", "SVAMP", "MultiArith", "AddSub", "SingleEq"}:
        try:
            return float(left) == float(right)
        except (TypeError, ValueError):
            return str(left).strip() == str(right).strip()
    return str(left).strip().upper() == str(right).strip().upper()


def transition_counts(path, task):
    data = json.loads(path.read_text(encoding="utf-8"))
    eval_args = SimpleNamespace(task=task)
    counts = Counter()

    for row in data:
        for context in row["agent_contexts"]:
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
                # Both predictions are correct but differ only in representation.
                category = "No Change"
            counts[category] += 1
    return counts


def add_labels(ax, values, lefts, y, total, minimum_width=0.7):
    for value, left in zip(values, lefts):
        width = value / total * 100 if total else 0
        if width >= minimum_width:
            ax.text(left + width / 2, y, f"{width:.1f}", ha="center", va="center", fontsize=9)


def plot(rows, output, task):
    labels = [label for _, label, _ in rows]
    y = np.arange(len(rows))
    fig, (ax_all, ax_changed) = plt.subplots(1, 2, figsize=(12.6, 4.8))
    fig.subplots_adjust(left=0.19, right=0.96, top=0.94, bottom=0.25, wspace=0.06)

    left = np.zeros(len(rows))
    for category in CATEGORIES:
        values = np.array([counts[category] / sum(counts.values()) * 100 for _, _, counts in rows])
        ax_all.barh(y, values, left=left, color=COLORS[category], height=0.55, label=category)
        for row_index, value in enumerate(values):
            # Small transition segments are expanded and labelled precisely in
            # the right panel; labelling them here would create collisions.
            if value >= 4:
                ax_all.text(left[row_index] + value / 2, row_index, f"{value:.1f}", ha="center", va="center", fontsize=9)
        left += values

    changed_categories = CATEGORIES[1:]
    left = np.zeros(len(rows))
    for category in changed_categories:
        values = []
        for _, _, counts in rows:
            changed = sum(counts[item] for item in changed_categories)
            values.append(counts[category] / changed * 100 if changed else 0)
        values = np.array(values)
        ax_changed.barh(y, values, left=left, color=COLORS[category], height=0.55)
        for row_index, value in enumerate(values):
            if value >= 5:
                ax_changed.text(left[row_index] + value / 2, row_index, f"{value:.1f}", ha="center", va="center", fontsize=9)
        left += values

    for ax in (ax_all, ax_changed):
        ax.set_xlim(0, 100)
        ax.set_xticks(np.arange(0, 101, 20))
        ax.set_xlabel("Share of answers (%)")
        ax.grid(axis="x", color="#E2E8F0", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.invert_yaxis()

    ax_all.set_yticks(y, labels)
    ax_changed.set_yticks(y, [""] * len(labels))
    changed_totals = [sum(c[item] for item in changed_categories) for _, _, c in rows]
    for index, changed in enumerate(changed_totals):
        ax_changed.text(101.5, index, f"n={changed}", va="center", fontsize=9, color="#475569", clip_on=False)

    handles, legend_labels = ax_all.get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.04))

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    args = parse_args()
    output = args.output or PROJECT_DIR / "pics" / f"answer_transitions_{args.task}_{args.time_flag}.png"
    rows = []
    for method, label in METHODS:
        path = args.result_dir / args.task / f"{args.task}_{method}_{EXAMPLE_NUM[args.task]}_{args.time_flag}.json"
        if not path.exists():
            raise FileNotFoundError(path)
        rows.append((method, label, transition_counts(path, args.task)))
    plot(rows, output, args.task)
    print(output)


if __name__ == "__main__":
    main()
