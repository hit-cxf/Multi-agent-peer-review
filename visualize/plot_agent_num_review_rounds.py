#!/usr/bin/env python3
"""Plot Figure 4 scaling curves from supplementary MAPR result files."""

import argparse
import json
import math
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from eval import compute_accuracy  # noqa: E402


TASKS = ("GSM8K", "AQuA", "StrategyQA")
TASK_SIZES = {"GSM8K": 500, "AQuA": 254, "StrategyQA": 500}
TASK_STYLE = {
    "GSM8K": {"color": "#2563EB", "marker": "o"},
    "AQuA": {"color": "#E88C30", "marker": "s"},
    "StrategyQA": {"color": "#2A9D6F", "marker": "^"},
}
CONFIG_RE = re.compile(r"agent_num_(\d+)_review_rounds_(\d+)$")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot agent-number and review-round accuracy curves."
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=PROJECT_DIR / "result_agent_num_review_rounds",
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "pics")
    parser.add_argument(
        "--time-flag",
        help="Use only files with this MMDD suffix. By default, use the latest file per configuration.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Draw available points instead of failing when a configuration is missing.",
    )
    parser.add_argument("--no-value-labels", action="store_true")
    return parser.parse_args()


def select_result_file(dataset_dir, task, size, time_flag):
    if time_flag:
        path = dataset_dir / f"{task}_peer_review_{size}_{time_flag}.json"
        return path if path.exists() else None

    candidates = list(dataset_dir.glob(f"{task}_peer_review_{size}_*.json"))
    if not candidates:
        return None

    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def majority_accuracy(path, task, expected_size):
    rows = json.loads(path.read_text(encoding="utf-8"))
    if len(rows) != expected_size:
        raise ValueError(f"Incomplete result: {path} has {len(rows)} rows, expected {expected_size}")
    args = SimpleNamespace(task=task)
    scores = []
    excluded = 0
    for row in rows:
        if row.get("excluded") or not row.get("agent_contexts"):
            excluded += 1
            continue
        final_solutions = [context[-1]["content"] for context in row["agent_contexts"]]
        scores.append(compute_accuracy(row["answer"], final_solutions, args))
    if not scores:
        raise ValueError(f"No evaluable rows in {path}")
    return sum(scores) / len(scores) * 100, len(scores), excluded


def collect_results(result_root, time_flag):
    values = {}
    files = {}
    for config_dir in sorted(result_root.glob("agent_num_*_review_rounds_*")):
        match = CONFIG_RE.fullmatch(config_dir.name)
        if not match:
            continue
        agent_num, review_rounds = map(int, match.groups())
        for task in TASKS:
            path = select_result_file(
                config_dir / task, task, TASK_SIZES[task], time_flag
            )
            if path is None:
                continue
            accuracy, valid_count, excluded_count = majority_accuracy(
                path, task, TASK_SIZES[task]
            )
            key = (agent_num, review_rounds, task)
            values[key] = accuracy
            files[key] = (path, valid_count, excluded_count)
    return values, files


def required_keys():
    keys = set()
    for task in TASKS:
        keys.update((agent_num, 1, task) for agent_num in (2, 3, 4, 5))
        keys.update((3, review_rounds, task) for review_rounds in (1, 2, 3, 4))
    return keys


def shared_y_limits(values):
    scores = list(values.values())
    low = math.floor((min(scores) - 2) / 5) * 5
    high = math.ceil((max(scores) + 2) / 5) * 5
    return max(0, low), min(100, high)


def draw_curve(values, varying, output, y_limits, show_value_labels):
    if varying == "agent_num":
        x_values = (2, 3, 4, 5)
        xlabel = "Number of agents"
        lookup = lambda x, task: (x, 1, task)
    else:
        x_values = (1, 2, 3, 4)
        xlabel = "Number of review rounds"
        lookup = lambda x, task: (3, x, task)

    fig, ax = plt.subplots(figsize=(6.2, 4.1))
    for task in TASKS:
        points = [(x, values[lookup(x, task)]) for x in x_values if lookup(x, task) in values]
        if not points:
            continue
        xs, ys = zip(*points)
        style = TASK_STYLE[task]
        ax.plot(
            xs,
            ys,
            label=task,
            color=style["color"],
            marker=style["marker"],
            linewidth=1.8,
            markersize=6,
        )
        if show_value_labels:
            for x, y in points:
                ax.annotate(
                    f"{y:.1f}",
                    (x, y),
                    xytext=(0, 7),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8.5,
                    color=style["color"],
                )

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(x_values)
    ax.set_xlim(min(x_values) - 0.15, max(x_values) + 0.15)
    ax.set_ylim(*y_limits)
    ax.grid(axis="y", color="#E2E8F0", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.01))

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def print_plot_data(values, files):
    print("\n[PLOT DATA] Agent-number curve (review_rounds=1)")
    print("task\tagent_num\treview_rounds\taccuracy_pct\tn\texcluded\tfile")
    for task in TASKS:
        for agent_num in (2, 3, 4, 5):
            key = (agent_num, 1, task)
            if key not in values:
                continue
            path, valid_count, excluded_count = files[key]
            print(
                f"{task}\t{agent_num}\t1\t{values[key]:.4f}\t{valid_count}\t"
                f"{excluded_count}\t{path}"
            )

    print("\n[PLOT DATA] Review-round curve (agent_num=3)")
    print("task\tagent_num\treview_rounds\taccuracy_pct\tn\texcluded\tfile")
    for task in TASKS:
        for review_rounds in (1, 2, 3, 4):
            key = (3, review_rounds, task)
            if key not in values:
                continue
            path, valid_count, excluded_count = files[key]
            print(
                f"{task}\t3\t{review_rounds}\t{values[key]:.4f}\t{valid_count}\t"
                f"{excluded_count}\t{path}"
            )


def main():
    args = parse_args()
    values, files = collect_results(args.result_root, args.time_flag)
    if not values:
        raise RuntimeError(f"No result files found under {args.result_root}")

    missing = sorted(required_keys() - set(values))
    if missing and not args.allow_missing:
        formatted = "\n".join(
            f"  agent_num={agent}, review_rounds={rounds}, task={task}"
            for agent, rounds, task in missing
        )
        raise RuntimeError(
            "Figure 4 results are incomplete. Missing configurations:\n"
            f"{formatted}\nUse --allow-missing only for a partial preview."
        )

    y_limits = shared_y_limits(values)
    print_plot_data(values, files)
    draw_curve(
        values,
        "agent_num",
        args.output_dir / "figure4_agent_num.png",
        y_limits,
        not args.no_value_labels,
    )
    draw_curve(
        values,
        "review_rounds",
        args.output_dir / "figure4_review_rounds.png",
        y_limits,
        not args.no_value_labels,
    )

    print(f"\n[OUTPUT] {args.output_dir / 'figure4_agent_num.png'}")
    print(f"[OUTPUT] {args.output_dir / 'figure4_agent_num.pdf'}")
    print(f"[OUTPUT] {args.output_dir / 'figure4_review_rounds.png'}")
    print(f"[OUTPUT] {args.output_dir / 'figure4_review_rounds.pdf'}")


if __name__ == "__main__":
    main()
