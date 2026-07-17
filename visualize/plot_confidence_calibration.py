#!/usr/bin/env python3
"""Create Figure 5 confidence-distribution and calibration plots."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from style import COLORS, CONFIDENCE_FIGURE_COLORS


PROJECT_DIR = Path(__file__).resolve().parents[1]
TASKS = [
    ("GSM8K", "GSM8K"),
    ("SVAMP", "SVAMP"),
    ("AQuA", "AQuA"),
    ("MultiArith", "MultiArith"),
    ("AddSub", "AddSub"),
    ("SingleEq", "SingleEq"),
    ("ARC-c", "ARC-c"),
    ("StrategyQA", "StrategyQA"),
    ("Colored_Objects", "Colored Objects"),
    ("Penguins", "Penguins"),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Plot Figure 5 confidence calibration.")
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=PROJECT_DIR / "result_figure5_llm_judge",
    )
    parser.add_argument("--time-flag", default="0713")
    parser.add_argument("--sample-size", type=int, default=600)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "pics")
    return parser.parse_args()


def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def auroc(labels, scores):
    """AUROC from pairwise ranking, including half credit for tied scores."""
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    if len(positive) == 0 or len(negative) == 0:
        return float("nan")
    comparisons = positive[:, None] - negative[None, :]
    return float((np.sum(comparisons > 0) + 0.5 * np.sum(comparisons == 0)) / comparisons.size)


def expected_calibration_error(labels, scores):
    total = len(labels)
    value = 0.0
    for confidence in range(1, 11):
        mask = scores == confidence / 10
        count = int(mask.sum())
        if count:
            value += count / total * abs(float(labels[mask].mean()) - confidence / 10)
    return value


def load_metrics(result_dir, time_flag, sample_size):
    datasets = []
    for task, label in TASKS:
        path = (
            result_dir
            / task
            / f"{task}_feedback_judgments_{sample_size}_{time_flag}.jsonl"
        )
        if not path.exists():
            raise FileNotFoundError(path)
        rows = read_jsonl(path)
        ids = [row["id"] for row in rows]
        if len(rows) != sample_size:
            raise ValueError(f"{path} contains {len(rows)} rows; expected {sample_size}")
        if len(set(ids)) != len(ids):
            raise ValueError(f"{path} contains duplicate feedback IDs")

        labels = np.asarray([int(row["feedback_correct"]) for row in rows], dtype=int)
        confidence_int = np.asarray(
            [int(row["verbalized_confidence"]) for row in rows], dtype=int
        )
        if np.any((confidence_int < 1) | (confidence_int > 10)):
            raise ValueError(f"{path} contains confidence outside 1-10")
        scores = confidence_int / 10
        counts = np.bincount(confidence_int, minlength=11)[1:11]
        bin_accuracy = np.full(10, np.nan)
        for confidence in range(1, 11):
            mask = confidence_int == confidence
            if mask.any():
                bin_accuracy[confidence - 1] = labels[mask].mean()

        accuracy = float(labels.mean())
        mean_confidence = float(scores.mean())
        datasets.append(
            {
                "task": task,
                "label": label,
                "n": len(rows),
                "judge_model": rows[0].get("judge_model", ""),
                "accuracy": accuracy,
                "mean_confidence": mean_confidence,
                "calibration_gap": mean_confidence - accuracy,
                "auroc": auroc(labels, scores),
                "ece": expected_calibration_error(labels, scores),
                "counts": counts,
                "shares": counts / len(rows) * 100,
                "bin_accuracy": bin_accuracy,
            }
        )
    return datasets


def add_group_separators(axis):
    for position in (5.5, 7.5):
        axis.axhline(position, color=COLORS["structure"], linewidth=1.2, zorder=0)


def plot_distribution(datasets, output_path):
    labels = [dataset["label"] for dataset in datasets]
    distribution = np.vstack([dataset["shares"] for dataset in datasets])
    y = np.arange(len(datasets))
    confidence_cmap = LinearSegmentedColormap.from_list(
        "mapr_confidence", [COLORS["neutral"], COLORS["confidence"]]
    )

    fig, ax_distribution = plt.subplots(figsize=(6.6, 5.8))
    fig.subplots_adjust(left=0.23, right=0.88, top=0.97, bottom=0.12)

    image = ax_distribution.imshow(
        distribution,
        aspect="auto",
        cmap=confidence_cmap,
        vmin=0,
        vmax=max(25, float(distribution.max())),
        interpolation="nearest",
    )
    ax_distribution.set_xticks(np.arange(10), np.arange(1, 11))
    ax_distribution.set_yticks(y, labels)
    ax_distribution.set_xlabel("Verbalized confidence")
    ax_distribution.tick_params(axis="both", length=0)
    for row in range(distribution.shape[0]):
        for column in range(distribution.shape[1]):
            value = distribution[row, column]
            if value >= 1:
                color = "white" if value > distribution.max() * 0.55 else "black"
                ax_distribution.text(
                    column,
                    row,
                    f"{value:.0f}",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color=color,
                )
    add_group_separators(ax_distribution)
    colorbar = fig.colorbar(image, ax=ax_distribution, fraction=0.035, pad=0.025)
    colorbar.set_label("Feedback share (%)")
    colorbar.outline.set_visible(False)
    ax_distribution.spines[["top", "right", "left"]].set_visible(False)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_accuracy_confidence_gap(datasets, output_path):
    labels = [dataset["label"] for dataset in datasets]
    y = np.arange(len(datasets))
    fig, ax_gap = plt.subplots(figsize=(6.6, 5.8))
    fig.subplots_adjust(left=0.24, right=0.97, top=0.97, bottom=0.2)

    accuracy = np.asarray([dataset["accuracy"] * 100 for dataset in datasets])
    confidence = np.asarray(
        [dataset["mean_confidence"] * 100 for dataset in datasets]
    )
    for row, (acc, conf) in enumerate(zip(accuracy, confidence)):
        ax_gap.plot(
            [acc, conf],
            [row, row],
            color=COLORS["neutral"],
            linewidth=3,
            solid_capstyle="round",
            zorder=1,
        )
    ax_gap.scatter(
        accuracy,
        y,
        s=48,
        color=CONFIDENCE_FIGURE_COLORS["feedback accuracy"],
        label="Feedback accuracy",
        zorder=3,
    )
    ax_gap.scatter(
        confidence,
        y,
        s=48,
        marker="D",
        color=CONFIDENCE_FIGURE_COLORS["mean confidence"],
        label="Mean confidence",
        zorder=3,
    )
    for row, dataset in enumerate(datasets):
        midpoint = (accuracy[row] + confidence[row]) / 2
        ax_gap.text(
            midpoint,
            row - 0.16,
            f"{dataset['calibration_gap'] * 100:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color=COLORS["secondary_text"],
        )
    ax_gap.set_xlim(50, 101)
    ax_gap.set_xticks(np.arange(50, 101, 10))
    ax_gap.set_yticks(y, labels)
    ax_gap.invert_yaxis()
    ax_gap.set_xlabel("Score (%)")
    ax_gap.grid(axis="x", color=COLORS["structure"], linewidth=0.8)
    ax_gap.set_axisbelow(True)
    ax_gap.tick_params(axis="y", length=0)
    add_group_separators(ax_gap)
    handles, legend_labels = ax_gap.get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="lower center",
        frameon=False,
        ncol=2,
        bbox_to_anchor=(0.5, 0.025),
    )

    ax_gap.spines[["top", "right", "left"]].set_visible(False)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_reliability(datasets, output_path):
    fig, axes = plt.subplots(2, 5, figsize=(13.2, 5.7), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.91, bottom=0.12, wspace=0.18, hspace=0.4)
    confidence_levels = np.arange(1, 11) / 10

    for axis, dataset in zip(axes.flat, datasets):
        counts = dataset["counts"]
        accuracy = dataset["bin_accuracy"]
        mask = counts > 0
        sizes = 24 + 170 * counts[mask] / counts.max()
        axis.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            linewidth=1.1,
            color=CONFIDENCE_FIGURE_COLORS["ideal calibration"],
            zorder=1,
        )
        axis.scatter(
            confidence_levels[mask],
            accuracy[mask],
            s=sizes,
            color=CONFIDENCE_FIGURE_COLORS["feedback accuracy"],
            alpha=0.9,
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )
        axis.set_title(
            f"{dataset['label']}\n"
            f"ACC {dataset['accuracy']:.2f}  AUROC {dataset['auroc']:.2f}  ECE {dataset['ece']:.2f}",
            fontsize=9,
        )
        axis.set_xlim(0, 1.02)
        axis.set_ylim(0, 1.02)
        axis.set_xticks(np.arange(0, 1.01, 0.2))
        axis.set_yticks(np.arange(0, 1.01, 0.2))
        axis.grid(color=COLORS["structure"], linewidth=0.7)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)

    for axis in axes[-1, :]:
        axis.set_xlabel("Confidence")
    for axis in axes[:, 0]:
        axis.set_ylabel("Feedback accuracy")

    fig.text(
        0.5,
        0.025,
        "Marker area represents the number of feedback samples at each confidence level.",
        ha="center",
        color=COLORS["secondary_text"],
        fontsize=9,
    )
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_metrics(datasets, output_path):
    fields = [
        "task",
        "n",
        "judge_model",
        "accuracy",
        "mean_confidence",
        "calibration_gap",
        "auroc",
        "ece",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for dataset in datasets:
            writer.writerow({field: dataset[field] for field in fields})


def main():
    args = parse_args()
    datasets = load_metrics(args.result_dir, args.time_flag, args.sample_size)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    distribution_path = args.output_dir / f"figure5_confidence_distribution_{args.time_flag}.png"
    gap_path = args.output_dir / f"figure5_accuracy_confidence_gap_{args.time_flag}.png"
    reliability_path = args.output_dir / f"figure5_reliability_{args.time_flag}.png"
    metrics_path = args.result_dir / f"figure5_metrics_{args.time_flag}.csv"
    plot_distribution(datasets, distribution_path)
    plot_accuracy_confidence_gap(datasets, gap_path)
    plot_reliability(datasets, reliability_path)
    write_metrics(datasets, metrics_path)
    print(distribution_path)
    print(gap_path)
    print(reliability_path)
    print(metrics_path)


if __name__ == "__main__":
    main()
