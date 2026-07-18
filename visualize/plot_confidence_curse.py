#!/usr/bin/env python3
"""Draw the focused Figure 5 analysis for GSM8K and StrategyQA."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
TASKS = ("GSM8K", "StrategyQA")
DISPLAY_BINS = ("50", "60", "70", "80", "90", "100")
CORRECT_COLOR = "#4C92C3"
WRONG_COLOR = "#E15759"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot confidence distribution and reliability for two datasets."
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=PROJECT_DIR / "result_figure5_llm_judge",
    )
    parser.add_argument("--time-flag", default="0713")
    parser.add_argument("--sample-size", type=int, default=600)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "pics")
    return parser.parse_args()


def auroc(labels, scores):
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    if not len(positive) or not len(negative):
        return float("nan")
    differences = positive[:, None] - negative[None, :]
    return float(
        (np.sum(differences > 0) + 0.5 * np.sum(differences == 0))
        / differences.size
    )


def expected_calibration_error(labels, confidence):
    ece = 0.0
    for value in range(1, 11):
        mask = confidence == value
        if mask.any():
            ece += mask.mean() * abs(labels[mask].mean() - value / 10)
    return float(ece)


def load_dataset(result_dir, task, sample_size, time_flag):
    path = result_dir / task / f"{task}_feedback_judgments_{sample_size}_{time_flag}.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if len(rows) != sample_size:
        raise ValueError(f"{path} contains {len(rows)} rows; expected {sample_size}")
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path} contains duplicate feedback IDs")

    labels = np.asarray([int(row["feedback_correct"]) for row in rows], dtype=int)
    confidence = np.asarray(
        [int(row["verbalized_confidence"]) for row in rows], dtype=int
    )
    if np.any((confidence < 1) | (confidence > 10)):
        raise ValueError(f"{path} contains confidence outside 1-10")

    # The original figure starts at 50%. Preserve lower-confidence observations
    # by grouping confidence levels 1--5 into the first displayed bin.
    groups = [confidence <= 5] + [confidence == value for value in range(6, 11)]
    correct = np.asarray([np.sum(mask & (labels == 1)) for mask in groups])
    wrong = np.asarray([np.sum(mask & (labels == 0)) for mask in groups])
    bin_accuracy = np.asarray(
        [labels[mask].mean() if mask.any() else np.nan for mask in groups]
    )
    scores = confidence / 10
    return {
        "task": task,
        "path": path,
        "n": len(rows),
        "labels": labels,
        "confidence": confidence,
        "correct": correct,
        "wrong": wrong,
        "bin_accuracy": bin_accuracy,
        "accuracy": float(labels.mean()),
        "auroc": auroc(labels, scores),
        "ece": expected_calibration_error(labels, confidence),
    }


def plot_dataset(dataset, output_dir, time_flag):
    output_dir.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(DISPLAY_BINS))

    distribution_figure, distribution_axis = plt.subplots(figsize=(5.0, 4.2))
    distribution_axis.bar(
        x,
        dataset["correct"],
        width=0.72,
        color=CORRECT_COLOR,
        label="correct feedback",
    )
    distribution_axis.bar(
        x,
        dataset["wrong"],
        bottom=dataset["correct"],
        width=0.72,
        color=WRONG_COLOR,
        label="wrong feedback",
    )
    handles, labels = distribution_axis.get_legend_handles_labels()
    distribution_axis.legend(
        handles[::-1], labels[::-1], loc="upper left", frameon=True, fontsize=8.5
    )
    distribution_axis.set_title(
        f"ACC {dataset['accuracy']:.2f} / AUROC {dataset['auroc']:.2f} / "
        f"ECE {dataset['ece']:.2f}",
        fontsize=11,
        pad=9,
    )
    distribution_axis.set_xticks(x, DISPLAY_BINS)
    distribution_axis.set_xlabel("Confidence (%)")
    distribution_axis.set_ylabel("Count")
    distribution_stem = (
        output_dir
        / f"figure5_confidence_distribution_{dataset['task']}_{time_flag}"
    )
    distribution_figure.savefig(
        distribution_stem.with_suffix(".png"),
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    distribution_figure.savefig(
        distribution_stem.with_suffix(".pdf"),
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(distribution_figure)

    reliability_figure, reliability_axis = plt.subplots(figsize=(5.0, 4.2))
    confidence_positions = np.arange(5, 11) / 10
    mask = ~np.isnan(dataset["bin_accuracy"])
    reliability_axis.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1.4,
        color="black",
        zorder=1,
    )
    reliability_axis.bar(
        confidence_positions[mask],
        dataset["bin_accuracy"][mask],
        width=0.052,
        color=CORRECT_COLOR,
        edgecolor="black",
        linewidth=0.8,
        zorder=2,
    )
    reliability_axis.set_xlim(0, 1)
    reliability_axis.set_ylim(0, 1)
    reliability_axis.set_xticks(np.arange(0, 1.01, 0.2))
    reliability_axis.set_yticks(np.arange(0, 1.01, 0.2))
    reliability_axis.set_xlabel("Confidence")
    reliability_axis.set_ylabel("Accuracy within bin")
    reliability_stem = (
        output_dir / f"figure5_reliability_{dataset['task']}_{time_flag}"
    )
    reliability_figure.savefig(
        reliability_stem.with_suffix(".png"),
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    reliability_figure.savefig(
        reliability_stem.with_suffix(".pdf"),
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(reliability_figure)
    return distribution_stem, reliability_stem


def print_plot_data(dataset):
    print(f"dataset\t{dataset['task']}")
    print(f"source\t{dataset['path']}")
    print(f"n\t{dataset['n']}")
    print(f"ACC\t{dataset['accuracy']:.6f}")
    print(f"AUROC\t{dataset['auroc']:.6f}")
    print(f"ECE\t{dataset['ece']:.6f}")
    print("bin\tcorrect_feedback\twrong_feedback\taccuracy_within_bin")
    for label, correct, wrong, accuracy in zip(
        DISPLAY_BINS,
        dataset["correct"],
        dataset["wrong"],
        dataset["bin_accuracy"],
    ):
        accuracy_text = "NA" if np.isnan(accuracy) else f"{accuracy:.6f}"
        print(f"{label}\t{correct}\t{wrong}\t{accuracy_text}")


def main():
    args = parse_args()
    outputs = []
    for task in TASKS:
        dataset = load_dataset(
            args.result_dir, task, args.sample_size, args.time_flag
        )
        print_plot_data(dataset)
        outputs.extend(plot_dataset(dataset, args.output_dir, args.time_flag))
    print("outputs")
    for stem in outputs:
        print(stem.with_suffix(".png"))
        print(stem.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
