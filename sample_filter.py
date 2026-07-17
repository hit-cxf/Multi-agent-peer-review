"""Shared dataset-sample exclusion support for MAPR experiments."""

import json
import logging
from pathlib import Path


def add_sample_filter_args(parser):
    parser.add_argument(
        "--exclude-sample",
        "--exclude_sample",
        dest="exclude_samples",
        action="append",
        default=[],
        metavar="DATASET:INDEX",
        help=(
            "Exclude one 1-based sample from a dataset, for example "
            "--exclude-sample StrategyQA:385. Repeat for multiple samples."
        ),
    )


def parse_exclusion_specs(specs, valid_tasks=None):
    """Return {(dataset, one_based_index)} after strict validation."""
    exclusions = set()
    valid_tasks = set(valid_tasks) if valid_tasks is not None else None
    for spec in specs or []:
        if spec.count(":") != 1:
            raise ValueError(
                f"Invalid exclusion {spec!r}; expected DATASET:INDEX, e.g. StrategyQA:385"
            )
        dataset, index_text = (part.strip() for part in spec.split(":", 1))
        if not dataset:
            raise ValueError(f"Invalid exclusion {spec!r}: dataset is empty")
        if valid_tasks is not None and dataset not in valid_tasks:
            raise ValueError(f"Invalid exclusion {spec!r}: unknown dataset {dataset!r}")
        try:
            index = int(index_text)
        except ValueError as exc:
            raise ValueError(f"Invalid exclusion {spec!r}: index must be an integer") from exc
        if index < 1:
            raise ValueError(f"Invalid exclusion {spec!r}: index must be positive and 1-based")
        exclusions.add((dataset, index))
    return exclusions


def finalize_sample_filter_args(args, valid_tasks):
    try:
        args.sample_exclusions = parse_exclusion_specs(args.exclude_samples, valid_tasks)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    return args


def is_excluded(exclusions, dataset, one_based_index):
    return (dataset, one_based_index) in exclusions


def excluded_record(data, dataset, one_based_index):
    return {
        "question": data["question"],
        "answer": data["answer"],
        "agent_contexts": [],
        "excluded": True,
        "excluded_dataset": dataset,
        "dataset_index": one_based_index,
        "exclusion_reason": "provider data_inspection_failed",
    }


def append_excluded_record_if_needed(
    generated, data, dataset, zero_based_index, output_file, exclusions
):
    """Append and persist an aligned placeholder; return True when excluded."""
    one_based_index = zero_based_index + 1
    if not is_excluded(exclusions, dataset, one_based_index):
        return False
    generated.append(excluded_record(data, dataset, one_based_index))
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    Path(output_file).write_text(
        json.dumps(generated, ensure_ascii=False), encoding="utf-8"
    )
    logging.warning(
        "excluded sample dataset=%s index=%d; no API request was sent",
        dataset,
        one_based_index,
    )
    return True


def filter_evaluation_rows(rows):
    included = [row for row in rows if not row.get("excluded", False)]
    excluded = [row for row in rows if row.get("excluded", False)]
    return included, excluded


def excluded_keys(rows):
    return {
        (row.get("excluded_dataset"), row.get("dataset_index"))
        for row in rows
        if row.get("excluded", False)
    }
