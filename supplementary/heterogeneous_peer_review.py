#!/usr/bin/env python3
"""Strict reproduction of the two-LLM peer-review protocol in paper Table 5."""

import argparse
import fcntl
import json
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

import sys

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import llm_client  # noqa: E402
import sample_filter  # noqa: E402


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def slug(value):
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run paper Table 5 with two heterogeneous LLM agents."
    )
    parser.add_argument("--models", required=True, help="Exactly two comma-separated model names")
    parser.add_argument("--task", choices=["GSM8K", "StrategyQA"], default="GSM8K")
    parser.add_argument("--dataset-dir", type=Path, default=PROJECT_DIR / "processed_data")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "result_heterogeneous_llm")
    parser.add_argument("--initial-cache-dir", type=Path)
    parser.add_argument("--max-example-num", type=int, default=500)
    parser.add_argument("--time-flag", default=os.getenv("MAPR_TIME_FLAG", datetime.now().strftime("%m%d")))
    parser.add_argument("--reload-data", action="store_true")
    sample_filter.add_sample_filter_args(parser)
    args = parser.parse_args()

    args.models = [model.strip() for model in args.models.split(",") if model.strip()]
    if len(args.models) != 2:
        parser.error("--models must contain exactly two model names")
    if args.models[0] == args.models[1]:
        parser.error("Table 5 heterogeneous runs require two different models")
    if not re.fullmatch(r"\d{4}", args.time_flag):
        parser.error("--time-flag must be four digits, e.g. 0719")

    args.sample_exclusions = sample_filter.parse_exclusion_specs(
        args.exclude_samples, [args.task]
    )
    args.task_file = args.dataset_dir / args.task / f"{args.task}_{args.max_example_num}.jsonl"
    pair_slug = "__".join(slug(model) for model in args.models)
    args.pair_dir = args.output_dir / args.task / pair_slug
    args.output_file = (
        args.pair_dir
        / f"{args.task}_heterogeneous_peer_review_{args.max_example_num}_{args.time_flag}.json"
    )
    args.initial_cache_dir = args.initial_cache_dir or (
        args.output_dir / args.task / "initial_cache" / args.time_flag
    )
    return args


def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def completion_content(completion):
    return completion["choices"][0]["message"]["content"]


def generate(messages, model):
    while True:
        try:
            return completion_content(
                llm_client.create_chat_completion(messages, model=model)
            )
        except Exception as exc:
            logging.warning("model=%s retrying due to error: %s", model, exc)
            time.sleep(20)


class InitialResponseStore:
    """One shared initial response per model/sample across every model pair."""

    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def get_or_create(self, model, index, question, prompt):
        path = self.root / f"{slug(model)}.jsonl"
        lock_path = self.root / f"{slug(model)}.lock"
        with lock_path.open("a+", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            records = {}
            if path.exists():
                for record in read_jsonl(path):
                    records[record["dataset_index"]] = record
            if index in records:
                record = records[index]
                if record["question"] != question:
                    raise ValueError(f"initial cache question mismatch: {model} sample {index}")
                return record["response"]

            response = generate([{"role": "user", "content": prompt}], model)
            record = {
                "dataset_index": index,
                "question": question,
                "model": model,
                "response": response,
            }
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            return response


def question_prompt(task, question):
    if task == "GSM8K":
        return (
            "Can you solve the following math problem? {} Explain your reasoning. "
            "Your final answer should be a single numerical number, in the form "
            "\\boxed{{answer}}, at the end of your response. "
        ).format(question)
    return (
        "Can you answer the following question as accurately as possible? {} Explain "
        "your answer, your answer should be Yes or No at the end of your response."
    ).format(question)


def review_prompt(peer_answer):
    return (
        f"Here is a solution from another agent: \n\n {peer_answer}\n\n "
        "Please examine this agent's reasoning process step by step and offer feedback on its reasoning. "
        "You can rate your confidence in your feedback on a scale from 1-10, where 10 indicates "
        "the highest level of confidence."
    )


def revision_prompt(task, question, feedback):
    prefix = (
        "Here are the feedbacks for your solution from the above one agents:\n\n "
        f"One agent feedback: {feedback} \n\n "
    )
    if task == "GSM8K":
        return prefix + (
            "Using other agents' solutions and feedbacks as additional information, "
            "can you provide your answer to the math problem? \n "
            f"The original math problem is {question}. "
            "Your final answer should be a single numerical number, "
            "in the form \\boxed{answer}, at the end of your response."
        )
    return prefix + (
        "Using the reasoning from other agents as additional advice, "
        "can you give an updated answer? Examine your solution and other agents' feedback step by step. "
        "Your answer should be Yes or No at the end of your response."
    )


def atomic_write(path, rows):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def run(args):
    args.pair_dir.mkdir(parents=True, exist_ok=True)
    if args.reload_data:
        if not args.output_file.exists():
            raise FileNotFoundError(f"cannot reload missing result: {args.output_file}")
        generated = json.loads(args.output_file.read_text(encoding="utf-8"))
    else:
        if args.output_file.exists():
            raise FileExistsError(f"refusing to overwrite: {args.output_file}")
        generated = []

    logging.info("models=%s", args.models)
    logging.info("initial settings=%s", [llm_client.safe_model_settings(m) for m in args.models])
    logging.info("output=%s reload_rows=%d", args.output_file, len(generated))
    initial_store = InitialResponseStore(args.initial_cache_dir)
    data_rows = read_jsonl(args.task_file)

    for zero_index, data in enumerate(tqdm(data_rows)):
        if zero_index < len(generated):
            continue
        if sample_filter.append_excluded_record_if_needed(
            generated,
            data,
            args.task,
            zero_index,
            args.output_file,
            args.sample_exclusions,
        ):
            continue

        dataset_index = zero_index + 1
        question = data["question"]
        prompt = question_prompt(args.task, question)
        initial_responses = [
            initial_store.get_or_create(model, dataset_index, question, prompt)
            for model in args.models
        ]
        contexts = [
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": initial_responses[index]},
            ]
            for index in range(2)
        ]

        reviews_given = []
        for reviewer_index in range(2):
            target_index = 1 - reviewer_index
            contexts[reviewer_index].append(
                {"role": "user", "content": review_prompt(initial_responses[target_index])}
            )
            review = generate(contexts[reviewer_index], args.models[reviewer_index])
            contexts[reviewer_index].append({"role": "assistant", "content": review})
            reviews_given.append(review)

        revised_responses = []
        for agent_index in range(2):
            feedback_received = reviews_given[1 - agent_index]
            contexts[agent_index].append(
                {
                    "role": "user",
                    "content": revision_prompt(args.task, question, feedback_received),
                }
            )
            revised = generate(contexts[agent_index], args.models[agent_index])
            contexts[agent_index].append({"role": "assistant", "content": revised})
            revised_responses.append(revised)

        generated.append(
            {
                "dataset_index": dataset_index,
                "question": question,
                "answer": data["answer"],
                "agent_models": args.models,
                "initial_responses": initial_responses,
                "reviews_given": reviews_given,
                "revised_responses": revised_responses,
                "agent_contexts": contexts,
            }
        )
        atomic_write(args.output_file, generated)


def main():
    args = parse_args()
    if not args.task_file.exists():
        raise FileNotFoundError(args.task_file)
    run(args)


if __name__ == "__main__":
    main()
