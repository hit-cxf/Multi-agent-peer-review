#!/usr/bin/env python3
"""Use a configurable OpenAI-compatible LLM to label MAPR feedback correctness.

The script reconstructs every Stage-2 review from an existing peer-review result,
samples a reproducible subset per task, and writes resumable JSONL judgments for
Figure 5 confidence-distribution and calibration analysis.
"""

import argparse
import concurrent.futures
import json
import os
import random
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
TASK_SIZES = {
    "GSM8K": 500,
    "SVAMP": 500,
    "AQuA": 254,
    "MultiArith": 500,
    "AddSub": 395,
    "SingleEq": 500,
    "ARC-c": 500,
    "StrategyQA": 500,
    "Colored_Objects": 250,
    "Penguins": 146,
}
SYSTEM_PROMPT = """You are a rigorous evaluator of peer feedback on reasoning solutions.
Judge whether the feedback correctly assesses the TARGET SOLUTION with respect to the
QUESTION and GOLD ANSWER.

Label the feedback "correct" only when its substantive assessment is accurate: it
correctly accepts a sound solution, or correctly identifies genuine errors in an
unsound solution without introducing a material false claim. Label it "incorrect"
when it accepts a flawed solution, rejects a sound solution, identifies a nonexistent
error, misses the central error while giving a misleading overall assessment, or
contains a material factual/reasoning mistake.

Evaluate the feedback itself, not whether the reviewer or target eventually produced
the right final answer. Ignore the feedback's stated confidence when deciding the
label. Return exactly one JSON object with keys: label, rationale. The label must be
either "correct" or "incorrect". Keep rationale concise."""

CONFIDENCE_PATTERNS = [
    re.compile(r"(?i)confidence(?:\s+(?:level|score))?\s*(?:is|of|:|=)?\s*(10|[1-9])\s*(?:/\s*10|out\s+of\s+10)?"),
    re.compile(r"(?i)rate\s+(?:my|the)?\s*confidence[^\d]{0,30}(10|[1-9])\s*(?:/\s*10|out\s+of\s+10)?"),
    re.compile(r"(?i)(10|[1-9])\s*(?:/\s*10|out\s+of\s+10)\s*(?:confidence)?"),
    re.compile(r"(?i)confidence[^.\n]{0,80}\b(10|[1-9])\b"),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Label peer-feedback correctness with a configurable LLM judge."
    )
    parser.add_argument("--result-dir", type=Path, default=PROJECT_DIR / "result_qwen3_8b")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "result_figure5_llm_judge")
    parser.add_argument("--tasks", nargs="+", choices=TASK_SIZES, default=list(TASK_SIZES))
    parser.add_argument("--time-flag", default="0713")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=600,
        help="Feedback records sampled per task; use 0 to judge every record.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-retries", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--api-key", default=os.getenv("JUDGE_API_KEY", os.getenv("OPENAI_API_KEY", "")))
    parser.add_argument(
        "--base-url",
        default=os.getenv(
            "JUDGE_BASE_URL",
            os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        ),
    )
    parser.add_argument("--model", default=os.getenv("JUDGE_MODEL", os.getenv("OPENAI_MODEL", "")))
    parser.add_argument(
        "--enable-thinking",
        choices=["true", "false"],
        default=os.getenv("JUDGE_ENABLE_THINKING"),
    )
    parser.add_argument("--organization", default=os.getenv("JUDGE_ORGANIZATION", ""))
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only extract and sample feedback; do not call the judge API.",
    )
    return parser.parse_args()


def extract_confidence(text):
    for pattern in CONFIDENCE_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            value = int(matches[-1])
            if 1 <= value <= 10:
                return value
    return None


def extract_feedback_records(result_file, task):
    rows = json.loads(result_file.read_text(encoding="utf-8"))
    records = []
    for sample_index, row in enumerate(rows):
        contexts = row["agent_contexts"]
        agent_num = len(contexts)
        initial_solutions = [context[1]["content"] for context in contexts]
        for reviewer_index, context in enumerate(contexts):
            target_indices = [index for index in range(agent_num) if index != reviewer_index]
            for review_order, target_index in enumerate(target_indices):
                feedback_position = 3 + review_order * 2
                if feedback_position >= len(context):
                    raise ValueError(
                        f"Missing review message: task={task}, sample={sample_index}, "
                        f"reviewer={reviewer_index}, target={target_index}"
                    )
                feedback = context[feedback_position]["content"]
                records.append(
                    {
                        "id": f"{task}:{sample_index}:{reviewer_index}:{target_index}",
                        "task": task,
                        "sample_index": sample_index,
                        "reviewer_index": reviewer_index,
                        "target_index": target_index,
                        "question": row["question"],
                        "gold_answer": row["answer"],
                        "target_solution": initial_solutions[target_index],
                        "feedback": feedback,
                        "verbalized_confidence": extract_confidence(feedback),
                    }
                )
    return records


def sample_records(records, sample_size, seed, task):
    eligible = [record for record in records if record["verbalized_confidence"] is not None]
    if sample_size == 0:
        return eligible
    if sample_size > len(eligible):
        raise ValueError(
            f"{task}: requested {sample_size} feedback records, but only "
            f"{len(eligible)} have a parsed confidence score"
        )
    if sample_size == len(eligible):
        return eligible
    rng = random.Random(f"{seed}:{task}")
    selected = rng.sample(eligible, sample_size)
    return sorted(selected, key=lambda item: item["id"])


def build_user_prompt(record):
    return f"""QUESTION:
{record['question']}

GOLD ANSWER:
{record['gold_answer']}

TARGET SOLUTION:
{record['target_solution']}

PEER FEEDBACK TO EVALUATE:
{record['feedback']}

Judge whether the peer feedback is correct. Return JSON only."""


def parse_judge_response(text):
    stripped = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.DOTALL | re.IGNORECASE)
    candidate = fenced.group(1) if fenced else stripped
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        object_match = re.search(r"\{.*\}", candidate, re.DOTALL)
        if not object_match:
            raise ValueError(f"Judge did not return JSON: {text[:300]}")
        payload = json.loads(object_match.group(0))
    label = str(payload.get("label", "")).strip().lower()
    if label not in {"correct", "incorrect"}:
        raise ValueError(f"Invalid judge label: {payload.get('label')!r}")
    rationale = str(payload.get("rationale", "")).strip()
    if not rationale:
        raise ValueError("Judge response is missing rationale")
    return label, rationale


class JudgeClient:
    def __init__(self, args):
        self.api_key = args.api_key
        self.base_url = args.base_url.rstrip("/")
        self.model = args.model
        self.enable_thinking = args.enable_thinking
        self.organization = args.organization
        self.timeout = args.timeout
        self.max_retries = args.max_retries

    def judge(self, record):
        request_body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(record)},
            ],
            "temperature": 0,
            "n": 1,
        }
        if self.enable_thinking is not None:
            request_body["enable_thinking"] = self.enable_thinking == "true"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.organization:
            headers["OpenAI-Organization"] = self.organization

        last_error = None
        for attempt in range(self.max_retries):
            try:
                request = urllib.request.Request(
                    f"{self.base_url}/chat/completions",
                    data=json.dumps(request_body).encode("utf-8"),
                    headers=headers,
                    method="POST",
                )
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    completion = json.load(response)
                raw_output = completion["choices"][0]["message"]["content"]
                label, rationale = parse_judge_response(raw_output)
                return {
                    **record,
                    "feedback_correctness": label,
                    "feedback_correct": int(label == "correct"),
                    "judge_rationale": rationale,
                    "judge_raw_output": raw_output,
                    "judge_model": self.model,
                }
            except Exception as exc:
                last_error = exc
                if attempt + 1 == self.max_retries:
                    break
                delay = min(60, 2 ** attempt) + random.random()
                time.sleep(delay)
        raise RuntimeError(f"Judge failed after {self.max_retries} attempts: {last_error}")


def read_completed(output_file):
    completed = {}
    if not output_file.exists():
        return completed
    with output_file.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {output_file}:{line_number}: {exc}") from exc
            completed[record["id"]] = record
    return completed


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def evaluate_task(args, task, client):
    size = TASK_SIZES[task]
    result_file = args.result_dir / task / f"{task}_peer_review_{size}_{args.time_flag}.json"
    if not result_file.exists():
        raise FileNotFoundError(result_file)
    all_records = extract_feedback_records(result_file, task)
    selected = sample_records(all_records, args.sample_size, args.seed, task)

    task_dir = args.output_dir / task
    manifest_file = task_dir / f"{task}_feedback_sample_{len(selected)}_{args.time_flag}.jsonl"
    output_file = task_dir / f"{task}_feedback_judgments_{len(selected)}_{args.time_flag}.jsonl"
    write_jsonl(manifest_file, selected)

    eligible_count = sum(
        record["verbalized_confidence"] is not None for record in all_records
    )
    print(
        f"{task}: extracted={len(all_records)}, confidence_eligible={eligible_count}, "
        f"skipped_without_confidence={len(all_records) - eligible_count}, "
        f"selected={len(selected)}"
    )
    if args.prepare_only:
        return

    completed = read_completed(output_file)
    pending = [record for record in selected if record["id"] not in completed]
    print(f"{task}: completed={len(completed)}, pending={len(pending)}, output={output_file}")
    if not pending:
        return

    write_lock = threading.Lock()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("a", encoding="utf-8") as handle:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(client.judge, record): record for record in pending}
            done = 0
            for future in concurrent.futures.as_completed(futures):
                record = future.result()
                with write_lock:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    handle.flush()
                done += 1
                if done % 25 == 0 or done == len(pending):
                    print(f"{task}: judged {done}/{len(pending)} new records")


def validate_args(args):
    if args.sample_size < 0:
        raise ValueError("--sample-size must be non-negative")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if not args.prepare_only:
        if not args.api_key:
            raise ValueError("Set --api-key or JUDGE_API_KEY")
        if not args.model:
            raise ValueError("Set --model or JUDGE_MODEL")


def main():
    args = parse_args()
    validate_args(args)
    client = None if args.prepare_only else JudgeClient(args)
    print(
        f"judge_model={args.model or '(prepare only)'}, base_url={args.base_url}, "
        f"workers={args.workers}, sample_size={args.sample_size}, seed={args.seed}"
    )
    for task in args.tasks:
        evaluate_task(args, task, client)


if __name__ == "__main__":
    main()
