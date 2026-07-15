#!/usr/bin/env python3
"""MAPR with a real, configurable number of Review -> Revision cycles."""

import argparse
import json
import logging
import os
import sys
import time
import urllib.error
from pathlib import Path

from tqdm import tqdm


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from data_proc import check_dirs_files  # noqa: E402
import llm_client  # noqa: E402
from params import TASK_FILE  # noqa: E402


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
TASKS = list(TASK_FILE)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run peer review with one or more complete Review -> Revision cycles."
    )
    parser.add_argument("--dataset_dir", default=str(PROJECT_DIR / "processed_data"))
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--max_example_num", type=int, required=True)
    parser.add_argument("--agent_num", type=int, required=True)
    parser.add_argument("--review_rounds", type=int, required=True)
    parser.add_argument(
        "--exclude_indices",
        default="",
        help="Comma-separated 1-based dataset indices excluded only from this supplementary run.",
    )
    parser.add_argument("--reload_data", action="store_true")
    parser.add_argument("--api_key", default=os.getenv("OPENAI_API_KEY", ""))
    parser.add_argument(
        "--base_url", default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    )
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo-0613"))
    parser.add_argument(
        "--enable_thinking", choices=["true", "false"], default=os.getenv("OPENAI_ENABLE_THINKING")
    )
    parser.add_argument("--openai_organization", default=os.getenv("OPENAI_ORGANIZATION", ""))
    args = parser.parse_args()
    if args.agent_num < 2:
        parser.error("--agent_num must be at least 2")
    if args.review_rounds < 1:
        parser.error("--review_rounds must be at least 1")
    try:
        args.exclude_indices = {
            int(value.strip()) for value in args.exclude_indices.split(",") if value.strip()
        }
    except ValueError:
        parser.error("--exclude_indices must be comma-separated positive integers")
    if any(index < 1 for index in args.exclude_indices):
        parser.error("--exclude_indices values must be positive 1-based indices")
    args.task_file = str(
        Path(args.dataset_dir) / args.task / f"{args.task}_{args.max_example_num}.jsonl"
    )
    return args


def construct_assistant_message(completion):
    return {"role": "assistant", "content": completion["choices"][0]["message"]["content"]}


def generate_answer(context):
    while True:
        try:
            return llm_client.create_chat_completion(context)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            context_chars = sum(len(message.get("content", "")) for message in context)
            logging.error(
                "[MAPR-HTTP] status=%s messages=%d context_chars=%d response=%s",
                exc.code,
                len(context),
                context_chars,
                body,
            )
            # 429 is transient. Other 4xx responses describe a request that
            # will not change, so retrying it forever only stalls the dataset.
            if exc.code != 429 and 400 <= exc.code < 500:
                raise RuntimeError(
                    f"non-retryable HTTP {exc.code}; inspect the [MAPR-HTTP] log entry"
                ) from exc
            logging.warning("retrying after HTTP %s", exc.code)
            time.sleep(20)
        except Exception as exc:
            logging.warning("retrying due to an error: %s", exc)
            time.sleep(20)


def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def question_prompt(task, question):
    if task in ["GSM8K", "SVAMP", "AddSub", "SingleEq", "MultiArith"]:
        return (
            "Can you solve the following math problem? {} Explain your reasoning. "
            "Your final answer should be a single numerical number, in the form "
            "\\boxed{{answer}}, at the end of your response. "
        ).format(question)
    if task in ["AQuA", "ARC-c", "Colored_Objects", "Penguins"]:
        return (
            "Can you answer the following question as accurately as possible? {} Explain "
            "your answer, putting the answer in the form (X) at the end of your response."
        ).format(question)
    if task == "StrategyQA":
        return (
            "Can you answer the following question as accurately as possible? {} Explain "
            "your answer, your answer should be Yes or No at the end of your response."
        ).format(question)
    raise ValueError(f"unknown task: {task}")


def review_prompt(peer_answer):
    # Kept byte-for-byte equivalent in content to peer_review.py.
    return (
        f"Here is a solution from another agent: \n\n {peer_answer}\n\n "
        "Please examine this agent's reasoning process step by step and offer feedback on its reasoning. "
        "You can rate your confidence in your feedback on a scale from 1-10, where 10 indicates "
        "the highest level of confidence."
    )


def revision_prompt(task, question, feedbacks, agent_num):
    number_words = {1: "one", 2: "two", 3: "three", 4: "four"}
    reviewer_count = agent_num - 1
    reviewer_label = number_words.get(reviewer_count, str(reviewer_count))
    content = f"Here are the feedbacks for your solution from the above {reviewer_label} agents:\n\n "
    for feedback in feedbacks:
        content += f"One agent feedback: {feedback['content']} \n\n "

    if task in ["GSM8K", "SVAMP", "AddSub", "SingleEq", "MultiArith"]:
        content += (
            "Using other agents' solutions and feedbacks as additional information, "
            "can you provide your answer to the math problem? \n "
            f"The original math problem is {question}. "
            "Your final answer should be a single numerical number, "
            "in the form \\boxed{answer}, at the end of your response."
        )
    elif task in ["AQuA", "ARC-c", "Colored_Objects", "Penguins"]:
        content += (
            "Using the reasoning from other agents as additional advice, can you give an updated answer? "
            "Examine your solution and other agents' feedback step by step. "
            "Put your answer in the form (X) at the end of your response."
        )
    elif task == "StrategyQA":
        content += (
            "Using the reasoning from other agents as additional advice, can you give an updated answer? "
            "Examine your solution and other agents' feedback step by step. "
            "Your answer should be Yes or No at the end of your response."
        )
    else:
        raise ValueError(f"unknown task: {task}")
    return content


def run_peer_review(args):
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if args.reload_data:
        check_dirs_files(dirs=[], files=[str(output_file)])
        generated = json.loads(output_file.read_text(encoding="utf-8"))
    else:
        generated = []
    generated_len = len(generated)
    if generated_len:
        logging.info("reload from: %s", output_file)
        logging.info("reload data num: %d", generated_len)

    for index, data in enumerate(tqdm(read_jsonl(args.task_file))):
        if args.reload_data and index < generated_len:
            continue

        dataset_index = index + 1
        if dataset_index in args.exclude_indices:
            generated.append(
                {
                    "question": data["question"],
                    "answer": data["answer"],
                    "agent_contexts": [],
                    "excluded": True,
                    "exclusion_reason": "DashScope data_inspection_failed",
                    "dataset_index": dataset_index,
                }
            )
            output_file.write_text(json.dumps(generated, ensure_ascii=False), encoding="utf-8")
            logging.warning(
                "excluded supplementary sample %d for task %s: DashScope data_inspection_failed",
                dataset_index,
                args.task,
            )
            continue

        question = data["question"]
        contexts = [
            [{"role": "user", "content": question_prompt(args.task, question)}]
            for _ in range(args.agent_num)
        ]

        # Creation is performed once.
        for context in contexts:
            context.append(construct_assistant_message(generate_answer(context)))

        round_answers = []
        for review_round in range(args.review_rounds):
            current_answers = [context[-1]["content"] for context in contexts]
            feedbacks = [[] for _ in range(args.agent_num)]

            # Review: preserve the original reviewer and target ordering.
            for reviewer_index, context in enumerate(contexts):
                for target_index in range(args.agent_num):
                    if target_index == reviewer_index:
                        continue
                    context.append(
                        {"role": "user", "content": review_prompt(current_answers[target_index])}
                    )
                    feedback = construct_assistant_message(generate_answer(context))
                    context.append(feedback)
                    feedbacks[target_index].append(feedback)

            # Revision: every agent receives all reviews of its current answer.
            for agent_index, context in enumerate(contexts):
                context.append(
                    {
                        "role": "user",
                        "content": revision_prompt(
                            args.task, question, feedbacks[agent_index], args.agent_num
                        ),
                    }
                )
                context.append(construct_assistant_message(generate_answer(context)))

            round_answers.append([context[-1]["content"] for context in contexts])
            logging.debug("completed review round %d", review_round + 1)

        generated.append(
            {
                "question": question,
                "answer": data["answer"],
                "agent_contexts": contexts,
                "round_answers": round_answers,
            }
        )
        output_file.write_text(json.dumps(generated, ensure_ascii=False), encoding="utf-8")


def main():
    args = parse_args()
    check_dirs_files(dirs=[args.dataset_dir], files=[args.task_file])
    logging.info(
        "task=%s agent_num=%d review_rounds=%d output=%s",
        args.task,
        args.agent_num,
        args.review_rounds,
        args.output_file,
    )
    llm_client.configure(args)
    run_peer_review(args)


if __name__ == "__main__":
    main()
