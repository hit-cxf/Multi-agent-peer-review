#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV="${CONDA_ENV:-MAPR}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/result_qwen3_8b}"
LOG_DIR="${LOG_DIR:-${PROJECT_DIR}/logs/qwen3_8b}"
SESSION_PREFIX="${SESSION_PREFIX:-MAPR}"
RELOAD_DATA="${RELOAD_DATA:-false}"

: "${OPENAI_API_KEY:?Set OPENAI_API_KEY before running this script}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
export OPENAI_MODEL="${OPENAI_MODEL:-qwen3-8b}"
export OPENAI_ENABLE_THINKING="${OPENAI_ENABLE_THINKING:-false}"

command -v tmux >/dev/null 2>&1 || { echo "tmux is not installed" >&2; exit 1; }
command -v conda >/dev/null 2>&1 || { echo "conda is not installed" >&2; exit 1; }

# A long-lived tmux server may have been created before these variables existed.
tmux set-environment -g OPENAI_API_KEY "${OPENAI_API_KEY}"
tmux set-environment -g OPENAI_BASE_URL "${OPENAI_BASE_URL}"
tmux set-environment -g OPENAI_MODEL "${OPENAI_MODEL}"
tmux set-environment -g OPENAI_ENABLE_THINKING "${OPENAI_ENABLE_THINKING}"

TASKS=(GSM8K SVAMP AQuA MultiArith AddSub SingleEq ARC-c StrategyQA Colored_Objects Penguins)
SIZES=(500 500 254 500 395 500 500 500 250 146)
METHODS=(single_agent self_correction debate feedback peer_review)

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

if [[ "${RELOAD_DATA}" == "true" ]]; then
    :
elif [[ "${RELOAD_DATA}" != "false" ]]; then
    echo "RELOAD_DATA must be true or false" >&2
    exit 1
fi

time_flag="$(date +%m%d)"
if [[ "${RELOAD_DATA}" == "false" ]]; then
    for i in "${!TASKS[@]}"; do
        task="${TASKS[$i]}"
        size="${SIZES[$i]}"
        for method in "${METHODS[@]}"; do
            output_file="${OUTPUT_DIR}/${task}/${task}_${method}_${size}_${time_flag}.json"
            if [[ -e "${output_file}" ]]; then
                echo "Refusing to overwrite existing result: ${output_file}" >&2
                echo "Use a new OUTPUT_DIR, or set RELOAD_DATA=true to resume." >&2
                exit 1
            fi
        done
    done
fi

for i in "${!TASKS[@]}"; do
    task="${TASKS[$i]}"
    size="${SIZES[$i]}"
    session="${SESSION_PREFIX}_${task//-/_}"
    task_output_dir="${OUTPUT_DIR}/${task}"
    task_log_dir="${LOG_DIR}/${task}"

    if tmux has-session -t "${session}" 2>/dev/null; then
        echo "tmux session already exists: ${session}" >&2
        exit 1
    fi

    mkdir -p "${task_output_dir}" "${task_log_dir}"

    for j in "${!METHODS[@]}"; do
        method="${METHODS[$j]}"
        window="${method}"
        log_file="${task_log_dir}/${method}.log"
        output_file="${OUTPUT_DIR}/${task}/${task}_${method}_${size}_${time_flag}.json"
        method_args=""
        reload_arg=""

        if [[ "${RELOAD_DATA}" == "true" && -e "${output_file}" ]]; then
            reload_arg="--reload_data True"
        fi

        if [[ "${method}" == "debate" ]]; then
            method_args="--rounds 2"
        elif [[ "${method}" == "feedback" || "${method}" == "peer_review" ]]; then
            method_args="--rounds 3"
        fi

        command="cd '${PROJECT_DIR}' && conda run --no-capture-output -n '${CONDA_ENV}' python '${method}.py' --task '${task}' --max_example_num '${size}' --agent_num 3 --output_dir '${OUTPUT_DIR}' ${method_args} ${reload_arg} 2>&1 | tee -a '${log_file}'"

        if [[ "$j" -eq 0 ]]; then
            tmux new-session -d -s "${session}" -n "${window}" "bash -lc \"${command}\""
        else
            tmux new-window -d -t "${session}:" -n "${window}" "bash -lc \"${command}\""
        fi
    done
done

echo "Started 50 experiment processes in 10 tmux sessions."
echo "Sessions:"
tmux list-sessions -F '#S' | grep "^${SESSION_PREFIX}_" || true
echo "Logs: ${LOG_DIR}"
echo "Results: ${OUTPUT_DIR}"
