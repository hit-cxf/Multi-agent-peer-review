#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV="${CONDA_ENV:-MAPR}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/result_qwen3_8b}"
LOG_DIR="${LOG_DIR:-${PROJECT_DIR}/logs/qwen3_8b}"
SESSION_PREFIX="${SESSION_PREFIX:-MAPR}"
RELOAD_DATA="${RELOAD_DATA:-false}"
TIME_FLAG="${TIME_FLAG:-$(date +%m%d)}"

export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
export OPENAI_MODEL="${OPENAI_MODEL:-qwen3-8b}"
export OPENAI_ENABLE_THINKING="${OPENAI_ENABLE_THINKING:-false}"

command -v tmux >/dev/null 2>&1 || { echo "tmux is not installed" >&2; exit 1; }
command -v conda >/dev/null 2>&1 || { echo "conda is not installed" >&2; exit 1; }

ALL_TASKS=(GSM8K SVAMP AQuA MultiArith AddSub SingleEq ARC-c StrategyQA Colored_Objects Penguins)
ALL_METHODS=(single_agent self_correction debate feedback peer_review ablation_solution)
TASKS=()
METHODS=()
tasks_csv=""
methods_csv=""

usage() {
    echo "Usage: $0 [--tasks TASK1,TASK2] [--methods METHOD1,METHOD2] [--time-flag MMDD]"
    echo "Tasks: ${ALL_TASKS[*]}"
    echo "Methods: ${ALL_METHODS[*]}"
}

contains() {
    value="$1"
    shift
    for item in "$@"; do
        [[ "${item}" == "${value}" ]] && return 0
    done
    return 1
}

task_size() {
    case "$1" in
        GSM8K|SVAMP|MultiArith|SingleEq|ARC-c|StrategyQA) echo 500 ;;
        AQuA) echo 254 ;;
        AddSub) echo 395 ;;
        Colored_Objects) echo 250 ;;
        Penguins) echo 146 ;;
        *) echo "Unknown task: $1" >&2; return 1 ;;
    esac
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tasks)
            [[ $# -ge 2 ]] || { echo "--tasks requires a comma-separated value" >&2; exit 2; }
            tasks_csv="$2"
            shift 2
            ;;
        --methods)
            [[ $# -ge 2 ]] || { echo "--methods requires a comma-separated value" >&2; exit 2; }
            methods_csv="$2"
            shift 2
            ;;
        --time-flag)
            [[ $# -ge 2 ]] || { echo "--time-flag requires a four-digit value such as 0713" >&2; exit 2; }
            TIME_FLAG="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ -n "${tasks_csv}" ]]; then
    IFS=',' read -r -a TASKS <<< "${tasks_csv}"
else
    TASKS=("${ALL_TASKS[@]}")
fi
if [[ -n "${methods_csv}" ]]; then
    IFS=',' read -r -a METHODS <<< "${methods_csv}"
else
    METHODS=("${ALL_METHODS[@]}")
fi

for task in "${TASKS[@]}"; do
    contains "${task}" "${ALL_TASKS[@]}" || { echo "Unknown task: ${task}" >&2; usage >&2; exit 2; }
done
for method in "${METHODS[@]}"; do
    contains "${method}" "${ALL_METHODS[@]}" || { echo "Unknown method: ${method}" >&2; usage >&2; exit 2; }
done
[[ ${#TASKS[@]} -gt 0 ]] || { echo "No tasks selected" >&2; exit 2; }
[[ ${#METHODS[@]} -gt 0 ]] || { echo "No methods selected" >&2; exit 2; }
[[ "${TIME_FLAG}" =~ ^[0-9]{4}$ ]] || { echo "--time-flag must contain exactly four digits, such as 0713" >&2; exit 2; }

: "${OPENAI_API_KEY:?Set OPENAI_API_KEY before running this script}"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

if [[ "${RELOAD_DATA}" == "true" ]]; then
    :
elif [[ "${RELOAD_DATA}" != "false" ]]; then
    echo "RELOAD_DATA must be true or false" >&2
    exit 1
fi

time_flag="${TIME_FLAG}"
if [[ "${RELOAD_DATA}" == "false" ]]; then
    for i in "${!TASKS[@]}"; do
        task="${TASKS[$i]}"
        size="$(task_size "${task}")"
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
    size="$(task_size "${task}")"
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
        elif [[ "${method}" == "feedback" || "${method}" == "peer_review" || "${method}" == "ablation_solution" ]]; then
            method_args="--rounds 3"
        fi

        command="cd '${PROJECT_DIR}' && conda run --no-capture-output -n '${CONDA_ENV}' python '${method}.py' --task '${task}' --max_example_num '${size}' --agent_num 3 --output_dir '${OUTPUT_DIR}' ${method_args} ${reload_arg} 2>&1 | tee -a '${log_file}'"

        if [[ "$j" -eq 0 ]]; then
            tmux new-session -d -s "${session}" -n "${window}" \
                -e "OPENAI_API_KEY=${OPENAI_API_KEY}" \
                -e "OPENAI_BASE_URL=${OPENAI_BASE_URL}" \
                -e "OPENAI_MODEL=${OPENAI_MODEL}" \
                -e "OPENAI_ENABLE_THINKING=${OPENAI_ENABLE_THINKING}" \
                -e "MAPR_TIME_FLAG=${time_flag}" \
                "bash -lc \"${command}\""
        else
            tmux new-window -d -t "${session}:" -n "${window}" "bash -lc \"${command}\""
        fi
    done
done

process_count=$((${#TASKS[@]} * ${#METHODS[@]}))
echo "Started ${process_count} experiment processes in ${#TASKS[@]} tmux sessions."
echo "Sessions:"
tmux list-sessions -F '#S' | grep "^${SESSION_PREFIX}_" || true
echo "Logs: ${LOG_DIR}"
echo "Results: ${OUTPUT_DIR}"
