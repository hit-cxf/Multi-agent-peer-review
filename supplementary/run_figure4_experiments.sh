#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-MAPR}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/result_agent_num_review_rounds}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_DIR}/logs/figure4_agent_num_review_rounds}"
SESSION_PREFIX="${SESSION_PREFIX:-MAPR_F4}"
TIME_FLAG="${TIME_FLAG:-$(date +%m%d)}"
RELOAD_DATA="${RELOAD_DATA:-false}"

export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
export OPENAI_MODEL="${OPENAI_MODEL:-qwen3-8b}"
export OPENAI_ENABLE_THINKING="${OPENAI_ENABLE_THINKING:-false}"

TASKS=(GSM8K AQuA StrategyQA)
CONFIGS=("2 1" "4 1" "5 1" "3 2" "3 3" "3 4")

task_size() {
    case "$1" in
        GSM8K|StrategyQA) echo 500 ;;
        AQuA) echo 254 ;;
        *) echo "Unknown task: $1" >&2; return 1 ;;
    esac
}

[[ "${TIME_FLAG}" =~ ^[0-9]{4}$ ]] || { echo "TIME_FLAG must be four digits such as 0715" >&2; exit 2; }
[[ "${RELOAD_DATA}" == "true" || "${RELOAD_DATA}" == "false" ]] || {
    echo "RELOAD_DATA must be true or false" >&2
    exit 2
}
command -v tmux >/dev/null 2>&1 || { echo "tmux is not installed" >&2; exit 1; }
command -v conda >/dev/null 2>&1 || { echo "conda is not installed" >&2; exit 1; }
: "${OPENAI_API_KEY:?Set OPENAI_API_KEY before running this script}"

for config in "${CONFIGS[@]}"; do
    read -r agent_num review_rounds <<< "${config}"
    config_name="agent_num_${agent_num}_review_rounds_${review_rounds}"
    session="${SESSION_PREFIX}_a${agent_num}_r${review_rounds}"

    if tmux has-session -t "${session}" 2>/dev/null; then
        echo "tmux session already exists: ${session}" >&2
        exit 1
    fi

    for index in "${!TASKS[@]}"; do
        task="${TASKS[$index]}"
        size="$(task_size "${task}")"
        output_dir="${OUTPUT_ROOT}/${config_name}/${task}"
        output_file="${output_dir}/${task}_peer_review_${size}_${TIME_FLAG}.json"
        log_dir="${LOG_ROOT}/${config_name}/${task}"
        log_file="${log_dir}/peer_review.log"
        reload_arg=""

        mkdir -p "${output_dir}" "${log_dir}"
        if [[ -e "${output_file}" ]]; then
            if [[ "${RELOAD_DATA}" == "true" ]]; then
                reload_arg="--reload_data"
            else
                echo "Refusing to overwrite existing result: ${output_file}" >&2
                exit 1
            fi
        fi

        command="cd '${PROJECT_DIR}' && conda run --no-capture-output -n '${CONDA_ENV}' python supplementary/peer_review_cycles.py --task '${task}' --max_example_num '${size}' --agent_num '${agent_num}' --review_rounds '${review_rounds}' --output_file '${output_file}' ${reload_arg} 2>&1 | tee -a '${log_file}'"

        if [[ "${index}" -eq 0 ]]; then
            tmux new-session -d -s "${session}" -n "${task}" \
                -e "OPENAI_API_KEY=${OPENAI_API_KEY}" \
                -e "OPENAI_BASE_URL=${OPENAI_BASE_URL}" \
                -e "OPENAI_MODEL=${OPENAI_MODEL}" \
                -e "OPENAI_ENABLE_THINKING=${OPENAI_ENABLE_THINKING}" \
                "bash -lc \"${command}\""
        else
            tmux new-window -d -t "${session}:" -n "${task}" "bash -lc \"${command}\""
        fi
    done
done

echo "Started 18 processes in 6 tmux sessions."
echo "Sessions:"
tmux list-sessions -F '#S' | grep "^${SESSION_PREFIX}_" || true
echo "Results: ${OUTPUT_ROOT}"
echo "Logs: ${LOG_ROOT}"
