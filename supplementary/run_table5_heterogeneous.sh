#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-MAPR}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/result_heterogeneous_llm}"
LOG_DIR="${LOG_DIR:-${PROJECT_DIR}/logs/heterogeneous_llm}"
TIME_FLAG="${TIME_FLAG:-$(date +%m%d)}"
SESSION="${SESSION:-MAPR_Table5_${TIME_FLAG}}"
RELOAD_DATA="false"

usage() {
    echo "Usage: $0 [--time-flag MMDD] [--reload-data]"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --time-flag)
            [[ $# -ge 2 ]] || { echo "--time-flag requires MMDD" >&2; exit 2; }
            TIME_FLAG="$2"
            SESSION="MAPR_Table5_${TIME_FLAG}"
            shift 2
            ;;
        --reload-data)
            RELOAD_DATA="true"
            shift
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

[[ "${TIME_FLAG}" =~ ^[0-9]{4}$ ]] || { echo "time flag must contain four digits" >&2; exit 2; }
command -v tmux >/dev/null 2>&1 || { echo "tmux is not installed" >&2; exit 1; }
command -v conda >/dev/null 2>&1 || { echo "conda is not installed" >&2; exit 1; }
[[ -f "${PROJECT_DIR}/.env" ]] || { echo "missing ${PROJECT_DIR}/.env" >&2; exit 1; }
tmux has-session -t "${SESSION}" 2>/dev/null && {
    echo "tmux session already exists: ${SESSION}" >&2
    exit 1
}

PAIRS=(
    "qwen3-8b,qwen3-14b"
    "qwen3-8b,llama3-8b"
    "qwen3-14b,llama3-8b"
)
TASKS=(GSM8K StrategyQA)

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

window_index=0
for task in "${TASKS[@]}"; do
  for pair in "${PAIRS[@]}"; do
    window="${task}_${pair//,/_and_}"
    log_file="${LOG_DIR}/${window}_${TIME_FLAG}.log"
    reload_arg=""
    exclude_arg=""
    if [[ "${RELOAD_DATA}" == "true" ]]; then
        reload_arg="--reload-data"
    fi
    if [[ "${task}" == "StrategyQA" ]]; then
        exclude_arg="--exclude-sample StrategyQA:385"
    fi
    command="cd '${PROJECT_DIR}' && conda run --no-capture-output -n '${CONDA_ENV}' python supplementary/heterogeneous_peer_review.py --task '${task}' --models '${pair}' --max-example-num 500 --time-flag '${TIME_FLAG}' --output-dir '${OUTPUT_DIR}' ${exclude_arg} ${reload_arg} 2>&1 | tee -a '${log_file}'"

    if [[ "${window_index}" -eq 0 ]]; then
        tmux new-session -d -s "${SESSION}" -n "${window}" "bash -lc \"${command}\""
    else
        tmux new-window -d -t "${SESSION}:" -n "${window}" "bash -lc \"${command}\""
    fi
    window_index=$((window_index + 1))
  done
done

echo "Started paper Table 5 reproduction in tmux session: ${SESSION}"
echo "Started 6 windows: 3 model pairs x GSM8K/StrategyQA."
echo "Models share one initial-response cache per model, task, and time flag."
echo "Attach: tmux attach -t ${SESSION}"
echo "Results: ${OUTPUT_DIR}"
echo "Logs: ${LOG_DIR}"
