#!/usr/bin/env bash
# Eval + profiling script for HOICLIP_LSTM on VidHOI. Wraps train_lstm_vidhoi.sh (same model
# flags) and appends `--eval`.
#
# Usage:
#   bash eval_lstm_vidhoi.sh /path/to/checkpoint.pth [extra main.py args]
#   CHECKPOINT=/path/to/checkpoint.pth MONITOR_GPU=1 bash eval_lstm_vidhoi.sh
#
# engine.py::evaluate_hoi (EVAL_PROFILE=1) prints params, GFLOPs, "Peak GPU memory"
# (allocated/reserved) and avg inference time / FPS. BATCH_SIZE defaults to 1 so the
# time is per sample (one clip of max_length frames).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_lstm_vidhoi.sh"

# RANDOM_INIT=1: no checkpoint, randomly initialised weights (CLIP is still loaded by clip.load at
# build time). Params / GFLOPs / memory do not depend on weight values; mAP is meaningless, so
# only EVAL_MAX_BATCHES batches are run.
RANDOM_INIT="${RANDOM_INIT:-0}"
export RANDOM_INIT
CHECKPOINT="${CHECKPOINT:-}"
if [ "$#" -gt 0 ] && [ "${RANDOM_INIT}" != "1" ]; then
    CHECKPOINT="$1"
    shift
fi
if [ "${RANDOM_INIT}" != "1" ] && [ -z "${CHECKPOINT}" ]; then
    echo "Usage: bash eval_lstm_vidhoi.sh /path/to/checkpoint.pth [extra-main.py-args]"
    exit 1
fi
if [ "${RANDOM_INIT}" != "1" ] && [ -d "${CHECKPOINT}" ]; then
    for f in checkpoint_best.pth checkpoint.pth; do
        [ -f "${CHECKPOINT}/${f}" ] && CHECKPOINT="${CHECKPOINT}/${f}" && break
    done
fi
if [ "${RANDOM_INIT}" != "1" ] && [ ! -f "${CHECKPOINT}" ]; then
    echo "Missing checkpoint: ${CHECKPOINT}"
    exit 1
fi

export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
export MASTER_PORT="${MASTER_PORT:-29603}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
# The checkpoint comes in via RESUME; an empty PRETRAINED keeps train script from also passing --pretrained.
export PRETRAINED=""
export RESUME="${CHECKPOINT}"
if [ "${RANDOM_INIT}" = "1" ]; then
    export RESUME=""
    # first batches are CUDA/cuDNN warm-up and are excluded from the timing
    export EVAL_WARMUP_BATCHES="${EVAL_WARMUP_BATCHES:-10}"
    set -- --eval_max_batches "${EVAL_MAX_BATCHES:-300}" "$@"
fi
export EVAL_PROFILE="${EVAL_PROFILE:-1}"

MEM_LOG=""
if [ "${MONITOR_GPU:-0}" = "1" ]; then
    MEM_LOG="$(mktemp)"
    # Sample used memory (MiB) of the first visible GPU every 0.5s.
    ( while true; do
          nvidia-smi -i "${CUDA_VISIBLE_DEVICES:-0}" --query-gpu=memory.used --format=csv,noheader,nounits >> "${MEM_LOG}"
          sleep 0.5
      done ) &
    MONITOR_PID=$!
    trap 'kill ${MONITOR_PID} 2>/dev/null || true' EXIT
fi

bash "${TRAIN_SCRIPT}" --eval "$@"

if [ -n "${MEM_LOG}" ]; then
    kill "${MONITOR_PID}" 2>/dev/null || true
    echo "Peak GPU memory used (MiB, nvidia-smi): $(sort -n "${MEM_LOG}" | tail -1)"
    rm -f "${MEM_LOG}"
fi
