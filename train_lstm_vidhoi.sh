#!/usr/bin/env bash

set -euo pipefail

# Make `conda activate` work in non-interactive scripts.
if [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
fi

conda activate rlip2

NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
MASTER_PORT="${MASTER_PORT:-29500}"
DATA_ROOT="${DATA_ROOT:-../VidHOI}"
PRETRAINED="${PRETRAINED:-output/hoiclip/hoiclip_05301154/checkpoint_best.pth}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./output}"
# `${PRETRAINED:-default}` also replaces an empty value, so RANDOM_INIT must clear it explicitly.
[ "${RANDOM_INIT:-0}" = "1" ] && PRETRAINED=""
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LR_DROP="${LR_DROP:-30}"
BACKBONE="${BACKBONE:-resnet50}"
RESUME="${RESUME:-}"
TRAIN_RATIO="${TRAIN_RATIO:-1.0}"
ANTICIPATION_HOI_COEF="${ANTICIPATION_HOI_COEF:-0.8}"
ANTICIPATION_HORIZON_GAMMA="${ANTICIPATION_HORIZON_GAMMA:-0.7}"
ANTICIPATION_RAMP_EPOCHS="${ANTICIPATION_RAMP_EPOCHS:-8}"
ANTICIPATION_RAMP_START_FACTOR="${ANTICIPATION_RAMP_START_FACTOR:-0.25}"
DETECTION_FRAME_WEIGHTING="${DETECTION_FRAME_WEIGHTING:-linear}"
DETECTION_FRAME_WEIGHT_POWER="${DETECTION_FRAME_WEIGHT_POWER:-1.0}"

# Run name for organising output directories — mirrors RLIPVISTR's --name convention.
NAME="${NAME:-hoiclip}"

# Output directory logic (mirrors RLIPVISTR/main.py):
#   fresh run  → OUTPUT_ROOT/NAME  (HOICLIP's main.py then appends /MMDDHHMI)
#   resume     → grandparent of checkpoint, so main.py's new timestamp dir is a
#                sibling of the original run rather than nested inside it
if [ -n "${RESUME}" ]; then
    OUTPUT_DIR="$(dirname "$(dirname "${RESUME}")")"
else
    OUTPUT_DIR="${OUTPUT_ROOT}/${NAME}"
fi

if [ ! -e "${DATA_ROOT}/images" ]; then
    echo "Expected ${DATA_ROOT}/images to exist."
    exit 1
fi

if [ ! -f "${DATA_ROOT}/VidHOI_annotation/train_frame_hoia.json" ]; then
    echo "Missing ${DATA_ROOT}/VidHOI_annotation/train_frame_hoia.json"
    exit 1
fi

if [ -n "${RESUME}" ] && [ ! -f "${RESUME}" ]; then
    echo "Missing resume checkpoint: ${RESUME}"
    exit 1
fi

# RANDOM_INIT=1: efficiency-only eval with randomly initialised weights, no checkpoint needed.
if [ "${RANDOM_INIT:-0}" != "1" ] && [ -z "${RESUME}" ] && [ ! -f "${PRETRAINED}" ]; then
    echo "Missing pretrained checkpoint: ${PRETRAINED}"
    exit 1
fi

python -m torch.distributed.run \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_port="${MASTER_PORT}" \
    main.py \
    --output_dir "${OUTPUT_DIR}" \
    --name "${NAME}" \
    --dataset_file vidhoi \
    --hoi_path "${DATA_ROOT}" \
    --num_obj_classes 78 \
    --num_verb_classes 50 \
    --backbone "${BACKBONE}" \
    --num_queries 64 \
    --dec_layers 3 \
    --epochs "${EPOCHS}" \
    --lr_drop "${LR_DROP}" \
    --use_nms_filter \
    --fix_clip \
    --batch_size "${BATCH_SIZE}" \
    ${PRETRAINED:+--pretrained "${PRETRAINED}"} \
    --with_clip_label \
    --with_obj_clip_label \
    --gradient_accumulation_steps 1 \
    --num_workers 4 \
    --opt_sched "multiStep" \
    --dataset_root GEN \
    --model_name HOICLIP_LSTM \
    --zero_shot_type default \
    --verb_pth ./tmp/verb.pth \
    --training_free_enhancement_path "" \
    --train_ratio "${TRAIN_RATIO}" \
    --max_length 6 \
    --min_length 6 \
    --min_future 3 \
    --sample_mode window \
    --lstm_hidden_dim 512 \
    --lstm_num_layers 2 \
    --anticipation_hoi_coef "${ANTICIPATION_HOI_COEF}" \
    --anticipation_horizon_gamma "${ANTICIPATION_HORIZON_GAMMA}" \
    --anticipation_ramp_epochs "${ANTICIPATION_RAMP_EPOCHS}" \
    --anticipation_ramp_start_factor "${ANTICIPATION_RAMP_START_FACTOR}" \
    --detection_frame_weighting "${DETECTION_FRAME_WEIGHTING}" \
    --detection_frame_weight_power "${DETECTION_FRAME_WEIGHT_POWER}" \
    ${RESUME:+--resume "${RESUME}"} \
    "$@"
