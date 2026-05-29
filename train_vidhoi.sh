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
MASTER_PORT="${MASTER_PORT:-29502}"
DATA_ROOT="${DATA_ROOT:-/mnt/d/VidHOI}"
PRETRAINED="${PRETRAINED:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-output}"
EPOCHS="${EPOCHS:-90}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LR_DROP="${LR_DROP:-60}"
BACKBONE="${BACKBONE:-resnet50}"
RESUME="${RESUME:-}"
# Output directory for this experiment
EXP_DIR="${EXP_DIR:-${OUTPUT_ROOT}/vidhoi_hoiclip}"
# Fraction of training data to use
TRAIN_RATIO="${TRAIN_RATIO:-1.0}"

if [ ! -e "${DATA_ROOT}/images" ]; then
    echo "Expected ${DATA_ROOT}/images to exist."
    exit 1
fi

if [ ! -f "${DATA_ROOT}/VidHOI_annotation/train_frame_hoia.json" ]; then
    echo "Missing ${DATA_ROOT}/VidHOI_annotation/train_frame_hoia.json"
    exit 1
fi

python -m torch.distributed.run \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_port="${MASTER_PORT}" \
    main.py \
    --output_dir "${EXP_DIR}" \
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
    --pretrained "${PRETRAINED}" \
    --with_clip_label \
    --with_obj_clip_label \
    --gradient_accumulation_steps 1 \
    --num_workers 4 \
    --opt_sched "multiStep" \
    --dataset_root GEN \
    --model_name HOICLIP \
    --zero_shot_type default \
    --verb_pth ./tmp/verb.pth \
    --training_free_enhancement_path ./training_free_ehnahcement/ \
    --train_ratio "${TRAIN_RATIO}" \
