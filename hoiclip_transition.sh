#!/usr/bin/env bash
# Evaluate HOICLIP with a triplet transition matrix for future-horizon HOI anticipation.
# Model architecture params mirror train_vidhoi.sh exactly.
#
# The model stays image-based: the dataset yields window clips, and evaluate_hoi_transition
# feeds only the clip's last frame to the model, then propagates the predicted HOI-triplet
# scores to future horizons [1,3,5,7] via the transition matrix.
#
# The matrix lives in datasets/priors/transition_matrix_triplet.py and is in HOI-triplet
# space ([1127, 1127], i.e. len(vidhoi_text_label)), NOT verb space. Until the generated
# constants are pasted in, it falls back to identity (= no transition).
#
# Usage: bash hoiclip_transition.sh [extra args]
#   RESUME=/path/to/ckpt.pth bash hoiclip_transition.sh
#   TRANSITION_MATRIX_TYPE=detant_v2 bash hoiclip_transition.sh

set -euo pipefail

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

NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_PORT="${MASTER_PORT:-29503}"
DATA_ROOT="${DATA_ROOT:-../VidHOI}"
OUTPUT_ROOT="${OUTPUT_ROOT:-../DetAnt1/output}"
NAME="${NAME:-hoiclip_transition}"
RESUME="${RESUME:-output/hoiclip/hoiclip_05301154/checkpoint_best.pth}"
BACKBONE="${BACKBONE:-resnet50}"
TRANSITION_MATRIX_TYPE="${TRANSITION_MATRIX_TYPE:-detant}"
MIN_LENGTH="${MIN_LENGTH:-6}"
MAX_LENGTH="${MAX_LENGTH:-6}"
MIN_FUTURE="${MIN_FUTURE:-3}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
VAL_RATIO="${VAL_RATIO:-1.0}"

if [ -z "${RESUME}" ]; then
    echo "Set RESUME=/path/to/checkpoint.pth"
    exit 1
fi

if [ ! -f "${RESUME}" ]; then
    echo "Missing checkpoint: ${RESUME}"
    exit 1
fi

if [ ! -e "${DATA_ROOT}/images" ]; then
    echo "Expected ${DATA_ROOT}/images to exist."
    exit 1
fi

OUTPUT_DIR="${OUTPUT_ROOT}/${NAME}"

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
    --use_nms_filter \
    --fix_clip \
    --batch_size "${BATCH_SIZE}" \
    --with_clip_label \
    --with_obj_clip_label \
    --gradient_accumulation_steps 1 \
    --num_workers "${NUM_WORKERS}" \
    --dataset_root GEN \
    --model_name HOICLIP \
    --zero_shot_type default \
    --verb_pth ./tmp/verb.pth \
    --sample_mode window \
    --min_length "${MIN_LENGTH}" \
    --max_length "${MAX_LENGTH}" \
    --min_future "${MIN_FUTURE}" \
    --val_ratio "${VAL_RATIO}" \
    --transition_matrix_type "${TRANSITION_MATRIX_TYPE}" \
    --eval \
    --resume "${RESUME}" \
    "$@"
