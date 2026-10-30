#!/usr/bin/env bash
# Plain HOICLIP cross-dataset transfer: VidHOI -> Action Genome.
#
# Direct counterpart of train_vidhoi.sh (same model_name=HOICLIP, same architecture),
# just pointed at the AG dataset. By default PRETRAINED loads a HOICLIP checkpoint
# trained on VidHOI: architecture-matching weights (backbone/transformer/hoi decoder)
# transfer, while the HOI-label-space-sized heads (visual_projection,
# obj_visual_projection, verb_projection, verb2hoi_proj) are shape-mismatched against
# AG's 538-triplet / 26-verb / 36-obj vocabulary and get dropped automatically by
# main.py, which keeps their build-time CLIP-prompt initialisation for AG instead.
#
# Usage:
#   bash train_ag.sh
#   PRETRAINED=/path/to/hoiclip_vidhoi_checkpoint.pth bash train_ag.sh

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
MASTER_PORT="${MASTER_PORT:-29506}"
DATA_ROOT="${DATA_ROOT:-../AG}"
# A plain HOICLIP checkpoint (any dataset — shape-mismatched heads are dropped and
# re-initialised from AG's own CLIP prompts, see main.py).
PRETRAINED="${PRETRAINED:-output/hoiclip/hoiclip_05301154/checkpoint_best.pth}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./output}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LR_DROP="${LR_DROP:-30}"
BACKBONE="${BACKBONE:-resnet50}"
RESUME="${RESUME:-"/home/yluo/HOICLIP/output/hoiclip_ag/hoiclip_ag_07292128/checkpoint_last.pth"}"
TRAIN_RATIO="${TRAIN_RATIO:-1.0}"
# Empty by default: enables only the final mAP eval.
# Set to a path (e.g. ./training_free_ehnahcement/) to also run the topk
# hyperparameter search loop (adds 6 extra full-dataset evaluations — slow).
ENHANCEMENT_PATH="${ENHANCEMENT_PATH:-}"

# Run name for organising output directories — mirrors RLIPVISTR's --name convention.
NAME="${NAME:-hoiclip_ag}"

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

if [ ! -f "${DATA_ROOT}/AG_annotation/train_frame_hoia.json" ]; then
    echo "Missing ${DATA_ROOT}/AG_annotation/train_frame_hoia.json"
    exit 1
fi

python -m torch.distributed.run \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_port="${MASTER_PORT}" \
    main.py \
    --output_dir "${OUTPUT_DIR}" \
    --name "${NAME}" \
    --dataset_file ag \
    --hoi_path "${DATA_ROOT}" \
    --num_obj_classes 36 \
    --num_verb_classes 26 \
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
    --model_name HOICLIP \
    --zero_shot_type default \
    --verb_pth ./tmp/verb.pth \
    ${ENHANCEMENT_PATH:+--training_free_enhancement_path "${ENHANCEMENT_PATH}"} \
    --train_ratio "${TRAIN_RATIO}" \
    --val_ratio "${VAL_RATIO:-0.4}" \
    ${RESUME:+--resume "${RESUME}"} \
    "$@"
