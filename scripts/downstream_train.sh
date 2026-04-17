#!/usr/bin/env bash
: <<'COMMENT'
Stage 2: Fine-tune a pre-trained CLARIMOL model on downstream tasks.
Runs all three Mol-Instructions tasks sequentially.

Usage:
    bash scripts/downstream_train.sh <pretrained_model_path>
    bash scripts/downstream_train.sh output/parsing_pretrain/final

To train on a single task:
    bash scripts/downstream_train.sh output/parsing_pretrain/final retrosynthesis
COMMENT

set -euo pipefail
cd "$(dirname "$0")/.."

MODEL_PATH="${1:?Usage: downstream_train.sh <model_path> [task]}"
SINGLE_TASK="${2:-}"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export CUDA_VISIBLE_DEVICES

TASKS=("retrosynthesis" "reagent_prediction" "forward_reaction_prediction")

if [ -n "$SINGLE_TASK" ]; then
    TASKS=("$SINGLE_TASK")
fi

for TASK in "${TASKS[@]}"; do
    echo "Fine-tuning on ${TASK} . . ."
    python -m clarimol downstream-train \
        --model "$MODEL_PATH" \
        --data-dir data/mol_instructions \
        --task "$TASK" \
        --output-dir "output/downstream/${TASK}" \
        --batch-size 8 \
        --grad-accum 2 \
        --lr 5e-4 \
        --epochs 1 \
        --lora-r 64 \
        --lora-alpha 16 \
        --seed 42
    echo ""
done
