#!/bin/bash
#
# P2-16: LoRA rank and alpha ablation
# Tests rank ∈ {16, 32, 64, 128} with alpha ∈ {rank, 2*rank}
#
# Usage: bash scripts/ablation_lora.sh [-- <extra train args>]

set -euo pipefail

BASE_OUT="output/ablation_lora"
TRAIN_ARGS=("$@")
SEED=42

echo "=== LoRA Rank/Alpha Ablation ==="

# Sweep rank with alpha=rank (common setting)
for RANK in 16 32 64 128; do
    for ALPHA_MULT in 1 2; do
        ALPHA=$((RANK * ALPHA_MULT))
        OUT_DIR="${BASE_OUT}/r${RANK}_a${ALPHA}"
        echo "--- rank=$RANK, alpha=$ALPHA ---"

        python -m clarimol train \
            --data-dir data/clarimol \
            --output-dir "$OUT_DIR" \
            --lora-r "$RANK" \
            --lora-alpha "$ALPHA" \
            --seed "$SEED" \
            "${TRAIN_ARGS[@]}"

        python -m clarimol evaluate \
            --model-path "$OUT_DIR/final" \
            --data-dir data/test \
            --output-file "$OUT_DIR/results.json" \
            --no-unsloth
    done
done

echo "=== LoRA ablation complete ==="
echo "Results in $BASE_OUT/"
