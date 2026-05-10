#!/bin/bash
#
# P1-10: Learning rate sweep
# Tests: 1e-4, 2e-4, 5e-4
#
# Usage: bash scripts/ablation_lr.sh [-- <extra train args>]

set -euo pipefail

LRS=("1e-4" "2e-4" "5e-4")
BASE_OUT="output/ablation_lr"
TRAIN_ARGS=("$@")
SEED=42

echo "=== Learning Rate Sweep ==="

for LR in "${LRS[@]}"; do
    OUT_DIR="${BASE_OUT}/lr_${LR}"
    echo "--- LR=$LR ---"

    python -m clarimol train \
        --data-dir data/clarimol \
        --output-dir "$OUT_DIR" \
        --lr "$LR" \
        --seed "$SEED" \
        "${TRAIN_ARGS[@]}"

    python -m clarimol evaluate \
        --model-path "$OUT_DIR/final" \
        --data-dir data/test \
        --output-file "$OUT_DIR/results.json" \
        --no-unsloth
done

echo "=== LR sweep complete ==="
for LR in "${LRS[@]}"; do
    echo "  lr=$LR: $(cat ${BASE_OUT}/lr_${LR}/results.json 2>/dev/null | python -c 'import json,sys; d=json.load(sys.stdin); accs=[v["accuracy"] for v in d.values()]; print(f"mean={sum(accs)/len(accs):.4f}")' 2>/dev/null || echo 'no results')"
done
