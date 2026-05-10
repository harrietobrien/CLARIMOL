#!/bin/bash
#
# P3-20: Multi-epoch training with curriculum analysis
# Tests: 1, 2, 3 epochs
# Also compares curriculum vs random across epochs
#
# Usage: bash scripts/ablation_epochs.sh [-- <extra train args>]

set -euo pipefail

EPOCHS=(1 2 3)
ORDERS=("easy-hard" "random")
BASE_OUT="output/ablation_epochs"
TRAIN_ARGS=("$@")
SEED=42

echo "=== Multi-Epoch + Curriculum Ablation ==="

for ORDER in "${ORDERS[@]}"; do
    DATA_DIR="data/clarimol_curriculum/${ORDER}"

    # Prepare data if needed
    if [ ! -d "$DATA_DIR" ] || [ -z "$(ls -A "$DATA_DIR" 2>/dev/null)" ]; then
        echo "Preparing data with curriculum=$ORDER"
        python -m clarimol prepare \
            --output-dir "$DATA_DIR" \
            --curriculum-order "$ORDER" \
            --seed "$SEED"
    fi

    for EP in "${EPOCHS[@]}"; do
        OUT_DIR="${BASE_OUT}/${ORDER}_ep${EP}"
        echo "--- curriculum=$ORDER, epochs=$EP ---"

        python -m clarimol train \
            --data-dir "$DATA_DIR" \
            --output-dir "$OUT_DIR" \
            --epochs "$EP" \
            --seed "$SEED" \
            "${TRAIN_ARGS[@]}"

        python -m clarimol evaluate \
            --model-path "$OUT_DIR/final" \
            --data-dir data/test \
            --output-file "$OUT_DIR/results.json" \
            --no-unsloth
    done
done

echo "=== Multi-epoch ablation complete ==="
echo "Results:"
for ORDER in "${ORDERS[@]}"; do
    for EP in "${EPOCHS[@]}"; do
        DIR="${BASE_OUT}/${ORDER}_ep${EP}"
        echo "  ${ORDER} ep=${EP}: $(cat ${DIR}/results.json 2>/dev/null | python -c 'import json,sys; d=json.load(sys.stdin); accs=[v["accuracy"] for v in d.values()]; print(f"mean={sum(accs)/len(accs):.4f}")' 2>/dev/null || echo 'no results')"
    done
done
