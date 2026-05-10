#!/bin/bash
#
# P1-8: Curriculum ordering ablation
# Compares: easy-hard, hard-easy, random, none
#
# Usage: bash scripts/ablation_curriculum.sh [-- <extra train args>]

set -euo pipefail

ORDERS=("easy-hard" "hard-easy" "random" "none")
BASE_DATA="data/clarimol_curriculum"
BASE_OUT="output/ablation_curriculum"
TRAIN_ARGS=("$@")
SEED=42

echo "=== Curriculum Ordering Ablation ==="

# Step 1: Prepare datasets with each ordering
for ORDER in "${ORDERS[@]}"; do
    DATA_DIR="${BASE_DATA}/${ORDER}"
    if [ -d "$DATA_DIR" ] && [ "$(ls -A "$DATA_DIR")" ]; then
        echo "Dataset exists: $DATA_DIR (skipping)"
        continue
    fi
    echo "Preparing dataset with curriculum_order=$ORDER..."
    python -m clarimol prepare \
        --output-dir "$DATA_DIR" \
        --curriculum-order "$ORDER" \
        --seed "$SEED"
done

# Step 2: Train on each
for ORDER in "${ORDERS[@]}"; do
    DATA_DIR="${BASE_DATA}/${ORDER}"
    OUT_DIR="${BASE_OUT}/${ORDER}"
    echo "--- Training with curriculum=$ORDER ---"

    python -m clarimol train \
        --data-dir "$DATA_DIR" \
        --output-dir "$OUT_DIR" \
        --seed "$SEED" \
        "${TRAIN_ARGS[@]}"

    python -m clarimol evaluate \
        --model-path "$OUT_DIR/final" \
        --data-dir data/test \
        --output-file "$OUT_DIR/results.json" \
        --no-unsloth
done

echo "=== Curriculum ablation complete ==="
echo "Compare results:"
for ORDER in "${ORDERS[@]}"; do
    echo "  $ORDER: $(cat ${BASE_OUT}/${ORDER}/results.json 2>/dev/null | python -c 'import json,sys; d=json.load(sys.stdin); print({k:round(v["accuracy"],4) for k,v in d.items()})' 2>/dev/null || echo 'no results')"
done
