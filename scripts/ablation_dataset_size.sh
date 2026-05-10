#!/bin/bash
#
# P2-17: Dataset size scaling curve
# Tests: 10K, 25K, 50K, 100K, 250K samples per task
#
# Usage: bash scripts/ablation_dataset_size.sh [-- <extra train args>]

set -euo pipefail

SIZES=(10000 25000 50000 100000 250000)
BASE_DATA="data/scaling"
BASE_OUT="output/ablation_dataset_size"
TRAIN_ARGS=("$@")
SEED=42

echo "=== Dataset Size Scaling ==="

for SIZE in "${SIZES[@]}"; do
    DATA_DIR="${BASE_DATA}/keep_${SIZE}"
    OUT_DIR="${BASE_OUT}/size_${SIZE}"

    if [ ! -d "$DATA_DIR" ] || [ -z "$(ls -A "$DATA_DIR" 2>/dev/null)" ]; then
        echo "Preparing data with keep_n=$SIZE"
        python -m clarimol prepare \
            --output-dir "$DATA_DIR" \
            --keep-n "$SIZE" \
            --seed "$SEED"
    fi

    echo "--- Training with $SIZE samples/task ---"
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

echo "=== Dataset size scaling complete ==="
for SIZE in "${SIZES[@]}"; do
    echo "  n=$SIZE: $(cat ${BASE_OUT}/size_${SIZE}/results.json 2>/dev/null | python -c 'import json,sys; d=json.load(sys.stdin); accs=[v["accuracy"] for v in d.values()]; print(f"mean={sum(accs)/len(accs):.4f}")' 2>/dev/null || echo 'no results')"
done
