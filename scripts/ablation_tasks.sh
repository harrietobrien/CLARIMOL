#!/bin/bash
#
# P1-9: Task contribution analysis
# Trains models with:
#   - Each single task alone (5 runs)
#   - All-but-one task (5 runs)
#   - All tasks (1 baseline)
#
# Usage: bash scripts/ablation_tasks.sh [-- <extra train args>]

set -euo pipefail

ALL_TASKS=("functional_group" "ring_counting" "chain_length" "canonicalization" "fragment_assembly")
BASE_OUT="output/ablation_tasks"
TRAIN_ARGS=("$@")
SEED=42

echo "=== Task Contribution Analysis ==="

# ─── Single-task models ──────────────────────────────────────────────────────
for TASK in "${ALL_TASKS[@]}"; do
    DATA_DIR="data/single_task_${TASK}"
    OUT_DIR="${BASE_OUT}/single_${TASK}"

    if [ ! -d "$DATA_DIR" ] || [ -z "$(ls -A "$DATA_DIR" 2>/dev/null)" ]; then
        echo "Preparing single-task data: $TASK"
        python -m clarimol prepare \
            --tasks "$TASK" \
            --output-dir "$DATA_DIR" \
            --seed "$SEED"
    fi

    echo "--- Training single task: $TASK ---"
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

# ─── Leave-one-out models ────────────────────────────────────────────────────
for EXCLUDE in "${ALL_TASKS[@]}"; do
    # Build comma-separated list of all tasks except the excluded one
    TASKS=""
    for T in "${ALL_TASKS[@]}"; do
        if [ "$T" != "$EXCLUDE" ]; then
            TASKS="${TASKS:+${TASKS},}${T}"
        fi
    done

    DATA_DIR="data/leave_out_${EXCLUDE}"
    OUT_DIR="${BASE_OUT}/leave_out_${EXCLUDE}"

    if [ ! -d "$DATA_DIR" ] || [ -z "$(ls -A "$DATA_DIR" 2>/dev/null)" ]; then
        echo "Preparing leave-one-out data: excluding $EXCLUDE"
        python -m clarimol prepare \
            --tasks "$TASKS" \
            --output-dir "$DATA_DIR" \
            --seed "$SEED"
    fi

    echo "--- Training without: $EXCLUDE ---"
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

echo "=== Task contribution analysis complete ==="
echo "Results in $BASE_OUT/"
