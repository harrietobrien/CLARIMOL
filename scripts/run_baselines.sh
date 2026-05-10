#!/bin/bash
#
# Run baseline evaluations for publication:
#   1. Zero-shot (untrained) base models on parsing tasks
#   2. Downstream-only (no parsing pre-training) for ablation
#
# Usage:
#   bash scripts/run_baselines.sh [--test-dir data/test] [--no-unsloth]
#
# Results saved to output/baselines/

set -euo pipefail

TEST_DIR="data/test"
DOWNSTREAM_DIR="data/mol_instructions"
EXTRA_FLAGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --test-dir) TEST_DIR="$2"; shift 2 ;;
        --downstream-dir) DOWNSTREAM_DIR="$2"; shift 2 ;;
        *) EXTRA_FLAGS+=("$1"); shift ;;
    esac
done

BASE_OUT="output/baselines"
mkdir -p "$BASE_OUT"

# ─── 1. Zero-shot parsing evaluation ─────────────────────────────────────────
# Evaluate untrained instruction-tuned models on parsing tasks
MODELS=(
    "meta-llama/Llama-3.1-8B-Instruct"
    "Qwen/Qwen2.5-7B-Instruct"
)

echo "=== Zero-shot baseline evaluation ==="
for MODEL in "${MODELS[@]}"; do
    # Derive a short name for output directory
    SHORT_NAME=$(echo "$MODEL" | sed 's|.*/||' | tr '[:upper:]' '[:lower:]')
    OUT_DIR="$BASE_OUT/zero_shot_${SHORT_NAME}"
    mkdir -p "$OUT_DIR"
    echo "--- $MODEL → $OUT_DIR ---"

    python -m clarimol evaluate \
        --model-path "$MODEL" \
        --data-dir "$TEST_DIR" \
        --output-file "$OUT_DIR/results.json" \
        --batch-size 4 \
        --max-samples 2000 \
        --no-unsloth \
        "${EXTRA_FLAGS[@]}" \
        2>&1 | tee "$OUT_DIR/eval.log"

    echo ""
done

# ─── 2. Downstream-only baselines (skip parsing pre-training) ────────────────
# Fine-tune directly on downstream tasks from the base model
DOWNSTREAM_TASKS=("retrosynthesis" "reagent_prediction" "forward_reaction_prediction")
BASE_MODEL="meta-llama/Llama-3.1-8B-Instruct"

echo "=== Downstream-only baselines (no parsing pre-training) ==="
for TASK in "${DOWNSTREAM_TASKS[@]}"; do
    OUT_DIR="$BASE_OUT/downstream_only_${TASK}"
    mkdir -p "$OUT_DIR"
    echo "--- $TASK (base model, no parsing) → $OUT_DIR ---"

    python -m clarimol downstream-train \
        --model "$BASE_MODEL" \
        --task "$TASK" \
        --data-dir "$DOWNSTREAM_DIR" \
        --output-dir "$OUT_DIR" \
        --batch-size 8 \
        --grad-accum 2 \
        --lr 5e-4 \
        --epochs 1 \
        --no-unsloth \
        --no-wandb \
        "${EXTRA_FLAGS[@]}" \
        2>&1 | tee "$OUT_DIR/train.log"

    echo "Evaluating..."
    python -m clarimol downstream-eval \
        --model-path "$OUT_DIR/final" \
        --data-dir "$DOWNSTREAM_DIR" \
        --tasks "$TASK" \
        --output-file "$OUT_DIR/results.json" \
        --no-unsloth \
        --max-samples 2000 \
        2>&1 | tee -a "$OUT_DIR/eval.log"

    echo ""
done

echo "=== All baselines complete ==="
echo "Results in $BASE_OUT/"
