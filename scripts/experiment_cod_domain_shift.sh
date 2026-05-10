#!/bin/bash
#
# P1-11: COD domain shift experiment
# 1. Evaluate ZINC-trained model on COD test set
# 2. Fine-tune on COD and re-evaluate
# 3. Compare: does ZINC transfer to crystal molecules?
#
# Usage: bash scripts/experiment_cod_domain_shift.sh <zinc_model_path> [-- <extra args>]

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <zinc_model_path> [-- <extra train args>]"
    echo "  zinc_model_path: Path to model pre-trained on ZINC250K"
    exit 1
fi

ZINC_MODEL="$1"
shift
TRAIN_ARGS=("$@")
COD_DATA="data/cod"
COD_TRAIN="data/cod_train"
COD_TEST="data/cod_test"
BASE_OUT="output/cod_domain_shift"
SEED=42

mkdir -p "$BASE_OUT"

echo "=== COD Domain Shift Experiment ==="

# ─── Step 1: Prepare COD dataset (if needed) ─────────────────────────────────
if [ ! -d "$COD_DATA" ] || [ -z "$(ls -A "$COD_DATA" 2>/dev/null)" ]; then
    echo "Preparing COD dataset (5000 molecules)..."
    python -m clarimol prepare \
        --source cod \
        --max-molecules 5000 \
        --output-dir "$COD_DATA" \
        --seed "$SEED"
fi

# ─── Step 2: Evaluate ZINC-trained model on COD ──────────────────────────────
echo "--- Evaluating ZINC-trained model on COD ---"
python -m clarimol evaluate \
    --model-path "$ZINC_MODEL" \
    --data-dir "$COD_DATA" \
    --output-file "$BASE_OUT/zinc_on_cod.json" \
    --no-unsloth

# ─── Step 3: Fine-tune on COD ────────────────────────────────────────────────
echo "--- Fine-tuning on COD ---"
python -m clarimol train \
    --data-dir "$COD_DATA" \
    --output-dir "$BASE_OUT/cod_finetuned" \
    --seed "$SEED" \
    "${TRAIN_ARGS[@]}"

# ─── Step 4: Evaluate COD-fine-tuned model ───────────────────────────────────
echo "--- Evaluating COD-finetuned model on COD ---"
python -m clarimol evaluate \
    --model-path "$BASE_OUT/cod_finetuned/final" \
    --data-dir "$COD_DATA" \
    --output-file "$BASE_OUT/cod_on_cod.json" \
    --no-unsloth

echo "--- Evaluating COD-finetuned model on ZINC (regression check) ---"
python -m clarimol evaluate \
    --model-path "$BASE_OUT/cod_finetuned/final" \
    --data-dir data/test \
    --output-file "$BASE_OUT/cod_on_zinc.json" \
    --no-unsloth

echo "=== COD domain shift complete ==="
echo "Results in $BASE_OUT/"
echo "  zinc_on_cod.json  — ZINC model evaluated on crystal molecules"
echo "  cod_on_cod.json   — COD-finetuned model on crystal molecules"
echo "  cod_on_zinc.json  — COD-finetuned model on ZINC (regression)"
