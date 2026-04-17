#!/usr/bin/env bash
: <<'COMMENT'
Evaluate trained model on both ZINC test set and COD molecules
    e.g., bash scripts/evaluate_all.sh output/parsing_pretrain/final
COMMENT

set -euo pipefail
cd "$(dirname "$0")/.."

MODEL_PATH="${1:?Usage: evaluate_all.sh <model_path>}"
echo "Evaluating on ZINC250K test set . . ."
shift
python -m clarimol evaluate \
    --model-path "$MODEL_PATH" \
    --data-dir data/test \
    --output-file output/results_zinc_test.json \
    "$@"

echo ""
echo "Evaluating on COD crystal molecules . . ."
if [ -d data/cod ]; then
    python -m clarimol evaluate \
        --model-path "$MODEL_PATH" \
        --data-dir data/cod \
        --output-file output/results_cod.json \
        "$@"
else
    echo "No COD dataset found at data/cod — skipping."
fi
