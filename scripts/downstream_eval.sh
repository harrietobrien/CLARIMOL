#!/usr/bin/env bash
: <<'COMMENT'
Evaluate fine-tuned downstream models on Mol-Instructions test sets.

Usage:
    bash scripts/downstream_eval.sh <downstream_output_dir>
    bash scripts/downstream_eval.sh output/downstream

Expects model checkpoints at:
    output/downstream/retrosynthesis/final
    output/downstream/reagent_prediction/final
    output/downstream/forward_reaction_prediction/final
COMMENT

set -euo pipefail
cd "$(dirname "$0")/.."

BASE_DIR="${1:?Usage: downstream_eval.sh <downstream_output_dir>}"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export CUDA_VISIBLE_DEVICES

for TASK in retrosynthesis reagent_prediction forward_reaction_prediction; do
    MODEL_PATH="${BASE_DIR}/${TASK}/final"
    if [ ! -d "$MODEL_PATH" ]; then
        echo "No model found at ${MODEL_PATH}, skipping ${TASK}."
        continue
    fi
    echo "Evaluating ${TASK} . . ."
    python -m clarimol downstream-eval \
        --model-path "$MODEL_PATH" \
        --data-dir data/mol_instructions \
        --tasks "$TASK" \
        --output-file "output/results_downstream_${TASK}.json" \
        --batch-size 4
    echo ""
done
