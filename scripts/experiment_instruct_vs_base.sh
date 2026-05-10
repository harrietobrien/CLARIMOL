#!/bin/bash
#
# P3-21: Instruction-tuned vs base model comparison
# Does instruction tuning help or hurt parsing pre-training?
#
# Usage: bash scripts/experiment_instruct_vs_base.sh [-- <extra train args>]

set -euo pipefail

BASE_OUT="output/instruct_vs_base"
TRAIN_ARGS=("$@")
SEED=42

# Model pairs: instruct vs base (same architecture/size)
declare -A INSTRUCT_MODELS
declare -A BASE_MODELS

INSTRUCT_MODELS=(
    ["llama-8b"]="meta-llama/Llama-3.1-8B-Instruct"
    ["qwen-7b"]="Qwen/Qwen2.5-7B-Instruct"
)
BASE_MODELS=(
    ["llama-8b"]="meta-llama/Llama-3.1-8B"
    ["qwen-7b"]="Qwen/Qwen2.5-7B"
)

echo "=== Instruction-Tuned vs Base Model ==="

for KEY in "${!INSTRUCT_MODELS[@]}"; do
    for VARIANT in "instruct" "base"; do
        if [ "$VARIANT" == "instruct" ]; then
            MODEL="${INSTRUCT_MODELS[$KEY]}"
        else
            MODEL="${BASE_MODELS[$KEY]}"
        fi

        OUT_DIR="${BASE_OUT}/${KEY}_${VARIANT}"
        echo "--- $KEY ($VARIANT): $MODEL ---"

        python -m clarimol train \
            --model "$MODEL" \
            --data-dir data/clarimol \
            --output-dir "$OUT_DIR" \
            --seed "$SEED" \
            "${TRAIN_ARGS[@]}"

        python -m clarimol evaluate \
            --model-path "$OUT_DIR/final" \
            --data-dir data/test \
            --output-file "$OUT_DIR/results.json" \
            --no-unsloth
    done
done

echo "=== Instruct vs base comparison complete ==="
echo "Results:"
for KEY in "${!INSTRUCT_MODELS[@]}"; do
    for VARIANT in "instruct" "base"; do
        DIR="${BASE_OUT}/${KEY}_${VARIANT}"
        echo "  ${KEY}_${VARIANT}: $(cat ${DIR}/results.json 2>/dev/null | python -c 'import json,sys; d=json.load(sys.stdin); accs=[v["accuracy"] for v in d.values()]; print(f"mean={sum(accs)/len(accs):.4f}")' 2>/dev/null || echo 'no results')"
    done
done
