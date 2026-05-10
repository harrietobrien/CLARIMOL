#!/bin/bash
#
# P2-13: Full model sweep
# Trains and evaluates across multiple model families at 7-8B and 3B scale.
# Uses multi-seed for statistical significance.
#
# Usage: bash scripts/experiment_model_sweep.sh [-- <extra train args>]

set -euo pipefail

SEEDS="42 137 2024"
BASE_OUT="output/model_sweep"
TRAIN_ARGS=("$@")

# ─── Model configurations ────────────────────────────────────────────────────
declare -A MODELS
MODELS=(
    ["llama-8b"]="meta-llama/Llama-3.1-8B-Instruct"
    ["qwen-7b"]="Qwen/Qwen2.5-7B-Instruct"
    ["mistral-7b"]="mistralai/Mistral-7B-Instruct-v0.3"
    ["gemma-9b"]="google/gemma-2-9b-it"
    ["llama-3b"]="meta-llama/Llama-3.2-3B-Instruct"
    ["qwen-3b"]="Qwen/Qwen2.5-3B-Instruct"
)

echo "=== Full Model Sweep ==="
echo "Models: ${!MODELS[*]}"
echo "Seeds: $SEEDS"
echo ""

for MODEL_KEY in "${!MODELS[@]}"; do
    MODEL_NAME="${MODELS[$MODEL_KEY]}"
    MODEL_OUT="${BASE_OUT}/${MODEL_KEY}"

    echo "=== $MODEL_KEY ($MODEL_NAME) ==="

    for SEED in $SEEDS; do
        SEED_DIR="${MODEL_OUT}/seed_${SEED}"
        echo "--- seed=$SEED ---"

        python -m clarimol train \
            --model "$MODEL_NAME" \
            --data-dir data/clarimol \
            --output-dir "$SEED_DIR" \
            --seed "$SEED" \
            "${TRAIN_ARGS[@]}"

        python -m clarimol evaluate \
            --model-path "$SEED_DIR/final" \
            --data-dir data/test \
            --output-file "$SEED_DIR/results.json" \
            --no-unsloth
    done

    echo ""
done

# ─── Aggregate results ───────────────────────────────────────────────────────
echo "=== Aggregating results ==="
python -c "
import json, pathlib, statistics

base = pathlib.Path('$BASE_OUT')
seeds = '$SEEDS'.split()

for model_dir in sorted(base.iterdir()):
    if not model_dir.is_dir():
        continue
    all_accs = {}
    for seed in seeds:
        rpath = model_dir / f'seed_{seed}' / 'results.json'
        if rpath.exists():
            with open(rpath) as f:
                data = json.load(f)
            for task, vals in data.items():
                all_accs.setdefault(task, []).append(vals['accuracy'])

    if not all_accs:
        continue
    print(f'\n{model_dir.name}:')
    for task in sorted(all_accs):
        accs = all_accs[task]
        mean = statistics.mean(accs)
        std = statistics.stdev(accs) if len(accs) > 1 else 0.0
        print(f'  {task:25s}  {mean:.4f} ± {std:.4f}  (n={len(accs)})')
"
