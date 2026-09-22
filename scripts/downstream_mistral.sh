#!/bin/bash
#SBATCH --job-name=cm_ds_mistral
#SBATCH -A scavenger-h200
#SBATCH -p scavenger-h200
#SBATCH --gres=gpu:h200:1
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=output/logs/downstream/downstream_mistral_%j.log
#SBATCH --error=output/logs/downstream/downstream_mistral_%j.err
#
# Downstream transfer on Mistral-7B: train + eval on all 3 Mol-Instructions
# tasks (retrosynthesis, reagent prediction, forward reaction prediction).
#
# Two conditions:
#   1. CLARIMOL-pretrained Mistral -> downstream fine-tuning
#   2. Direct Mistral -> downstream fine-tuning (no CLARIMOL pretraining)
#
# Mirrors the existing LLaMA-8B downstream experiment for cross-model comparison.
#
# Submit: sbatch scripts/downstream_mistral.sh

set -euo pipefail
cd ~/storage/CLARIMOL
mkdir -p output/logs/downstream

export CONDARC=/work/gc237/.condarc
export HF_HOME=/work/gc237/.cache/huggingface
export HF_TOKEN=$(cat /work/gc237/.cache/huggingface/token 2>/dev/null || cat ~/.cache/huggingface/token)

source /opt/apps/rhel9/Anaconda3-2024.02/etc/profile.d/conda.sh
conda activate /work/gc237/conda_envs/clarimol

echo "=== Downstream Transfer: Mistral-7B ==="
nvidia-smi
echo "Start: $(date)"

MISTRAL_PRETRAINED="output/multi_seed/mistral-7b/seed_42/final"
TASKS=("retrosynthesis" "reagent_prediction" "forward_reaction_prediction")

# Condition 1: CLARIMOL-pretrained Mistral -> downstream
echo "=== Condition 1: CLARIMOL-pretrained Mistral ==="
OUT_PRETRAINED="output/downstream_mistral_pretrained"

for TASK in "${TASKS[@]}"; do
    TASK_OUT="$OUT_PRETRAINED/$TASK"
    if [ -f "$TASK_OUT/results.json" ]; then
        echo "Skipping $TASK (pretrained): results already exist"
        continue
    fi
    echo "--- Training $TASK (pretrained) ---"
    python -m clarimol downstream-train --no-unsloth \
        --model "$MISTRAL_PRETRAINED" \
        --data-dir data/mol_instructions \
        --task "$TASK" \
        --output-dir "$TASK_OUT" \
        --batch-size 8 \
        --grad-accum 2 \
        --lr 5e-4 \
        --epochs 1 \
        --lora-r 64 \
        --lora-alpha 16 \
        --seed 42

    echo "--- Evaluating $TASK (pretrained) ---"
    python -m clarimol downstream-eval --no-unsloth \
        --model-path "$TASK_OUT/final" \
        --data-dir data/mol_instructions \
        --tasks "$TASK" \
        --output-file "$TASK_OUT/results.json" \
        --batch-size 4
done

# Condition 2: Direct Mistral (no CLARIMOL pretraining) -> downstream
echo "=== Condition 2: Direct Mistral (no pretraining) ==="
OUT_DIRECT="output/downstream_mistral_direct"

for TASK in "${TASKS[@]}"; do
    TASK_OUT="$OUT_DIRECT/$TASK"
    if [ -f "$TASK_OUT/results.json" ]; then
        echo "Skipping $TASK (direct): results already exist"
        continue
    fi
    echo "--- Training $TASK (direct) ---"
    python -m clarimol downstream-train --no-unsloth \
        --model "mistralai/Mistral-7B-Instruct-v0.3" \
        --data-dir data/mol_instructions \
        --task "$TASK" \
        --output-dir "$TASK_OUT" \
        --batch-size 8 \
        --grad-accum 2 \
        --lr 5e-4 \
        --epochs 1 \
        --lora-r 64 \
        --lora-alpha 16 \
        --seed 42

    echo "--- Evaluating $TASK (direct) ---"
    python -m clarimol downstream-eval --no-unsloth \
        --model-path "$TASK_OUT/final" \
        --data-dir data/mol_instructions \
        --tasks "$TASK" \
        --output-file "$TASK_OUT/results.json" \
        --batch-size 4
done

echo "=== Downstream Mistral complete ==="
echo "End: $(date)"

# Print summary
echo ""
echo "=== SUMMARY ==="
for TASK in "${TASKS[@]}"; do
    echo "--- $TASK ---"
    echo "  Pretrained:"
    cat "$OUT_PRETRAINED/$TASK/results.json" 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print(f'    exact={d.get(\"exact_match\",\"?\")}, validity={d.get(\"validity\",\"?\")}')" 2>/dev/null || echo "    (no results)"
    echo "  Direct:"
    cat "$OUT_DIRECT/$TASK/results.json" 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print(f'    exact={d.get(\"exact_match\",\"?\")}, validity={d.get(\"validity\",\"?\")}')" 2>/dev/null || echo "    (no results)"
done
