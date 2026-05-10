#!/bin/bash
#SBATCH --job-name=cm_tasks
#SBATCH -A scavenger-h200
#SBATCH -p scavenger-h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=48:00:00
#SBATCH --output=output/task_contrib_%j.log
#SBATCH --error=output/task_contrib_%j.err
#SBATCH --requeue
#
# P1-9: Task contribution analysis
# Which parsing tasks actually help downstream performance?
#   - 5 single-task models
#   - 5 leave-one-out models
#   - All evaluated on parsing + downstream tasks
#
# 10 training runs @ ~20min each = ~3-4 hours
#
# Pairs with job_lr_lora_sweep.slurm on the other H200.
#   sbatch training_duke/job_task_contribution.slurm
#   sbatch training_duke/job_lr_lora_sweep.slurm

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

export CONDARC=/work/gc237/.condarc
export HF_HOME=/work/gc237/.cache/huggingface
export HF_TOKEN=$(cat /work/gc237/.cache/huggingface/token)

module load Anaconda3 2>/dev/null || module load conda 2>/dev/null || true
eval "$(conda shell.bash hook)"
conda activate /work/gc237/conda_envs/clarimol

MODEL="meta-llama/Llama-3.1-8B-Instruct"
ALL_TASKS=("functional_group" "ring_counting" "chain_length" "canonicalization" "fragment_assembly")
BASE_OUT="output/ablation_tasks"
SEED=42

# Shared training args
TRAIN_ARGS=(
    --model "$MODEL"
    --no-unsloth --max-length 512
    --batch-size 16 --grad-accum 1 --lr 1e-4 --epochs 1
    --lora-r 64 --lora-alpha 16
    --bf16 --no-4bit --no-wandb --save-steps 500
    --seed "$SEED"
)

echo "=== Task Contribution Analysis ==="
nvidia-smi
echo "Start: $(date)"

# ─── Single-task models ──────────────────────────────────────────────────────
for TASK in "${ALL_TASKS[@]}"; do
    DATA_DIR="data/single_task_${TASK}"
    OUT_DIR="${BASE_OUT}/single_${TASK}"

    # Prepare single-task dataset
    if [ ! -d "$DATA_DIR" ] || [ -z "$(ls -A "$DATA_DIR" 2>/dev/null)" ]; then
        echo "Preparing single-task data: $TASK"
        python -m clarimol prepare \
            --tasks "$TASK" \
            --output-dir "$DATA_DIR" \
            --seed "$SEED"
    fi

    mkdir -p "$OUT_DIR"
    echo ""
    echo "=== Single task: $TASK ==="
    echo "Start: $(date)"

    python -m clarimol train \
        --data-dir "$DATA_DIR" \
        --output-dir "$OUT_DIR" \
        "${TRAIN_ARGS[@]}"

    python -m clarimol evaluate \
        --model-path "$OUT_DIR/final" \
        --data-dir data/test \
        --output-file "$OUT_DIR/results.json" \
        --no-unsloth --batch-size 16

    echo "Done: $(date)"
done

# ─── Leave-one-out models ────────────────────────────────────────────────────
for EXCLUDE in "${ALL_TASKS[@]}"; do
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

    mkdir -p "$OUT_DIR"
    echo ""
    echo "=== Leave-one-out: excluding $EXCLUDE ==="
    echo "Start: $(date)"

    python -m clarimol train \
        --data-dir "$DATA_DIR" \
        --output-dir "$OUT_DIR" \
        "${TRAIN_ARGS[@]}"

    python -m clarimol evaluate \
        --model-path "$OUT_DIR/final" \
        --data-dir data/test \
        --output-file "$OUT_DIR/results.json" \
        --no-unsloth --batch-size 16

    echo "Done: $(date)"
done

echo ""
echo "=== Task contribution analysis complete at $(date) ==="
echo ""
echo "Single-task results:"
for TASK in "${ALL_TASKS[@]}"; do
    echo -n "  $TASK: "
    python -c "
import json
with open('${BASE_OUT}/single_${TASK}/results.json') as f:
    d = json.load(f)
accs = [v['accuracy'] for v in d.values()]
print(f'mean={sum(accs)/len(accs):.4f}')
" 2>/dev/null || echo "no results"
done
echo ""
echo "Leave-one-out results:"
for TASK in "${ALL_TASKS[@]}"; do
    echo -n "  w/o $TASK: "
    python -c "
import json
with open('${BASE_OUT}/leave_out_${TASK}/results.json') as f:
    d = json.load(f)
accs = [v['accuracy'] for v in d.values()]
print(f'mean={sum(accs)/len(accs):.4f}')
" 2>/dev/null || echo "no results"
done
