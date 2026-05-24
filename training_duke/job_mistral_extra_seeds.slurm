#!/bin/bash
#SBATCH --job-name=cm_mistral_5seed
#SBATCH -A scavenger-h200
#SBATCH -p scavenger-h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=24:00:00
#SBATCH --output=output/logs/multi_seed/mistral_extra_seeds_%j.log
#SBATCH --error=output/logs/multi_seed/mistral_extra_seeds_%j.err
#SBATCH --requeue
#
# Mistral-7B extra seeds (7, 99) for 5-seed statistical power.
# Complements existing seeds 42, 137, 2024.
#
# Submit: sbatch training_duke/job_mistral_extra_seeds.slurm

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p output/logs/multi_seed

export CONDARC=/work/gc237/.condarc
export HF_HOME=/work/gc237/.cache/huggingface
export HF_TOKEN=$(cat /work/gc237/.cache/huggingface/token 2>/dev/null || cat ~/.cache/huggingface/token)

module load Anaconda3 2>/dev/null || module load conda 2>/dev/null || true
eval "$(conda shell.bash hook)"
conda activate /work/gc237/conda_envs/clarimol

MODEL="mistralai/Mistral-7B-Instruct-v0.3"
BASE_OUT="output/multi_seed/mistral-7b"
SEEDS=(7 99)

echo "=== Mistral-7B extra seeds ==="
nvidia-smi
echo "Start: $(date)"

for SEED in "${SEEDS[@]}"; do
    SEED_DIR="${BASE_OUT}/seed_${SEED}"

    if [ -f "$SEED_DIR/results.json" ]; then
        echo "SKIP: seed=$SEED already has results.json"
        continue
    fi

    mkdir -p "$SEED_DIR"

    RESUME_FLAG=""
    if ls "$SEED_DIR"/checkpoint-* 1>/dev/null 2>&1; then
        echo "Checkpoint(s) detected for seed=$SEED — resuming"
        RESUME_FLAG="--resume"
    fi

    if [ ! -d "$SEED_DIR/final" ]; then
        echo "--- Training seed=$SEED: $(date) ---"
        python -m clarimol train \
            --model "$MODEL" \
            --data-dir data/clarimol \
            --output-dir "$SEED_DIR" \
            --no-unsloth \
            --max-length 512 \
            --batch-size 16 \
            --grad-accum 1 \
            --lr 1e-4 \
            --epochs 1 \
            --lora-r 64 \
            --lora-alpha 16 \
            --bf16 \
            --no-4bit \
            --no-wandb \
            --save-steps 500 \
            --seed "$SEED" \
            $RESUME_FLAG
    fi

    echo "Evaluating seed=$SEED: $(date)"
    python -m clarimol evaluate \
        --model-path "$SEED_DIR/final" \
        --data-dir data/test \
        --output-file "$SEED_DIR/results.json" \
        --no-unsloth \
        --batch-size 16

    echo "Seed $SEED done at $(date)"
done

echo "=== Complete at $(date) ==="
