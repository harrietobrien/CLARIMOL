#!/bin/bash
#SBATCH --job-name=cm_mistral_lr
#SBATCH -A scavenger-h200
#SBATCH -p scavenger-h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=24:00:00
#SBATCH --output=output/logs/lr_lora/mistral_lr_%j.log
#SBATCH --error=output/logs/lr_lora/mistral_lr_%j.err
#SBATCH --requeue
#
# Mistral-7B at lr=2e-4 and lr=5e-4 (best model x best LRs)
# Current multi-seed used lr=1e-4. This tests if the LR finding generalizes.
#
# Submit: sbatch training_duke/job_mistral_lr_sweep.slurm

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p output/logs/lr_lora

export CONDARC=/work/gc237/.condarc
export HF_HOME=/work/gc237/.cache/huggingface
export HF_TOKEN=$(cat /work/gc237/.cache/huggingface/token 2>/dev/null || cat ~/.cache/huggingface/token)

module load Anaconda3 2>/dev/null || module load conda 2>/dev/null || true
eval "$(conda shell.bash hook)"
conda activate /work/gc237/conda_envs/clarimol

MODEL="mistralai/Mistral-7B-Instruct-v0.3"
BASE_OUT="output/ablation_lr_mistral"
SEED=42

echo "=== Mistral-7B LR sweep ==="
nvidia-smi
echo "Start: $(date)"

for LR in 2e-4 5e-4; do
    LR_DIR="${BASE_OUT}/lr_${LR}"

    if [ -f "$LR_DIR/results.json" ]; then
        echo "SKIP: $LR_DIR/results.json exists"
        continue
    fi

    mkdir -p "$LR_DIR"

    RESUME_FLAG=""
    if ls "$LR_DIR"/checkpoint-* 1>/dev/null 2>&1; then
        echo "Checkpoint(s) detected for lr=$LR — resuming"
        RESUME_FLAG="--resume"
    fi

    if [ ! -d "$LR_DIR/final" ]; then
        echo "--- Training lr=$LR: $(date) ---"
        python -m clarimol train \
            --model "$MODEL" \
            --data-dir data/clarimol \
            --output-dir "$LR_DIR" \
            --no-unsloth \
            --max-length 512 \
            --batch-size 16 \
            --grad-accum 1 \
            --lr "$LR" \
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

    echo "Evaluating lr=$LR: $(date)"
    python -m clarimol evaluate \
        --model-path "$LR_DIR/final" \
        --data-dir data/test \
        --output-file "$LR_DIR/results.json" \
        --no-unsloth \
        --batch-size 16

    echo "lr=$LR done at $(date)"
done

echo ""
echo "=== Results ==="
for LR in 2e-4 5e-4; do
    echo "--- lr=$LR ---"
    python3 -c "
import json
d = json.load(open('${BASE_OUT}/lr_${LR}/results.json'))
for t in ['functional_group','ring_counting','chain_length','canonicalization','fragment_assembly']:
    if t in d: print(f'  {t}: {d[t][\"accuracy\"]:.4f}')
" 2>/dev/null || echo "  not available"
done

echo "=== Complete at $(date) ==="
