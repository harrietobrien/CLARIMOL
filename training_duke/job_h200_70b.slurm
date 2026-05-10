#!/bin/bash
#SBATCH --job-name=clarimol_70b
#SBATCH -A scavenger-h200
#SBATCH -p scavenger-h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=24:00:00
#SBATCH --output=output/h200_70b_%j.log
#SBATCH --error=output/h200_70b_%j.err
#SBATCH --requeue
#SBATCH --exclude=dcc-h200-gpu-05
#
# P2-14: 70B model experiment
# LLaMA 3.1-70B-Instruct in bf16 on single H200 (141GB)
# 70B model ≈ 140GB in bf16 — tight fit, use gradient checkpointing
#
# Submit: sbatch training_duke/job_h200_70b.slurm

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

export CONDARC=/work/gc237/.condarc
export HF_HOME=/work/gc237/.cache/huggingface
export HF_TOKEN=$(cat /work/gc237/.cache/huggingface/token)

module load Anaconda3 2>/dev/null || module load conda 2>/dev/null || true
eval "$(conda shell.bash hook)"
conda activate /work/gc237/conda_envs/clarimol

OUTPUT_DIR=output/parsing_pretrain_h200_70b
mkdir -p "$OUTPUT_DIR"

echo "=== CLARIMOL H200 LLaMA 3.1-70B ==="
nvidia-smi
echo "Start: $(date)"

RESUME_FLAG=""
if ls "$OUTPUT_DIR"/checkpoint-* 1>/dev/null 2>&1; then
    echo "Checkpoint(s) detected — resuming training"
    RESUME_FLAG="--resume"
fi

# 70B in bf16: ~140GB, fits on 1 H200 (141GB)
# Small batch + grad checkpointing to fit in remaining memory
# Using select-sample to limit data for feasibility
python -m clarimol train \
    --model meta-llama/Llama-3.1-70B-Instruct \
    --data-dir data/clarimol \
    --output-dir "$OUTPUT_DIR" \
    --no-unsloth \
    --max-length 512 \
    --batch-size 2 \
    --grad-accum 8 \
    --lr 1e-4 \
    --epochs 1 \
    --lora-r 32 \
    --lora-alpha 64 \
    --bf16 \
    --no-4bit \
    --no-wandb \
    --save-steps 250 \
    --select-sample 10000 \
    --seed 42 \
    $RESUME_FLAG

echo "Training complete at $(date)"

bash scripts/evaluate_all.sh "$OUTPUT_DIR/final" --max-samples 2000
