#!/bin/bash
#SBATCH --job-name=clarimol_h200_qwen3b
#SBATCH -A scavenger-h200
#SBATCH -p scavenger-h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=24:00:00
#SBATCH --output=output/h200_qwen3b_%j.log
#SBATCH --error=output/h200_qwen3b_%j.err
#SBATCH --requeue
#SBATCH --exclude=dcc-h200-gpu-05
#
# Single H200: Qwen 2.5-3B on full 250K ZINC dataset
# Size ablation: does parsing pre-training benefit smaller models more or less?
#
# Submit: sbatch training_duke/job_h200_250k_qwen3b.slurm

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

export CONDARC=/work/gc237/.condarc
export HF_HOME=/work/gc237/.cache/huggingface
export HF_TOKEN=$(cat /work/gc237/.cache/huggingface/token)

module load Anaconda3 2>/dev/null || module load conda 2>/dev/null || true
eval "$(conda shell.bash hook)"
conda activate /work/gc237/conda_envs/clarimol

OUTPUT_DIR=output/parsing_pretrain_h200_qwen3b
mkdir -p "$OUTPUT_DIR"

echo "=== CLARIMOL H200 Qwen 2.5-3B (250K) ==="
nvidia-smi
echo "Start: $(date)"

RESUME_FLAG=""
if ls "$OUTPUT_DIR"/checkpoint-* 1>/dev/null 2>&1; then
    echo "Checkpoint(s) detected — resuming training"
    RESUME_FLAG="--resume"
fi

python -m clarimol train \
    --model Qwen/Qwen2.5-3B-Instruct \
    --data-dir data/clarimol \
    --output-dir "$OUTPUT_DIR" \
    --no-unsloth \
    --max-length 512 \
    --batch-size 32 \
    --grad-accum 1 \
    --lr 1e-4 \
    --epochs 1 \
    --lora-r 64 \
    --lora-alpha 16 \
    --bf16 \
    --no-4bit \
    --no-wandb \
    --save-steps 500 \
    --seed 42 \
    $RESUME_FLAG

echo "Training complete at $(date)"

bash scripts/evaluate_all.sh "$OUTPUT_DIR/final" --max-samples 2000
