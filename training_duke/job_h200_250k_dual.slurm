#!/bin/bash
#SBATCH --job-name=clarimol_h200_2gpu
#SBATCH --partition=scavenger-h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:h200:2
#SBATCH --time=04:00:00
#SBATCH --output=output/h200_2gpu_%j.log
#SBATCH --error=output/h200_2gpu_%j.err
#
# Dual H200: Qwen 2.5-7B on full 250K ZINC dataset across 2 GPUs
# accelerate handles data-parallel sharding via DDP
# ~2x throughput vs single GPU → halved wall time
#
# Submit: sbatch training_duke/job_h200_250k_dual.slurm

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

export CONDARC=/work/gc237/.condarc
export HF_HOME=/work/gc237/.cache/huggingface

module load Anaconda3 2>/dev/null || module load conda 2>/dev/null || true
eval "$(conda shell.bash hook)"
conda activate /work/gc237/conda_envs/clarimol

mkdir -p output

echo "=== CLARIMOL H200 dual-GPU (250K) ==="
nvidia-smi
echo "Start: $(date)"

accelerate launch \
    --num_processes 2 \
    --multi_gpu \
    -m clarimol train \
    --model Qwen/Qwen2.5-7B-Instruct \
    --data-dir data/clarimol \
    --output-dir output/parsing_pretrain_h200 \
    --no-unsloth \
    --max-length 512 \
    --batch-size 64 \
    --grad-accum 1 \
    --lr 5e-4 \
    --epochs 1 \
    --lora-r 64 \
    --lora-alpha 16 \
    --no-fp16 \
    --no-wandb \
    --seed 42

echo "Training complete at $(date)"

# Evaluate on test set
bash scripts/evaluate_all.sh output/parsing_pretrain_h200/final --max-samples 2000
