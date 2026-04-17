#!/bin/bash
#SBATCH --job-name=clarimol_fullprec
#SBATCH --partition=gpu-common
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=06:00:00
#SBATCH --output=output/duke_fullprec_%j.log
#SBATCH --error=output/duke_fullprec_%j.err
#
# Full-precision bf16 training w/o LoRA on a single H200
# H200 has enough VRAM to fine-tune all 8B params in bf16
# Produces a stronger model than LoRA (but takes longer)
#
# Submit: sbatch job_h200_full_precision.slurm

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module load Anaconda3 2>/dev/null || module load conda 2>/dev/null || true
conda activate clarimol

mkdir -p output

echo "Full-precision bf16 training on H200"
nvidia-smi
echo "Start: $(date)"

# Full fine-tuning: no LoRA, bf16, full dataset
# 8B model in bf16 = ~16GB w/ batch=32, activations ~40GB, total ~56GB

python -m clarimol train \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --data-dir data/clarimol \
    --output-dir output/parsing_pretrain_fullprec \
    --no-unsloth \
    --packing \
    --max-length 512 \
    --batch-size 32 \
    --grad-accum 2 \
    --lr 2e-5 \
    --epochs 1 \
    --lora-r 64 \
    --lora-alpha 16 \
    --no-fp16 \
    --no-wandb \
    --seed 42

echo "Done at $(date) :)"
