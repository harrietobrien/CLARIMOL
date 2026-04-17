#!/bin/bash
#SBATCH --job-name=clarimol_h200
#SBATCH --partition=gpu-common
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=04:00:00
#SBATCH --output=output/duke_train_%j.log
#SBATCH --error=output/duke_train_%j.err
#
# Single H200 training: full 250K dataset, bf16, no quantization needed
# H200 has 141GB HBM3e — 8B model fits entirely in bf16 (~16GB)
#
# Submit: sbatch job_h200_single.slurm

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module load Anaconda3 2>/dev/null || module load conda 2>/dev/null || true
conda activate clarimol

mkdir -p output

echo "CLARIMOL on Duke DCC H200:"
nvidia-smi
echo "Start: $(date)"

# H200 config: bf16, large batch, full dataset, no quantization
# With 141GB VRAM we can skip 4-bit quant and use full bf16 model
python -m clarimol train \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --data-dir data/clarimol \
    --output-dir output/parsing_pretrain \
    --no-unsloth \
    --packing \
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

echo "Training complete at $(date) :)"

# Evaluate
bash scripts/evaluate_all.sh output/parsing_pretrain/final
