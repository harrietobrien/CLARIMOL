#!/bin/bash
#SBATCH --job-name=clarimol_multi
#SBATCH --partition=gpu-common
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:h200:4
#SBATCH --time=02:00:00
#SBATCH --output=output/duke_multi_%j.log
#SBATCH --error=output/duke_multi_%j.err
#
# Multi-GPU H200 training w/ DeepSpeed ZeRO-2
# 4x H200 w/ NVLink — can process full 250K dataset quickly
#
# Submit: sbatch job_h200_multi.slurm

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module load Anaconda3 2>/dev/null || module load conda 2>/dev/null || true
conda activate clarimol

mkdir -p output

echo "CLARIMOL Multi-GPU on Duke DCC"
nvidia-smi
echo "Start: $(date)"

# DeepSpeed ZeRO-2 across 4 H200s
accelerate launch \
    --num_processes 4 \
    --use_deepspeed \
    --deepspeed_config_file training_duke/deepspeed_config.json \
    -m clarimol train \
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

echo "Multi-GPU training complete at $(date) :)"
bash scripts/evaluate_all.sh output/parsing_pretrain/final
