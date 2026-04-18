#!/bin/bash
# Training for NRP JupyterHub NVIDIA Titan Xp (12GB VRAM)
#
# Titan Xp constraints:
#   - 12GB VRAM — too small for 8B models even at 4-bit
#   - Pascal arch (CC 6.1): no bf16, no fp16 tensor cores
#   - Requires PyTorch 2.2.2 (newer versions dropped CC 6.1)
#
# Strategy: Qwen 3B + 4-bit quant + LoRA-16 + batch=1 + grad_accum=16
# 50K samples (10K/task), fp32 compute, no packing.

set -euo pipefail
cd "$(dirname "$0")/.."

OUTPUT_DIR="output/parsing_pretrain"
LOG_FILE="output/train_nrp.log"
mkdir -p output

echo "Starting NRP training at $(date)"
nvidia-smi

python -m clarimol train \
    --model Qwen/Qwen2.5-3B-Instruct \
    --data-dir data/clarimol \
    --output-dir "$OUTPUT_DIR" \
    --no-unsloth \
    --max-length 512 \
    --select-sample 10000 \
    --batch-size 1 \
    --grad-accum 16 \
    --lr 5e-4 \
    --epochs 1 \
    --lora-r 16 \
    --lora-alpha 16 \
    --no-fp16 \
    --no-wandb \
    --seed 42 \
    2>&1 | tee "$LOG_FILE"

echo "Training complete at $(date) :)"
echo "Model: $OUTPUT_DIR/final"
