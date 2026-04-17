#!/bin/bash
# Optimized training for NRP JupyterHub RTX 2080 Ti (11GB VRAM)
#
# RTX 2080 Ti constraints:
#   - 11GB VRAM (vs P100 16GB) — tighter memory budget
#   - Turing arch: fp16 tensor cores (2x speedup over P100's Pascal)
#   - No bf16 support
#
# Strategy: 4-bit quant + LoRA + packing + short sequences
# batch=8 fits safely; batch=12 might work but risks OOM.

set -euo pipefail
cd "$(dirname "$0")/.."

OUTPUT_DIR="output/parsing_pretrain"
LOG_FILE="output/train_nrp.log"
mkdir -p output

echo "Starting NRP training at $(date)"
nvidia-smi

python -m clarimol train \
    --model unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit \
    --data-dir data/clarimol \
    --output-dir "$OUTPUT_DIR" \
    --no-unsloth \
    --packing \
    --max-length 512 \
    --select-sample 10000 \
    --batch-size 8 \
    --grad-accum 2 \
    --lr 5e-4 \
    --epochs 1 \
    --lora-r 64 \
    --lora-alpha 16 \
    --no-wandb \
    --seed 42 \
    2>&1 | tee "$LOG_FILE"

echo "Training complete at $(date) :)"
echo "Model: $OUTPUT_DIR/final"
