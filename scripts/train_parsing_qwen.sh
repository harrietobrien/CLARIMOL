#!/usr/bin/env bash
: <<'COMMENT'
Stage 1: Pre-train Qwen2.5-7B-Instruct on SMILES parsing tasks
    Configured for Tesla P100 (16GB VRAM):
        batch_size=8, grad_accum=2 -> effective=16
    e.g., bash scripts/train_parsing_qwen.sh

With standard HF (no unsloth):
    e.g., bash scripts/train_parsing_qwen.sh --no-unsloth
COMMENT

set -euo pipefail
cd "$(dirname "$0")/.."

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export CUDA_VISIBLE_DEVICES

python -m clarimol train \
    --model "unsloth/Qwen2.5-7B-Instruct-bnb-4bit" \
    --data-dir data/clarimol \
    --output-dir output/parsing_pretrain_qwen \
    --batch-size 8 \
    --grad-accum 2 \
    --lr 5e-4 \
    --epochs 1 \
    --lora-r 64 \
    --lora-alpha 16 \
    --seed 42 \
    "$@"
