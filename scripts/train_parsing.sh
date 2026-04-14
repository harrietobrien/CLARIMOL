#!/usr/bin/env bash
: <<'COMMENT'
Stage 1: Pre-train on SMILES parsing tasks
    Configured for Tesla P100 (16GB VRAM):
        batch_size=8, grad_accum=2 → effective=16
    e.g., bash scripts/train_parsing.sh

With standard HF (no unsloth):
    e.g., bash scripts/train_parsing.sh --no-unsloth
COMMENT

set -euo pipefail
cd "$(dirname "$0")/.."

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export CUDA_VISIBLE_DEVICES

python -m clarimol train \
    --model "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit" \
    --data-dir data/clarimol \
    --output-dir output/parsing_pretrain \
    --batch-size 8 \
    --grad-accum 2 \
    --lr 5e-4 \
    --epochs 1 \
    --lora-r 64 \
    --lora-alpha 16 \
    --seed 42 \
    "$@"