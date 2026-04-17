#!/usr/bin/env bash
: <<'COMMENT'
Stage 1: Pre-train on SMILES parsing tasks
    Auto-detects GPU VRAM and configures accordingly:
        ≤6GB  → batch=2, grad_accum=8, max_length=512
        >6GB  → batch=8, grad_accum=2, max_length=2048
    Both yield effective batch size 16.
    Uses Qwen2.5-1.5B-Instruct at 4-bit quantization.

    e.g., bash scripts/train_4gb.sh
          CUDA_VISIBLE_DEVICES=0 bash scripts/train_4gb.sh
COMMENT

set -euo pipefail
cd "$(dirname "$0")/.."

# Auto-select GPU with most VRAM if no override
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    BEST_GPU=$(python -c "
import torch
best_idx, best_mem = 0, 0
for i in range(torch.cuda.device_count()):
    mem = torch.cuda.get_device_properties(i).total_memory
    if mem > best_mem:
        best_idx, best_mem = i, mem
print(best_idx)
" 2>/dev/null)
    export CUDA_VISIBLE_DEVICES="${BEST_GPU:-0}"
fi

# Detect VRAM via PyTorch (accounts for CUDA_VISIBLE_DEVICES remapping)
VRAM_MB=$(CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" python -c "
import torch
print(int(torch.cuda.get_device_properties(0).total_memory / 1024**2))
" 2>/dev/null)

GPU_NAME=$(CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" python -c "
import torch; print(torch.cuda.get_device_name(0))
" 2>/dev/null)

echo "Using GPU: $GPU_NAME (${VRAM_MB}MB VRAM, CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"

if [ "$VRAM_MB" -le 6144 ]; then
    echo "Low-VRAM mode: batch=2, grad_accum=8, max_length=512"
    BATCH=2
    GRAD_ACCUM=8
    MAX_LENGTH=512
else
    echo "Standard mode: batch=8, grad_accum=2, max_length=2048"
    BATCH=8
    GRAD_ACCUM=2
    MAX_LENGTH=2048
fi

python -m clarimol train \
    --model "Qwen/Qwen2.5-1.5B-Instruct" \
    --data-dir data/clarimol \
    --output-dir output/parsing_pretrain \
    --batch-size "$BATCH" \
    --grad-accum "$GRAD_ACCUM" \
    --max-length "$MAX_LENGTH" \
    --lr 5e-4 \
    --epochs 1 \
    --lora-r 32 \
    --lora-alpha 16 \
    \
    --no-unsloth \
    --no-wandb \
    --seed 42 \
    "$@"
