#!/bin/bash
# Environment setup - DCC
set -euo pipefail

ENV_NAME="clarimol"

module load Anaconda3 2>/dev/null || module load conda 2>/dev/null || true

if ! conda env list | grep -q "$ENV_NAME"; then
    conda create -n "$ENV_NAME" python=3.11 -y
fi

conda activate "$ENV_NAME"

# PyTorch with CUDA 12.4 (H200 compatible)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

pip install -r "$(dirname "$0")/requirements.txt"
pip install -e "$(dirname "$0")/.."

echo "Environment '$ENV_NAME' ready."
