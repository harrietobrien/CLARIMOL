#!/bin/bash
# NRP JupyterHub environment setup.
# The pytorch-notebook image already has PyTorch + CUDA.
# We only need to install the ML/chem dependencies.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Checking PyTorch..."
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.cuda.is_available()}')"

echo "Installing dependencies..."
pip install --user -r "$SCRIPT_DIR/requirements.txt"

# Install clarimol package
pip install --user -e "$SCRIPT_DIR/.."

echo "Verifying GPU..."
python -c "
import torch
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
"

echo "Setup complete :)"
