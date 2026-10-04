#!/bin/bash
#SBATCH --job-name=cm_chembench
#SBATCH -A scavenger-h200
#SBATCH -p scavenger-h200
#SBATCH --gres=gpu:h200:1
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=output/logs/benchmarks/chembench_%j.log
#SBATCH --error=output/logs/benchmarks/chembench_%j.err
#
# ChemBench evaluation (Mirza et al., Nature Chemistry 2025).
# 2,700+ QA pairs across chemistry topics.
# Tests whether SMILES parsing pre-training transfers to general
# chemistry question answering.
#
# GitHub: https://github.com/lamalab-org/chembench
# Paper: https://www.nature.com/articles/s41557-025-01815-x
#
# Submit: sbatch scripts/eval_benchmark_chembench.sh

set -euo pipefail
cd ~/storage/CLARIMOL
mkdir -p output/logs/benchmarks
mkdir -p output/benchmarks/chembench

export CONDARC=/work/gc237/.condarc
export HF_HOME=/work/gc237/.cache/huggingface
export HF_TOKEN=$(cat /work/gc237/.cache/huggingface/token 2>/dev/null || cat ~/.cache/huggingface/token 2>/dev/null || true)

source /opt/apps/rhel9/Anaconda3-2024.02/etc/profile.d/conda.sh
conda activate /work/gc237/conda_envs/clarimol

echo "ChemBench Benchmark Evaluation"
nvidia-smi
echo "Start: $(date)"

# Install chembench if not present
if ! python3 -c "import chembench" 2>/dev/null; then
    echo "Installing chembench..."
    pip install chembench
fi

# Skip if all results already exist
if [ -f "output/benchmarks/chembench/results.json" ]; then
    echo "SKIP: ChemBench combined results already exist"
    echo "Delete output/benchmarks/chembench/results.json to re-run"
    exit 0
fi

python3 scripts/eval_benchmark_chembench.py \
    --output-dir output/benchmarks/chembench

echo "ChemBench evaluation complete at $(date)"
