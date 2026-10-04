#!/bin/bash
#SBATCH --job-name=cm_fgbench
#SBATCH -A scavenger-h200
#SBATCH -p scavenger-h200
#SBATCH --gres=gpu:h200:1
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=output/logs/benchmarks/fgbench_%j.log
#SBATCH --error=output/logs/benchmarks/fgbench_%j.err
#
# FGBench evaluation (Liu et al., NeurIPS 2025 D&B).
# 7K curated molecular property reasoning questions at functional-group level.
# Tests whether SMILES parsing pre-training transfers to FG-level reasoning.
#
# Dataset: https://huggingface.co/datasets/xuan-liu/FGBench
# Paper: arXiv:2508.01055
#
# Submit: sbatch scripts/eval_benchmark_fgbench.sh

set -euo pipefail
cd ~/storage/CLARIMOL
mkdir -p output/logs/benchmarks
mkdir -p output/benchmarks/fgbench

export CONDARC=/work/gc237/.condarc
export HF_HOME=/work/gc237/.cache/huggingface
export HF_TOKEN=$(cat /work/gc237/.cache/huggingface/token 2>/dev/null || cat ~/.cache/huggingface/token 2>/dev/null || true)

source /opt/apps/rhel9/Anaconda3-2024.02/etc/profile.d/conda.sh
conda activate /work/gc237/conda_envs/clarimol

echo "FGBench Benchmark Evaluation"
nvidia-smi
echo "Start: $(date)"

# Skip if all results already exist
if [ -f "output/benchmarks/fgbench/results.json" ]; then
    echo "SKIP: FGBench combined results already exist"
    echo "Delete output/benchmarks/fgbench/results.json to re-run"
    exit 0
fi

python3 scripts/eval_benchmark_fgbench.py \
    --output-dir output/benchmarks/fgbench

echo "FGBench evaluation complete at $(date)"
