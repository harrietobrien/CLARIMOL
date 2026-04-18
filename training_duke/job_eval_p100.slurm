#!/bin/bash
#SBATCH --job-name=clarimol_p100
#SBATCH --partition=courses-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:p100:1
#SBATCH --time=08:00:00
#SBATCH --output=output/duke_p100eval_%j.log
#SBATCH --error=output/duke_p100eval_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

export CONDARC=/work/gc237/.condarc
export HF_HOME=/work/gc237/.cache/huggingface

module load Anaconda3 2>/dev/null || module load conda 2>/dev/null || true
eval "$(conda shell.bash hook)"
conda activate /work/gc237/conda_envs/clarimol

echo "CLARIMOL P100 Evaluation:"
nvidia-smi
echo "Start: $(date)"

# Evaluate P100-trained LLaMA 8B model
python -m clarimol evaluate \
    --model-path output/parsing_pretrain_p100/final \
    --data-dir data/test \
    --output-file output/results_p100_trained.json \
    --max-samples 2000 \
    --no-unsloth

echo "P100 evaluation complete at $(date) :)"
