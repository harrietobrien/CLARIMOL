#!/bin/bash
#SBATCH --job-name=clarimol_cod
#SBATCH --partition=scavenger-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:5000_ada:1
#SBATCH --time=04:00:00
#SBATCH --output=output/duke_cod_%j.log
#SBATCH --error=output/duke_cod_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

export CONDARC=/work/gc237/.condarc
export HF_HOME=/work/gc237/.cache/huggingface

module load Anaconda3 2>/dev/null || module load conda 2>/dev/null || true
eval "$(conda shell.bash hook)"
conda activate /work/gc237/conda_envs/clarimol

mkdir -p output

echo "CLARIMOL COD Extension Evaluation on Ada:"
nvidia-smi
echo "Start: $(date)"

# Step 1: Prepare COD dataset (fetches from COD API + PubChem)
echo "=== Preparing COD dataset ==="
python -m clarimol prepare \
    --source cod \
    --max-molecules 1000 \
    --output-dir data/cod

# Step 2: Evaluate pre-trained model on COD molecules
echo "=== Evaluating pre-trained model on COD ==="
python -m clarimol evaluate \
    --model-path output/parsing_pretrain_ada_2/final \
    --data-dir data/cod \
    --output-file output/results_cod_trained.json \
    --max-samples 500 \
    --no-unsloth

# Step 3: Evaluate baseline on COD for comparison
echo "=== Evaluating baseline on COD ==="
python -m clarimol evaluate \
    --model-path Qwen/Qwen2.5-7B-Instruct \
    --data-dir data/cod \
    --output-file output/results_cod_baseline.json \
    --max-samples 500 \
    --no-unsloth

echo "COD evaluation complete at $(date) :)"
