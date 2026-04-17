#!/bin/bash
#SBATCH --job-name=clarimol_3b
#SBATCH --partition=scavenger-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --gres=gpu:2080:1
#SBATCH --time=04:00:00
#SBATCH --output=output/duke_3b_%j.log
#SBATCH --error=output/duke_3b_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

export CONDARC=/work/gc237/.condarc
export HF_HOME=/work/gc237/.cache/huggingface

module load Anaconda3 2>/dev/null || module load conda 2>/dev/null || true
eval "$(conda shell.bash hook)"
conda activate /work/gc237/conda_envs/clarimol

mkdir -p output

echo "CLARIMOL 3B on scavenger-gpu (2080 Ti):"
nvidia-smi
echo "Start: $(date)"

python -m clarimol train \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --data-dir data/clarimol \
    --output-dir output/parsing_pretrain_3b \
    --max-length 512 \
    --batch-size 1 \
    --grad-accum 16 \
    --lr 5e-4 \
    --epochs 1 \
    --lora-r 16 \
    --lora-alpha 16 \
    --select-sample 10000 \
    --no-wandb \
    --no-grad-ckpt \
    --no-fp16 \
    --seed 42

echo "Training complete at $(date) :)"
bash scripts/evaluate_all.sh output/parsing_pretrain_3b/final
