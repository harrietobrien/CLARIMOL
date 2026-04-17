#!/bin/bash
#SBATCH --job-name=clarimol_ada
#SBATCH --partition=gpu-common
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:5000_ada:1
#SBATCH --time=04:00:00
#SBATCH --output=output/duke_train_%j.log
#SBATCH --error=output/duke_train_%j.err
#
#
# Submit: sbatch job_ada_single.slurm

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module load Anaconda3 2>/dev/null || module load conda 2>/dev/null || true
export CONDARC=/work/gc237/.condarc
conda activate /work/gc237/conda_envs/clarimol

mkdir -p output

echo "CLARIMOL on Duke DCC ada:"
nvidia-smi
echo "Start: $(date)"

# H200 config: bf16, large batch, full dataset, no quantization
# With 141GB VRAM we can skip 4-bit quant and use full bf16 model
python -m clarimol train \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --data-dir /hpc/home/gc237/storage/CLARIMOL/data/clarimol \
    --output-dir /hpc/home/gc237/storage/CLARIMOL/output/parsing_pretrain \
    --no-unsloth \
    --packing \
    --max-length 512 \
    --batch-size 32 \
    --grad-accum 1 \
    --lr 5e-4 \
    --epochs 1 \
    --lora-r 64 \
    --lora-alpha 16 \
    --no-fp16 \
    --no-wandb \
    --seed 42

echo "Training complete at $(date) :)"

# Evaluate
bash scripts/evaluate_all.sh /hpc/home/gc237/storage/CLARIMOL/output/parsing_pretrain/final
