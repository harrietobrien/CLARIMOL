#!/bin/bash
#SBATCH --job-name=clarimol_dsmulti
#SBATCH --partition=scavenger-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:a5000:4
#SBATCH --time=1-00:00:00
#SBATCH --output=output/duke_dsmulti_%j.log
#SBATCH --error=output/duke_dsmulti_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

export CONDARC=/work/gc237/.condarc
export HF_HOME=/work/gc237/.cache/huggingface

module load Anaconda3 2>/dev/null || module load conda 2>/dev/null || true
eval "$(conda shell.bash hook)"
conda activate /work/gc237/conda_envs/clarimol

mkdir -p output

echo "CLARIMOL Downstream Multi-GPU (4x A5000, base model, 3K):"
nvidia-smi
echo "Start: $(date)"

for TASK in retrosynthesis reagent_prediction forward_reaction_prediction; do
    echo "=== Training $TASK at $(date) ==="
    accelerate launch \
        --num_processes 4 \
        --multi_gpu \
        -m clarimol downstream-train \
        --model Qwen/Qwen2.5-7B-Instruct \
        --task "$TASK" \
        --data-dir data/mol_instructions \
        --output-dir "output/downstream_multi/$TASK" \
        --no-unsloth \
        --batch-size 4 \
        --grad-accum 1 \
        --lr 2e-5 \
        --epochs 1 \
        --lora-r 64 \
        --lora-alpha 16 \
        --max-samples 3000 \
        --no-wandb \
        --no-fp16 \
        --seed 42
    echo "=== $TASK training complete at $(date) ==="
done

echo "All training complete. Starting evaluation..."

for TASK in retrosynthesis reagent_prediction forward_reaction_prediction; do
    echo "=== Evaluating $TASK at $(date) ==="
    python -m clarimol downstream-eval \
        --model-path "output/downstream_multi/$TASK/final" \
        --data-dir data/mol_instructions \
        --tasks "$TASK" \
        --output-file "output/results_downstream_multi_$TASK.json" \
        --max-samples 500 \
        --no-unsloth
    echo "=== $TASK eval complete at $(date) ==="
done

echo "All downstream multi-GPU complete at $(date) :)"
