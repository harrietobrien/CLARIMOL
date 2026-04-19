#!/bin/bash
#SBATCH --job-name=clarimol_dsbase
#SBATCH --partition=scavenger-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --output=output/duke_dsbase_%j.log
#SBATCH --error=output/duke_dsbase_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

export CONDARC=/work/gc237/.condarc
export HF_HOME=/work/gc237/.cache/huggingface

module load Anaconda3 2>/dev/null || module load conda 2>/dev/null || true
eval "$(conda shell.bash hook)"
conda activate /work/gc237/conda_envs/clarimol

mkdir -p output

echo "CLARIMOL Downstream (BASE model, no pre-training, 10K):"
nvidia-smi
echo "Start: $(date)"

for TASK in retrosynthesis reagent_prediction forward_reaction_prediction; do
    echo "=== Training $TASK at $(date) ==="
    python -m clarimol downstream-train \
        --model Qwen/Qwen2.5-7B-Instruct \
        --task "$TASK" \
        --data-dir data/mol_instructions \
        --output-dir "output/downstream_base/$TASK" \
        --no-unsloth \
        --batch-size 8 \
        --grad-accum 2 \
        --lr 5e-4 \
        --epochs 1 \
        --lora-r 64 \
        --lora-alpha 16 \
        --max-samples 10000 \
        --no-wandb \
        --no-fp16 \
        --seed 42
    echo "=== $TASK training complete at $(date) ==="
done

echo "All training complete. Starting evaluation..."

for TASK in retrosynthesis reagent_prediction forward_reaction_prediction; do
    echo "=== Evaluating $TASK at $(date) ==="
    python -m clarimol downstream-eval \
        --model-path "output/downstream_base/$TASK/final" \
        --data-dir data/mol_instructions \
        --tasks "$TASK" \
        --output-file "output/results_downstream_base_$TASK.json" \
        --max-samples 500 \
        --no-unsloth
    echo "=== $TASK eval complete at $(date) ==="
done

echo "All downstream base 10K complete at $(date) :)"
