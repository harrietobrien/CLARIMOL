#!/bin/bash
#SBATCH --job-name=clarimol_ds
#SBATCH --partition=scavenger-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:5000_ada:1
#SBATCH --time=14:00:00
#SBATCH --output=output/duke_downstream_%j.log
#SBATCH --error=output/duke_downstream_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

export CONDARC=/work/gc237/.condarc
export HF_HOME=/work/gc237/.cache/huggingface

module load Anaconda3 2>/dev/null || module load conda 2>/dev/null || true
eval "$(conda shell.bash hook)"
conda activate /work/gc237/conda_envs/clarimol

mkdir -p output

echo "CLARIMOL Downstream Training on Ada:"
nvidia-smi
echo "Start: $(date)"

# Stage 2: Fine-tune on downstream tasks using pre-trained model
PRETRAINED=output/parsing_pretrain_ada_2/final

for TASK in retrosynthesis reagent_prediction forward_reaction_prediction; do
    echo "=== Training on $TASK at $(date) ==="
    python -m clarimol downstream-train \
        --model "$PRETRAINED" \
        --task "$TASK" \
        --data-dir data/mol_instructions \
        --output-dir "output/downstream/$TASK" \
        --no-unsloth \
        --batch-size 8 \
        --grad-accum 2 \
        --lr 5e-4 \
        --epochs 1 \
        --lora-r 64 \
        --lora-alpha 16 \
        --max-samples 5000 \
        --no-wandb \
        --no-fp16 \
        --seed 42
    echo "=== $TASK complete at $(date) ==="
done

echo "All downstream training complete at $(date) :)"

# Evaluate all downstream tasks
for TASK in retrosynthesis reagent_prediction forward_reaction_prediction; do
    echo "=== Evaluating $TASK at $(date) ==="
    python -m clarimol downstream-eval \
        --model-path "output/downstream/$TASK/final" \
        --data-dir data/mol_instructions \
        --tasks "$TASK" \
        --output-file "output/results_downstream_$TASK.json" \
        --max-samples 500 \
        --no-unsloth
    echo "=== $TASK eval complete at $(date) ==="
done

echo "All downstream evaluation complete at $(date) :)"
