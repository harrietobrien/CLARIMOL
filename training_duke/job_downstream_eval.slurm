#!/bin/bash
#SBATCH --job-name=clarimol_dseval
#SBATCH --partition=scavenger-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:5000_ada:1
#SBATCH --time=04:00:00
#SBATCH --output=output/duke_dseval_%j.log
#SBATCH --error=output/duke_dseval_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

export CONDARC=/work/gc237/.condarc
export HF_HOME=/work/gc237/.cache/huggingface

module load Anaconda3 2>/dev/null || module load conda 2>/dev/null || true
eval "$(conda shell.bash hook)"
conda activate /work/gc237/conda_envs/clarimol

echo "CLARIMOL Downstream Evaluation:"
nvidia-smi
echo "Start: $(date)"

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
