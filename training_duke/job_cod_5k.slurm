#!/bin/bash
#SBATCH --job-name=clarimol_cod5k
#SBATCH --partition=scavenger
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=output/duke_cod5k_%j.log
#SBATCH --error=output/duke_cod5k_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

export CONDARC=/work/gc237/.condarc
export HF_HOME=/work/gc237/.cache/huggingface

module load Anaconda3 2>/dev/null || module load conda 2>/dev/null || true
eval "$(conda shell.bash hook)"
conda activate /work/gc237/conda_envs/clarimol

echo "CLARIMOL COD 5K fetch + prepare:"
echo "Start: $(date)"

python -m clarimol prepare \
    --source cod \
    --max-molecules 5000 \
    --output-dir data/cod_5k

echo "COD 5K preparation complete at $(date) :)"
