#!/bin/bash
#SBATCH --job-name=cm_cod250k_prep
#SBATCH --partition=scavenger
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=output/logs/cod_shift/cod_250k_prep_%j.log
#SBATCH --error=output/logs/cod_shift/cod_250k_prep_%j.err
#SBATCH --requeue
#
# Step 1: Fetch and prepare 250K COD crystal SMILES (CPU only)
# Run this BEFORE job_cod_250k_shift.slurm
#

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p output/logs/cod_shift

export CONDARC=/work/gc237/.condarc
export HF_HOME=/work/gc237/.cache/huggingface

module load Anaconda3 2>/dev/null || module load conda 2>/dev/null || true
eval "$(conda shell.bash hook)"
conda activate /work/gc237/conda_envs/clarimol

COD_DATA="data/cod_250k"
SEED=42

echo "=== COD 250K Data Preparation ==="
echo "Start: $(date)"

python -m clarimol prepare \
    --source cod-bulk \
    --max-molecules 250000 \
    --output-dir "$COD_DATA" \
    --cod-cache-dir data/cod_cache \
    --seed "$SEED"

echo ""
echo "=== Dataset summary ==="
for f in "$COD_DATA"/*.json; do
    task=$(basename "$f" .json)
    count=$(python -c "import json; print(len(json.load(open('$f'))))")
    echo "  $task: $count samples"
done

echo ""
echo "Done: $(date)"
