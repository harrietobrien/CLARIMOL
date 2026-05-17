#!/bin/bash
#SBATCH --job-name=cm_chembl_prep
#SBATCH --partition=scavenger
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=output/logs/domain_shift/chembl_prep_%j.log
#SBATCH --error=output/logs/domain_shift/chembl_prep_%j.err
#SBATCH --requeue
#
# Prepare ChEMBL 250K dataset (CPU only, no GPU needed)
#

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p output/logs/domain_shift

export CONDARC=/work/gc237/.condarc
export HF_HOME=/work/gc237/.cache/huggingface

module load Anaconda3 2>/dev/null || module load conda 2>/dev/null || true
eval "$(conda shell.bash hook)"
conda activate /work/gc237/conda_envs/clarimol

echo "=== ChEMBL 250K Data Preparation ==="
echo "Start: $(date)"

python -m clarimol prepare \
    --source chembl \
    --max-molecules 250000 \
    --output-dir data/chembl_250k \
    --seed 42

echo ""
echo "=== Dataset summary ==="
for f in data/chembl_250k/*.json; do
    task=$(basename "$f" .json)
    count=$(python -c "import json; print(len(json.load(open('$f'))))")
    echo "  $task: $count samples"
done

echo ""
echo "Done: $(date)"
