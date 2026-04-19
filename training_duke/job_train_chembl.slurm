#!/bin/bash
#SBATCH --job-name=clarimol_chembl
#SBATCH --partition=scavenger-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:6000_ada:1
#SBATCH --time=1-00:00:00
#SBATCH --output=output/duke_chembl_%j.log
#SBATCH --error=output/duke_chembl_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

export CONDARC=/work/gc237/.condarc
export HF_HOME=/work/gc237/.cache/huggingface

module load Anaconda3 2>/dev/null || module load conda 2>/dev/null || true
eval "$(conda shell.bash hook)"
conda activate /work/gc237/conda_envs/clarimol

mkdir -p output data/chembl

echo "CLARIMOL: ChEMBL pre-training pipeline"
nvidia-smi
echo "Start: $(date)"

# Step 1: Download ChEMBL SMILES and prepare parsing dataset
echo "=== Preparing ChEMBL dataset ==="
python -c "
from datasets import load_dataset
import json

ds = load_dataset('antoinebcx/smiles-molecules-chembl', split='train')
# Extract SMILES column
smiles = [row['smiles'] for row in ds if row.get('smiles')]
print(f'Loaded {len(smiles)} SMILES from ChEMBL')

# Save raw SMILES for clarimol prepare
with open('data/chembl/smiles.json', 'w') as f:
    json.dump(smiles[:50000], f)  # 50K for manageable size
print('Saved 50K SMILES to data/chembl/smiles.json')
"

# Step 2: Generate parsing task data from ChEMBL molecules
echo "=== Generating parsing tasks from ChEMBL ==="
python -m clarimol prepare \
    --source file \
    --smiles-file data/chembl/smiles.json \
    --output-dir data/chembl_tasks \
    --max-molecules 50000

# Step 3: Train on ChEMBL parsing tasks
echo "=== Training on ChEMBL parsing tasks ==="
python -m clarimol train \
    --model Qwen/Qwen2.5-7B-Instruct \
    --data-dir data/chembl_tasks \
    --output-dir output/parsing_pretrain_chembl \
    --no-unsloth \
    --max-length 512 \
    --batch-size 8 \
    --grad-accum 2 \
    --lr 5e-4 \
    --epochs 1 \
    --lora-r 64 \
    --lora-alpha 16 \
    --select-sample 10000 \
    --no-wandb \
    --no-fp16 \
    --seed 42

# Step 4: Evaluate
echo "=== Evaluating on ZINC test set ==="
python -m clarimol evaluate \
    --model-path output/parsing_pretrain_chembl/final \
    --data-dir data/test \
    --output-file output/results_chembl_trained.json \
    --max-samples 2000 \
    --no-unsloth

echo "ChEMBL pipeline complete at $(date) :)"
