#!/bin/bash
#SBATCH --job-name=cm_nci_domain
#SBATCH -A scavenger-h200
#SBATCH -p scavenger-h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=24:00:00
#SBATCH --output=output/logs/domain_shift/nci_domain_%j.log
#SBATCH --error=output/logs/domain_shift/nci_domain_%j.err
#SBATCH --requeue
#SBATCH --exclude=dcc-h200-gpu-05
#
# NCI DTP domain transfer: train on NCI drug compounds (263K),
# evaluate cross-domain on NCI and ZINC test sets.
#
# Prereq: data/sources/nci.smi must exist (run download_nci.py first)
#
# Submit: sbatch scripts/train_domain_nci.sh

set -euo pipefail
cd ~/storage/CLARIMOL
mkdir -p output/logs/domain_shift

export CONDARC=/work/gc237/.condarc
export HF_HOME=/work/gc237/.cache/huggingface
export HF_TOKEN=$(cat /work/gc237/.cache/huggingface/token 2>/dev/null || cat ~/.cache/huggingface/token)

source /opt/apps/rhel9/Anaconda3-2024.02/etc/profile.d/conda.sh
conda activate /work/gc237/conda_envs/clarimol

echo "=== NCI Domain Transfer ==="
nvidia-smi
echo "Start: $(date)"

MODEL_ID="meta-llama/Llama-3.1-8B-Instruct"
SMILES_FILE="data/sources/nci.smi"
TRAIN_DATA="data/nci_train"
TEST_DATA="data/nci_test"
OUT_DIR="output/domain_transfer/nci"
ZINC_MODEL="output/multi_seed/llama-8b/seed_42/final"

if [ ! -f "$SMILES_FILE" ]; then
    echo "ERROR: $SMILES_FILE not found. Run download_nci.py first."
    exit 1
fi

echo "NCI SMILES count: $(wc -l < $SMILES_FILE)"

# Step 1: Prepare NCI training data
if [ ! -f "$TRAIN_DATA/functional_group.json" ]; then
    echo "=== Preparing NCI training data ==="
    python -m clarimol prepare \
        --source file \
        --smiles-file "$SMILES_FILE" \
        --output-dir "$TRAIN_DATA" \
        --max-molecules 60000 \
        --keep-n 50000 \
        --subsample random \
        --no-curriculum \
        --seed 42
else
    echo "SKIP: NCI training data exists"
fi

# Step 2: Prepare NCI test data (non-overlapping)
if [ ! -f "$TEST_DATA/functional_group.json" ]; then
    echo "=== Preparing NCI test data ==="
    python -m clarimol prepare \
        --source file \
        --smiles-file "$SMILES_FILE" \
        --output-dir "$TEST_DATA" \
        --max-molecules 15000 \
        --keep-n 10000 \
        --subsample random \
        --no-curriculum \
        --seed 99
else
    echo "SKIP: NCI test data exists"
fi

mkdir -p "$OUT_DIR"

# Step 3: Train on NCI
if [ ! -d "$OUT_DIR/final" ]; then
    echo "=== Training on NCI ==="
    RESUME_FLAG=""
    if ls "$OUT_DIR"/checkpoint-* 1>/dev/null 2>&1; then
        RESUME_FLAG="--resume"
    fi
    python -m clarimol train \
        --model "$MODEL_ID" \
        --data-dir "$TRAIN_DATA" \
        --output-dir "$OUT_DIR" \
        --no-unsloth \
        --max-length 512 \
        --batch-size 16 \
        --grad-accum 1 \
        --lr 1e-4 \
        --epochs 1 \
        --lora-r 64 \
        --lora-alpha 16 \
        --bf16 \
        --no-4bit \
        --no-wandb \
        --save-steps 500 \
        --seed 42 \
        $RESUME_FLAG
else
    echo "SKIP: NCI model already trained"
fi

# Step 4: Evaluate NCI-trained on NCI test
if [ -d "$OUT_DIR/final" ] && [ ! -f "$OUT_DIR/nci_on_nci.json" ]; then
    echo "=== Evaluating NCI-trained on NCI test ==="
    python -m clarimol evaluate \
        --model-path "$OUT_DIR/final" \
        --data-dir "$TEST_DATA" \
        --output-file "$OUT_DIR/nci_on_nci.json" \
        --no-unsloth \
        --batch-size 16
fi

# Step 5: Evaluate NCI-trained on ZINC test
if [ -d "$OUT_DIR/final" ] && [ ! -f "$OUT_DIR/nci_on_zinc.json" ]; then
    echo "=== Evaluating NCI-trained on ZINC test ==="
    python -m clarimol evaluate \
        --model-path "$OUT_DIR/final" \
        --data-dir data/test \
        --output-file "$OUT_DIR/nci_on_zinc.json" \
        --no-unsloth \
        --batch-size 16
fi

# Step 6: Evaluate ZINC-trained on NCI test
if [ -d "$ZINC_MODEL" ] && [ ! -f "$OUT_DIR/zinc_on_nci.json" ]; then
    echo "=== Evaluating ZINC-trained on NCI test ==="
    python -m clarimol evaluate \
        --model-path "$ZINC_MODEL" \
        --data-dir "$TEST_DATA" \
        --output-file "$OUT_DIR/zinc_on_nci.json" \
        --no-unsloth \
        --batch-size 16
fi

# Print results
echo ""
echo "=== NCI Domain Transfer Results ==="
for f in "$OUT_DIR"/*.json; do
    echo "--- $(basename $f) ---"
    python3 -c "
import json
d = json.load(open('$f'))
accs = {k: round(v['accuracy'], 4) for k, v in d.items() if 'accuracy' in v}
print(accs)
print(f'mean={sum(accs.values())/len(accs):.4f}')
"
done

echo ""
echo "=== Complete at $(date) ==="
