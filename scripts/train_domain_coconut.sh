#!/bin/bash
#SBATCH --job-name=cm_dom_coconut
#SBATCH -A scavenger-h200
#SBATCH -p scavenger-h200
#SBATCH --gres=gpu:h200:1
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=output/logs/domain_transfer/coconut_%j.log
#SBATCH --error=output/logs/domain_transfer/coconut_%j.err
#
# Domain transfer: COCONUT natural products
# 1. Prepare COCONUT SMILES into 5-task parsing format
# 2. Train LLaMA on COCONUT data
# 3. Eval COCONUT-trained model on COCONUT test + ZINC test (bidirectional)
# 4. Eval ZINC-trained model on COCONUT test (zero-shot transfer)
#
# Prereq: data/sources/coconut.smi must exist (one SMILES per line)
# Submit: sbatch scripts/train_domain_coconut.sh

set -euo pipefail
cd ~/storage/CLARIMOL
mkdir -p output/logs/domain_transfer

export CONDARC=/work/gc237/.condarc
export HF_HOME=/work/gc237/.cache/huggingface
export HF_TOKEN=$(cat /work/gc237/.cache/huggingface/token 2>/dev/null || cat ~/.cache/huggingface/token)

source /opt/apps/rhel9/Anaconda3-2024.02/etc/profile.d/conda.sh
conda activate /work/gc237/conda_envs/clarimol

echo "=== Domain Transfer: COCONUT Natural Products ==="
nvidia-smi
echo "Start: $(date)"

SMILES_FILE="data/sources/coconut.smi"
TRAIN_DATA="data/coconut_train"
TEST_DATA="data/coconut_test"
OUT_DIR="output/domain_transfer/coconut"
ZINC_MODEL="output/multi_seed/llama-8b/seed_42/final"

if [ ! -f "$SMILES_FILE" ]; then
    echo "ERROR: $SMILES_FILE not found. Place COCONUT SMILES file there first."
    exit 1
fi

# Step 1: Prepare COCONUT training data (50K samples per task)
if [ ! -f "$TRAIN_DATA/functional_group.json" ]; then
    echo "=== Preparing COCONUT training data ==="
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
    echo "SKIP: COCONUT training data exists"
fi

# Step 2: Prepare COCONUT test data (10K samples, non-overlapping)
if [ ! -f "$TEST_DATA/functional_group.json" ]; then
    echo "=== Preparing COCONUT test data ==="
    # Use a different seed + skip to get non-overlapping molecules
    python -m clarimol prepare \
        --source file \
        --smiles-file "$SMILES_FILE" \
        --output-dir "$TEST_DATA" \
        --max-molecules 80000 \
        --keep-n 10000 \
        --subsample random \
        --no-curriculum \
        --seed 999
else
    echo "SKIP: COCONUT test data exists"
fi

# Step 3: Train LLaMA on COCONUT
if [ ! -d "$OUT_DIR/final" ]; then
    echo "=== Training LLaMA on COCONUT ==="
    mkdir -p "$OUT_DIR"
    RESUME_FLAG=""
    if ls "$OUT_DIR"/checkpoint-* 1>/dev/null 2>&1; then
        RESUME_FLAG="--resume"
    fi
    python -m clarimol train \
        --model meta-llama/Llama-3.1-8B-Instruct \
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
    echo "SKIP: COCONUT training complete"
fi

# Step 4: Eval COCONUT-trained model on COCONUT test
if [ ! -f "$OUT_DIR/coconut_on_coconut.json" ]; then
    echo "=== Eval COCONUT model on COCONUT test ==="
    python -m clarimol evaluate \
        --model-path "$OUT_DIR/final" \
        --data-dir "$TEST_DATA" \
        --output-file "$OUT_DIR/coconut_on_coconut.json" \
        --no-unsloth \
        --batch-size 16
else
    echo "SKIP: coconut_on_coconut.json exists"
fi

# Step 5: Eval COCONUT-trained model on ZINC test (cross-domain)
if [ ! -f "$OUT_DIR/coconut_on_zinc.json" ]; then
    echo "=== Eval COCONUT model on ZINC test ==="
    python -m clarimol evaluate \
        --model-path "$OUT_DIR/final" \
        --data-dir data/test \
        --output-file "$OUT_DIR/coconut_on_zinc.json" \
        --no-unsloth \
        --batch-size 16
else
    echo "SKIP: coconut_on_zinc.json exists"
fi

# Step 6: Eval ZINC-trained model on COCONUT test (zero-shot transfer)
if [ -d "$ZINC_MODEL" ] && [ ! -f "$OUT_DIR/zinc_on_coconut.json" ]; then
    echo "=== Eval ZINC model on COCONUT test ==="
    python -m clarimol evaluate \
        --model-path "$ZINC_MODEL" \
        --data-dir "$TEST_DATA" \
        --output-file "$OUT_DIR/zinc_on_coconut.json" \
        --no-unsloth \
        --batch-size 16
elif [ ! -d "$ZINC_MODEL" ]; then
    echo "SKIP: ZINC model not found at $ZINC_MODEL"
else
    echo "SKIP: zinc_on_coconut.json exists"
fi

echo ""
echo "=== COCONUT Domain Transfer Results ==="
for f in "$OUT_DIR"/*.json; do
    [ -f "$f" ] || continue
    echo "--- $(basename "$f") ---"
    python3 -c "
import json, sys
d = json.load(open('$f'))
for t in sorted(d.keys()):
    if 'accuracy' in d[t]:
        print(f'  {t}: {d[t][\"accuracy\"]:.4f}')
accs = [d[t]['accuracy'] for t in d if 'accuracy' in d[t]]
if accs: print(f'  mean: {sum(accs)/len(accs):.4f}')
" 2>/dev/null
done

echo "=== Complete at $(date) ==="
