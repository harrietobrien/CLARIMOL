#!/bin/bash
#SBATCH --job-name=cm_dom_hmdb
#SBATCH -A scavenger-h200
#SBATCH -p scavenger-h200
#SBATCH --gres=gpu:h200:1
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --exclude=dcc-h200-gpu-05
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=output/logs/domain_transfer/hmdb_%j.log
#SBATCH --error=output/logs/domain_transfer/hmdb_%j.err
#
# Domain transfer: HMDB metabolites
# HMDB is smaller (~5K unique structures), so this script does:
# 1. Eval-only: ZINC-trained model on HMDB test (zero-shot transfer)
# 2. In-domain training on HMDB (smaller dataset, fewer samples per task)
# 3. Eval HMDB-trained model on HMDB + ZINC (bidirectional)
#
# Prereq: data/sources/hmdb.smi must exist (one SMILES per line)
# Submit: sbatch scripts/train_domain_hmdb.sh

set -euo pipefail
cd ~/storage/CLARIMOL
mkdir -p output/logs/domain_transfer

export CONDARC=/work/gc237/.condarc
export HF_HOME=/work/gc237/.cache/huggingface
export HF_TOKEN=$(cat /work/gc237/.cache/huggingface/token 2>/dev/null || cat ~/.cache/huggingface/token)

source /opt/apps/rhel9/Anaconda3-2024.02/etc/profile.d/conda.sh
conda activate /work/gc237/conda_envs/clarimol

echo "=== Domain Transfer: HMDB Metabolites ==="
nvidia-smi
echo "Start: $(date)"

SMILES_FILE="data/sources/hmdb.smi"
TRAIN_DATA="data/hmdb_train"
TEST_DATA="data/hmdb_test"
OUT_DIR="output/domain_transfer/hmdb"
ZINC_MODEL="output/multi_seed/llama-8b/seed_42/final"

if [ ! -f "$SMILES_FILE" ]; then
    echo "ERROR: $SMILES_FILE not found. Place HMDB SMILES file there first."
    exit 1
fi

# Count available molecules
TOTAL_SMILES=$(wc -l < "$SMILES_FILE")
echo "HMDB SMILES file: $TOTAL_SMILES lines"

# HMDB is small (~5K structures). Use 80/20 train/test split.
# Limit keep-n to available molecules (may be fewer than 50K).
TRAIN_KEEP=$(python3 -c "print(min(50000, int($TOTAL_SMILES * 0.8)))")
TEST_KEEP=$(python3 -c "print(min(10000, int($TOTAL_SMILES * 0.2)))")
echo "Train keep: $TRAIN_KEEP, Test keep: $TEST_KEEP"

# Step 1: Prepare HMDB training data
if [ ! -f "$TRAIN_DATA/functional_group.json" ]; then
    echo "=== Preparing HMDB training data ==="
    python -m clarimol prepare \
        --source file \
        --smiles-file "$SMILES_FILE" \
        --output-dir "$TRAIN_DATA" \
        --max-molecules "$TOTAL_SMILES" \
        --keep-n "$TRAIN_KEEP" \
        --subsample random \
        --no-curriculum \
        --seed 42
else
    echo "SKIP: HMDB training data exists"
fi

# Step 2: Prepare HMDB test data
if [ ! -f "$TEST_DATA/functional_group.json" ]; then
    echo "=== Preparing HMDB test data ==="
    python -m clarimol prepare \
        --source file \
        --smiles-file "$SMILES_FILE" \
        --output-dir "$TEST_DATA" \
        --max-molecules "$TOTAL_SMILES" \
        --keep-n "$TEST_KEEP" \
        --subsample random \
        --no-curriculum \
        --seed 999
else
    echo "SKIP: HMDB test data exists"
fi

# Step 3: Eval ZINC-trained model on HMDB test (zero-shot transfer)
if [ -d "$ZINC_MODEL" ] && [ ! -f "$OUT_DIR/zinc_on_hmdb.json" ]; then
    echo "=== Eval ZINC model on HMDB test (zero-shot) ==="
    mkdir -p "$OUT_DIR"
    python -m clarimol evaluate \
        --model-path "$ZINC_MODEL" \
        --data-dir "$TEST_DATA" \
        --output-file "$OUT_DIR/zinc_on_hmdb.json" \
        --no-unsloth \
        --batch-size 16
elif [ ! -d "$ZINC_MODEL" ]; then
    echo "SKIP: ZINC model not found at $ZINC_MODEL"
else
    echo "SKIP: zinc_on_hmdb.json exists"
fi

# Step 4: Train LLaMA on HMDB
if [ ! -d "$OUT_DIR/final" ]; then
    echo "=== Training LLaMA on HMDB ==="
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
    echo "SKIP: HMDB training complete"
fi

# Step 5: Eval HMDB-trained model on HMDB test (in-domain)
if [ ! -f "$OUT_DIR/hmdb_on_hmdb.json" ]; then
    echo "=== Eval HMDB model on HMDB test ==="
    python -m clarimol evaluate \
        --model-path "$OUT_DIR/final" \
        --data-dir "$TEST_DATA" \
        --output-file "$OUT_DIR/hmdb_on_hmdb.json" \
        --no-unsloth \
        --batch-size 16
else
    echo "SKIP: hmdb_on_hmdb.json exists"
fi

# Step 6: Eval HMDB-trained model on ZINC test (cross-domain regression)
if [ ! -f "$OUT_DIR/hmdb_on_zinc.json" ]; then
    echo "=== Eval HMDB model on ZINC test ==="
    python -m clarimol evaluate \
        --model-path "$OUT_DIR/final" \
        --data-dir data/test \
        --output-file "$OUT_DIR/hmdb_on_zinc.json" \
        --no-unsloth \
        --batch-size 16
else
    echo "SKIP: hmdb_on_zinc.json exists"
fi

echo ""
echo "=== HMDB Domain Transfer Results ==="
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
