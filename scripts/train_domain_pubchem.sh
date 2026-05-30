#!/bin/bash
#SBATCH --job-name=cm_dom_pubchem
#SBATCH -A scavenger-h200
#SBATCH -p scavenger-h200
#SBATCH --gres=gpu:h200:1
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --exclude=dcc-h200-gpu-05
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=output/logs/domain_transfer/pubchem_%j.log
#SBATCH --error=output/logs/domain_transfer/pubchem_%j.err
#
# Domain transfer: PubChem 1M subsample
# 1. Prepare PubChem SMILES into 5-task parsing format
# 2. Train LLaMA on PubChem data
# 3. Eval PubChem-trained model on PubChem test + ZINC test (bidirectional)
# 4. Eval ZINC-trained model on PubChem test (zero-shot transfer)
#
# Prereq: data/sources/pubchem.smi must exist (one SMILES per line)
# Submit: sbatch scripts/train_domain_pubchem.sh

set -euo pipefail
cd ~/storage/CLARIMOL
mkdir -p output/logs/domain_transfer

export CONDARC=/work/gc237/.condarc
export HF_HOME=/work/gc237/.cache/huggingface
export HF_TOKEN=$(cat /work/gc237/.cache/huggingface/token 2>/dev/null || cat ~/.cache/huggingface/token)

source /opt/apps/rhel9/Anaconda3-2024.02/etc/profile.d/conda.sh
conda activate /work/gc237/conda_envs/clarimol

echo "=== Domain Transfer: PubChem 1M Subsample ==="
nvidia-smi
echo "Start: $(date)"

SMILES_FILE="data/sources/pubchem.smi"
TRAIN_DATA="data/pubchem_train"
TEST_DATA="data/pubchem_test"
OUT_DIR="output/domain_transfer/pubchem"
ZINC_MODEL="output/multi_seed/llama-8b/seed_42/final"

if [ ! -f "$SMILES_FILE" ]; then
    echo "ERROR: $SMILES_FILE not found. Place PubChem SMILES file there first."
    exit 1
fi

# Step 1: Prepare PubChem training data (50K samples per task from 1M pool)
if [ ! -f "$TRAIN_DATA/functional_group.json" ]; then
    echo "=== Preparing PubChem training data ==="
    python -m clarimol prepare \
        --source file \
        --smiles-file "$SMILES_FILE" \
        --output-dir "$TRAIN_DATA" \
        --max-molecules 1000000 \
        --keep-n 50000 \
        --subsample random \
        --no-curriculum \
        --seed 42
else
    echo "SKIP: PubChem training data exists"
fi

# Step 2: Prepare PubChem test data (10K samples, different seed for non-overlap)
if [ ! -f "$TEST_DATA/functional_group.json" ]; then
    echo "=== Preparing PubChem test data ==="
    python -m clarimol prepare \
        --source file \
        --smiles-file "$SMILES_FILE" \
        --output-dir "$TEST_DATA" \
        --max-molecules 1000000 \
        --keep-n 10000 \
        --subsample random \
        --no-curriculum \
        --seed 999
else
    echo "SKIP: PubChem test data exists"
fi

# Step 3: Train LLaMA on PubChem
if [ ! -d "$OUT_DIR/final" ]; then
    echo "=== Training LLaMA on PubChem ==="
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
    echo "SKIP: PubChem training complete"
fi

# Step 4: Eval PubChem-trained model on PubChem test
if [ ! -f "$OUT_DIR/pubchem_on_pubchem.json" ]; then
    echo "=== Eval PubChem model on PubChem test ==="
    python -m clarimol evaluate \
        --model-path "$OUT_DIR/final" \
        --data-dir "$TEST_DATA" \
        --output-file "$OUT_DIR/pubchem_on_pubchem.json" \
        --no-unsloth \
        --batch-size 16
else
    echo "SKIP: pubchem_on_pubchem.json exists"
fi

# Step 5: Eval PubChem-trained model on ZINC test (cross-domain)
if [ ! -f "$OUT_DIR/pubchem_on_zinc.json" ]; then
    echo "=== Eval PubChem model on ZINC test ==="
    python -m clarimol evaluate \
        --model-path "$OUT_DIR/final" \
        --data-dir data/test \
        --output-file "$OUT_DIR/pubchem_on_zinc.json" \
        --no-unsloth \
        --batch-size 16
else
    echo "SKIP: pubchem_on_zinc.json exists"
fi

# Step 6: Eval ZINC-trained model on PubChem test (zero-shot transfer)
if [ -d "$ZINC_MODEL" ] && [ ! -f "$OUT_DIR/zinc_on_pubchem.json" ]; then
    echo "=== Eval ZINC model on PubChem test ==="
    python -m clarimol evaluate \
        --model-path "$ZINC_MODEL" \
        --data-dir "$TEST_DATA" \
        --output-file "$OUT_DIR/zinc_on_pubchem.json" \
        --no-unsloth \
        --batch-size 16
elif [ ! -d "$ZINC_MODEL" ]; then
    echo "SKIP: ZINC model not found at $ZINC_MODEL"
else
    echo "SKIP: zinc_on_pubchem.json exists"
fi

echo ""
echo "=== PubChem Domain Transfer Results ==="
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
