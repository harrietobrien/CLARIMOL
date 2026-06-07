#!/bin/bash
#SBATCH --job-name=cm_mixed_domain
#SBATCH -A scavenger-h200
#SBATCH -p scavenger-h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=24:00:00
#SBATCH --output=output/logs/domain_shift/mixed_domain_%j.log
#SBATCH --error=output/logs/domain_shift/mixed_domain_%j.err
#SBATCH --requeue
#
# Mixed-domain training: ZINC + COD combined, then evaluate on both
# Tests whether domain mixing improves cross-domain generalization
#
# Submit: sbatch scripts/train_mixed_domain.sh

set -euo pipefail
cd ~/storage/CLARIMOL
mkdir -p output/logs/domain_shift

export CONDARC=/work/gc237/.condarc
export HF_HOME=/work/gc237/.cache/huggingface
export HF_TOKEN=$(cat /work/gc237/.cache/huggingface/token 2>/dev/null || cat ~/.cache/huggingface/token)

source /opt/apps/rhel9/Anaconda3-2024.02/etc/profile.d/conda.sh
conda activate /work/gc237/conda_envs/clarimol

echo "=== Mixed-Domain Training: ZINC + COD ==="
nvidia-smi
echo "Start: $(date)"

# Step 1: Prepare mixed training data (merge ZINC + COD)
MIXED_DIR="data/mixed_zinc_cod"
if [ ! -f "$MIXED_DIR/functional_group.json" ]; then
    echo "=== Preparing mixed training data ==="
    mkdir -p "$MIXED_DIR"
    for task in functional_group ring_counting chain_length canonicalization fragment_assembly; do
        python3 -c "
import json
zinc = json.load(open('data/clarimol/${task}.json'))
cod_path = 'data/cod_250k/${task}.json'
import os
if os.path.exists(cod_path):
    cod = json.load(open(cod_path))
    # Take 50K from each to keep balanced (50K ZINC + 50K COD = 100K per task)
    mixed = zinc[:50000] + cod[:50000]
    print(f'  ${task}: {len(zinc[:50000])} ZINC + {len(cod[:50000])} COD = {len(mixed)}')
else:
    mixed = zinc[:50000]
    print(f'  ${task}: {len(mixed)} ZINC only (no COD data)')
json.dump(mixed, open('${MIXED_DIR}/${task}.json', 'w'))
"
    done
    echo "Mixed data prepared"
else
    echo "SKIP: mixed data exists"
fi

# Step 2: Train on mixed data
OUT_DIR="output/mixed_domain"
if [ ! -d "$OUT_DIR/final" ]; then
    echo "=== Training on mixed ZINC+COD ==="
    RESUME_FLAG=""
    if ls "$OUT_DIR"/checkpoint-* 1>/dev/null 2>&1; then
        RESUME_FLAG="--resume"
    fi
    python -m clarimol train \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --data-dir "$MIXED_DIR" \
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
    echo "SKIP: training complete"
fi

# Step 3: Evaluate on ZINC test
if [ ! -f "$OUT_DIR/mixed_on_zinc.json" ]; then
    echo "=== Eval mixed model on ZINC ==="
    python -m clarimol evaluate \
        --model-path "$OUT_DIR/final" \
        --data-dir data/test \
        --output-file "$OUT_DIR/mixed_on_zinc.json" \
        --no-unsloth \
        --batch-size 16
fi

# Step 4: Evaluate on COD test
COD_TEST="data/cod"
if [ -d "$COD_TEST" ] && [ ! -f "$OUT_DIR/mixed_on_cod.json" ]; then
    echo "=== Eval mixed model on COD ==="
    python -m clarimol evaluate \
        --model-path "$OUT_DIR/final" \
        --data-dir "$COD_TEST" \
        --output-file "$OUT_DIR/mixed_on_cod.json" \
        --no-unsloth \
        --batch-size 16
fi

# Step 5: Evaluate on ChEMBL
CHEMBL_TEST="data/chembl_250k"
if [ -d "$CHEMBL_TEST" ] && [ ! -f "$OUT_DIR/mixed_on_chembl.json" ]; then
    echo "=== Eval mixed model on ChEMBL ==="
    python -m clarimol evaluate \
        --model-path "$OUT_DIR/final" \
        --data-dir "$CHEMBL_TEST" \
        --output-file "$OUT_DIR/mixed_on_chembl.json" \
        --no-unsloth \
        --max-samples 10000 \
        --batch-size 16
fi

echo ""
echo "=== Results ==="
for f in "$OUT_DIR"/*.json; do
    echo "--- $(basename $f) ---"
    python3 -c "
import json
d = json.load(open('$f'))
for t in sorted(d.keys()):
    if 'accuracy' in d[t]:
        print(f'  {t}: {d[t][\"accuracy\"]:.4f}')
accs = [d[t]['accuracy'] for t in d if 'accuracy' in d[t]]
if accs: print(f'  mean: {sum(accs)/len(accs):.4f}')
" 2>/dev/null
done

echo "=== Complete at $(date) ==="
