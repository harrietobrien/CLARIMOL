#!/bin/bash
#SBATCH --job-name=cm_chembl_ada
#SBATCH --partition=scavenger-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:a6000:1
#SBATCH --time=24:00:00
#SBATCH --output=output/logs/domain_shift/chembl_shift_ada_%j.log
#SBATCH --error=output/logs/domain_shift/chembl_shift_ada_%j.err
#SBATCH --requeue
#
# ChEMBL domain shift on Ada GPU (4-bit quantization to fit 48GB)
# PREREQUISITE: data/chembl_250k/ prepared, multi_seed llama model exists
# Preemption-safe: skips completed steps, resumes from checkpoint
#

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p output/logs/domain_shift

export CONDARC=/work/gc237/.condarc
export HF_HOME=/work/gc237/.cache/huggingface
export HF_TOKEN=$(cat /work/gc237/.cache/huggingface/token)

module load Anaconda3 2>/dev/null || module load conda 2>/dev/null || true
eval "$(conda shell.bash hook)"
conda activate /work/gc237/conda_envs/clarimol

MODEL="meta-llama/Llama-3.1-8B-Instruct"
ZINC_MODEL="output/multi_seed/llama-8b/seed_42/final"
CHEMBL_DATA="data/chembl_250k"
BASE_OUT="output/chembl_domain_shift"
SEED=42

echo "=== ChEMBL Domain Shift (Ada) ==="
nvidia-smi
echo "Start: $(date)"

if [ ! -d "$CHEMBL_DATA" ] || [ -z "$(ls -A "$CHEMBL_DATA" 2>/dev/null)" ]; then
    echo "ERROR: ChEMBL data not found at $CHEMBL_DATA"
    exit 1
fi

mkdir -p "$BASE_OUT"

# Step 1: Zero-shot baseline
if [ -f "$BASE_OUT/zero_shot_chembl.json" ]; then
    echo "SKIP: zero_shot_chembl.json exists"
else
    echo ""
    echo "=== Zero-shot on ChEMBL ==="
    python -m clarimol evaluate \
        --model-path "$MODEL" \
        --data-dir "$CHEMBL_DATA" \
        --output-file "$BASE_OUT/zero_shot_chembl.json" \
        --no-unsloth --batch-size 4
fi

# Step 2: ZINC-trained on ChEMBL
if [ -f "$BASE_OUT/zinc_on_chembl.json" ]; then
    echo "SKIP: zinc_on_chembl.json exists"
elif [ -d "$ZINC_MODEL" ]; then
    echo ""
    echo "=== ZINC-trained -> ChEMBL ==="
    python -m clarimol evaluate \
        --model-path "$ZINC_MODEL" \
        --data-dir "$CHEMBL_DATA" \
        --output-file "$BASE_OUT/zinc_on_chembl.json" \
        --no-unsloth --batch-size 8
fi

# Step 3: Fine-tune on ChEMBL
if [ ! -d "$BASE_OUT/chembl_finetuned/final" ]; then
    echo ""
    echo "=== Fine-tuning on ChEMBL ==="
    RESUME_ARG=""
    if ls "$BASE_OUT"/chembl_finetuned/checkpoint-* 1>/dev/null 2>&1; then
        echo "Found checkpoints, will resume"
        RESUME_ARG="--resume"
    fi
    python -m clarimol train \
        --model "$MODEL" \
        --data-dir "$CHEMBL_DATA" \
        --output-dir "$BASE_OUT/chembl_finetuned" \
        --no-unsloth --max-length 512 \
        --batch-size 8 --grad-accum 2 \
        --lr 1e-4 --epochs 1 \
        --lora-r 64 --lora-alpha 16 \
        --no-fp16 --bf16 --no-wandb \
        --save-steps 100 --seed "$SEED" $RESUME_ARG
else
    echo "SKIP: chembl_finetuned/final exists"
fi

# Step 4: ChEMBL-finetuned on ChEMBL
if [ -f "$BASE_OUT/chembl_on_chembl.json" ]; then
    echo "SKIP: chembl_on_chembl.json exists"
else
    echo ""
    echo "=== ChEMBL-finetuned -> ChEMBL ==="
    python -m clarimol evaluate \
        --model-path "$BASE_OUT/chembl_finetuned/final" \
        --data-dir "$CHEMBL_DATA" \
        --output-file "$BASE_OUT/chembl_on_chembl.json" \
        --no-unsloth --batch-size 8
fi

# Step 5: ChEMBL-finetuned on ZINC (regression)
if [ -f "$BASE_OUT/chembl_on_zinc.json" ]; then
    echo "SKIP: chembl_on_zinc.json exists"
else
    echo ""
    echo "=== ChEMBL-finetuned -> ZINC (regression) ==="
    python -m clarimol evaluate \
        --model-path "$BASE_OUT/chembl_finetuned/final" \
        --data-dir data/test \
        --output-file "$BASE_OUT/chembl_on_zinc.json" \
        --no-unsloth --batch-size 8
fi

echo ""
echo "=== ChEMBL domain shift complete at $(date) ==="
for F in zero_shot_chembl zinc_on_chembl chembl_on_chembl chembl_on_zinc; do
    echo -n "  $F: "
    python -c "
import json, pathlib
p = pathlib.Path('$BASE_OUT/${F}.json')
if p.exists():
    with open(p) as f: d = json.load(f)
    accs = [v['accuracy'] for v in d.values()]
    print(f'mean={sum(accs)/len(accs):.4f}')
else:
    print('not run')
" 2>/dev/null || echo "error"
done
