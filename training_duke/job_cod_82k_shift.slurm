#!/bin/bash
#SBATCH --job-name=cm_cod82k
#SBATCH -A scavenger-h200
#SBATCH -p scavenger-h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=24:00:00
#SBATCH --output=output/logs/cod_shift/cod_82k_shift_%j.log
#SBATCH --error=output/logs/cod_shift/cod_82k_shift_%j.err
#SBATCH --requeue
#
# COD 82K domain shift — full pipeline
# Uses Ada A6000 GPU (scavenger-gpu partition)
#
# PREREQUISITE: scp -r data/cod_82k from local to DCC
#
# Submit: sbatch training_duke/job_cod_82k_shift.slurm

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p output/logs/cod_shift

export CONDARC=/work/gc237/.condarc
export HF_HOME=/work/gc237/.cache/huggingface
export HF_TOKEN=$(cat /work/gc237/.cache/huggingface/token 2>/dev/null || cat ~/.cache/huggingface/token)

module load Anaconda3 2>/dev/null || module load conda 2>/dev/null || true
eval "$(conda shell.bash hook)"
conda activate /work/gc237/conda_envs/clarimol

MODEL="meta-llama/Llama-3.1-8B-Instruct"
ZINC_MODEL="output/multi_seed/llama-8b/seed_42/final"
COD_DATA="data/cod_82k"
BASE_OUT="output/cod_domain_shift_82k"
SEED=42

echo "=== COD 82K Domain Shift ==="
nvidia-smi
echo "Start: $(date)"

if [ ! -d "$COD_DATA" ] || [ -z "$(ls -A "$COD_DATA" 2>/dev/null)" ]; then
    echo "ERROR: COD 82K data not found at $COD_DATA"
    echo "Run: scp -r ~/CLARIMOL/CLARIMOL/data/cod_82k dcc:~/storage/CLARIMOL/data/cod_82k"
    exit 1
fi

mkdir -p "$BASE_OUT"

# Step 1: Zero-shot baseline
if [ -f "$BASE_OUT/zero_shot_cod.json" ]; then
    echo "SKIP: zero_shot_cod.json exists"
else
    echo ""
    echo "=== Step 1: Zero-shot on COD 82K ==="
    python -m clarimol evaluate \
        --model-path "$MODEL" \
        --data-dir "$COD_DATA" \
        --output-file "$BASE_OUT/zero_shot_cod.json" \
        --no-unsloth --batch-size 4
fi

# Step 2: ZINC-trained on COD
if [ -f "$BASE_OUT/zinc_on_cod.json" ]; then
    echo "SKIP: zinc_on_cod.json exists"
elif [ -d "$ZINC_MODEL" ]; then
    echo ""
    echo "=== Step 2: ZINC-trained -> COD 82K ==="
    python -m clarimol evaluate \
        --model-path "$ZINC_MODEL" \
        --data-dir "$COD_DATA" \
        --output-file "$BASE_OUT/zinc_on_cod.json" \
        --no-unsloth --batch-size 8
else
    echo "WARNING: ZINC model not found at $ZINC_MODEL"
fi

# Step 3: Fine-tune on COD 82K
if [ -d "$BASE_OUT/cod_finetuned/final" ]; then
    echo "SKIP: cod_finetuned/final exists"
else
    echo ""
    echo "=== Step 3: Fine-tuning on COD 82K ==="
    RESUME_ARG=""
    if ls "$BASE_OUT"/cod_finetuned/checkpoint-* 1>/dev/null 2>&1; then
        echo "Found checkpoints, will resume"
        RESUME_ARG="--resume"
    fi
    python -m clarimol train \
        --model "$MODEL" \
        --data-dir "$COD_DATA" \
        --output-dir "$BASE_OUT/cod_finetuned" \
        --no-unsloth --max-length 512 \
        --batch-size 8 --grad-accum 2 \
        --lr 1e-4 --epochs 1 \
        --lora-r 64 --lora-alpha 16 \
        --no-4bit --no-wandb \
        --save-steps 500 --seed "$SEED" $RESUME_ARG
fi

# Step 4: COD-finetuned on COD
if [ -f "$BASE_OUT/cod_on_cod.json" ]; then
    echo "SKIP: cod_on_cod.json exists"
else
    echo ""
    echo "=== Step 4: COD-finetuned -> COD 82K ==="
    python -m clarimol evaluate \
        --model-path "$BASE_OUT/cod_finetuned/final" \
        --data-dir "$COD_DATA" \
        --output-file "$BASE_OUT/cod_on_cod.json" \
        --no-unsloth --batch-size 8
fi

# Step 5: COD-finetuned on ZINC
if [ -f "$BASE_OUT/cod_on_zinc.json" ]; then
    echo "SKIP: cod_on_zinc.json exists"
else
    echo ""
    echo "=== Step 5: COD-finetuned -> ZINC ==="
    python -m clarimol evaluate \
        --model-path "$BASE_OUT/cod_finetuned/final" \
        --data-dir data/test \
        --output-file "$BASE_OUT/cod_on_zinc.json" \
        --no-unsloth --batch-size 8
fi

echo ""
echo "=== Results Summary ==="
for F in zero_shot_cod zinc_on_cod cod_on_cod cod_on_zinc; do
    echo -n "  $F: "
    python3 -c "
import json, pathlib
p = pathlib.Path('$BASE_OUT/${F}.json')
if p.exists():
    d = json.load(open(p))
    tasks = ['functional_group','ring_counting','chain_length','canonicalization','fragment_assembly']
    accs = [d[t]['accuracy'] for t in tasks if t in d]
    print(f'mean={sum(accs)/len(accs):.4f}')
else:
    print('not run')
" 2>/dev/null || echo "error"
done

echo "=== Complete at $(date) ==="
