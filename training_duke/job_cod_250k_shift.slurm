#!/bin/bash
#SBATCH --job-name=cm_cod250k_shift
#SBATCH -A scavenger-h200
#SBATCH -p scavenger-h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=24:00:00
#SBATCH --output=output/logs/cod_shift/cod_250k_shift_%j.log
#SBATCH --error=output/logs/cod_shift/cod_250k_shift_%j.err
#SBATCH --requeue
#SBATCH --exclude=dcc-h200-gpu-05
#
# Step 2: COD 250K domain shift experiment (train + cross-eval)
# PREREQUISITE: job_cod_250k_prepare.slurm (needs data/cod_250k)
# PREREQUISITE: job_multi_seed_llama.slurm (needs ZINC-trained model)
# Preemption-safe: skips completed steps, resumes from checkpoint
#

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p output/logs/cod_shift

export CONDARC=/work/gc237/.condarc
export HF_HOME=/work/gc237/.cache/huggingface
export HF_TOKEN=$(cat /work/gc237/.cache/huggingface/token)

module load Anaconda3 2>/dev/null || module load conda 2>/dev/null || true
eval "$(conda shell.bash hook)"
conda activate /work/gc237/conda_envs/clarimol

MODEL="meta-llama/Llama-3.1-8B-Instruct"
ZINC_MODEL="output/multi_seed/llama-8b/seed_42/final"
COD_DATA="data/cod_250k"
BASE_OUT="output/cod_domain_shift_250k"
SEED=42

echo "=== COD 250K Domain Shift Experiment ==="
nvidia-smi
echo "Start: $(date)"

if [ ! -d "$COD_DATA" ] || [ -z "$(ls -A "$COD_DATA" 2>/dev/null)" ]; then
    echo "ERROR: COD 250K data not found at $COD_DATA"
    echo "Run job_cod_250k_prepare.slurm first"
    exit 1
fi

mkdir -p "$BASE_OUT"

# Step 1: Zero-shot baseline on COD 250K
if [ -f "$BASE_OUT/zero_shot_cod.json" ]; then
    echo "SKIP: zero_shot_cod.json exists"
else
    echo ""
    echo "=== Zero-shot on COD 250K ==="
    python -m clarimol evaluate \
        --model-path "$MODEL" \
        --data-dir "$COD_DATA" \
        --output-file "$BASE_OUT/zero_shot_cod.json" \
        --no-unsloth --batch-size 8
fi

# Step 2: ZINC-trained model on COD 250K
if [ -f "$BASE_OUT/zinc_on_cod.json" ]; then
    echo "SKIP: zinc_on_cod.json exists"
elif [ -d "$ZINC_MODEL" ]; then
    echo ""
    echo "=== ZINC-trained -> COD 250K ==="
    python -m clarimol evaluate \
        --model-path "$ZINC_MODEL" \
        --data-dir "$COD_DATA" \
        --output-file "$BASE_OUT/zinc_on_cod.json" \
        --no-unsloth --batch-size 16
else
    echo "WARNING: ZINC model not found at $ZINC_MODEL"
fi

# Step 3: Fine-tune on COD 250K
if [ ! -d "$BASE_OUT/cod_finetuned/final" ]; then
    echo ""
    echo "=== Fine-tuning on COD 250K ==="
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
        --batch-size 16 --grad-accum 1 \
        --lr 1e-4 --epochs 1 \
        --lora-r 64 --lora-alpha 16 \
        --bf16 --no-4bit --no-wandb \
        --save-steps 100 --seed "$SEED" $RESUME_ARG
else
    echo "SKIP: cod_finetuned/final exists"
fi

# Step 4: Evaluate COD-finetuned on COD
if [ -f "$BASE_OUT/cod_on_cod.json" ]; then
    echo "SKIP: cod_on_cod.json exists"
else
    echo ""
    echo "=== COD-finetuned -> COD 250K ==="
    python -m clarimol evaluate \
        --model-path "$BASE_OUT/cod_finetuned/final" \
        --data-dir "$COD_DATA" \
        --output-file "$BASE_OUT/cod_on_cod.json" \
        --no-unsloth --batch-size 16
fi

# Step 5: Regression check -- COD-finetuned on ZINC
if [ -f "$BASE_OUT/cod_on_zinc.json" ]; then
    echo "SKIP: cod_on_zinc.json exists"
else
    echo ""
    echo "=== COD-finetuned -> ZINC (regression) ==="
    python -m clarimol evaluate \
        --model-path "$BASE_OUT/cod_finetuned/final" \
        --data-dir data/test \
        --output-file "$BASE_OUT/cod_on_zinc.json" \
        --no-unsloth --batch-size 16
fi

echo ""
echo "=== COD 250K domain shift complete at $(date) ==="
echo ""
echo "Results summary:"
for F in zero_shot_cod zinc_on_cod cod_on_cod cod_on_zinc; do
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
