#!/bin/bash
#SBATCH --job-name=cm_cod_shift
#SBATCH -A scavenger-h200
#SBATCH -p scavenger-h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=24:00:00
#SBATCH --output=output/cod_shift_%j.log
#SBATCH --error=output/cod_shift_%j.err
#SBATCH --requeue
#
# P1-11: COD domain shift experiment
# Does a ZINC-trained model transfer to crystal molecules?
#   1. Evaluate ZINC-trained LLaMA on COD
#   2. Fine-tune on COD
#   3. Evaluate COD-finetuned on COD + ZINC (regression check)
#
# PREREQUISITE: run job_multi_seed_llama.slurm first (needs the seed_42 model)
#
# Can pair with job_h200_70b.slurm on the other H200:
#   sbatch training_duke/job_cod_domain_shift.slurm
#   sbatch training_duke/job_h200_70b.slurm

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

export CONDARC=/work/gc237/.condarc
export HF_HOME=/work/gc237/.cache/huggingface
export HF_TOKEN=$(cat /work/gc237/.cache/huggingface/token)

module load Anaconda3 2>/dev/null || module load conda 2>/dev/null || true
eval "$(conda shell.bash hook)"
conda activate /work/gc237/conda_envs/clarimol

MODEL="meta-llama/Llama-3.1-8B-Instruct"
ZINC_MODEL="output/multi_seed/llama-8b/seed_42/final"
COD_DATA="data/cod"
BASE_OUT="output/cod_domain_shift"
SEED=42

echo "=== COD Domain Shift Experiment ==="
nvidia-smi
echo "Start: $(date)"

# ─── Step 0: Prepare COD dataset ─────────────────────────────────────────────
if [ ! -d "$COD_DATA" ] || [ -z "$(ls -A "$COD_DATA" 2>/dev/null)" ]; then
    echo "Preparing COD dataset (5000 molecules)..."
    python -m clarimol prepare \
        --source cod \
        --max-molecules 5000 \
        --output-dir "$COD_DATA" \
        --cod-cache-dir data/cod_cache \
        --seed "$SEED"
fi

mkdir -p "$BASE_OUT"

# ─── Step 1: Zero-shot baseline on COD ───────────────────────────────────────
echo ""
echo "=== Zero-shot on COD ==="
python -m clarimol evaluate \
    --model-path "$MODEL" \
    --data-dir "$COD_DATA" \
    --output-file "$BASE_OUT/zero_shot_cod.json" \
    --no-unsloth --batch-size 8

# ─── Step 2: ZINC-trained model on COD ───────────────────────────────────────
if [ -d "$ZINC_MODEL" ]; then
    echo ""
    echo "=== ZINC-trained model on COD ==="
    python -m clarimol evaluate \
        --model-path "$ZINC_MODEL" \
        --data-dir "$COD_DATA" \
        --output-file "$BASE_OUT/zinc_on_cod.json" \
        --no-unsloth --batch-size 16
else
    echo "WARNING: ZINC model not found at $ZINC_MODEL"
    echo "Run job_multi_seed_llama.slurm first"
fi

# ─── Step 3: Fine-tune on COD ────────────────────────────────────────────────
echo ""
echo "=== Fine-tuning on COD ==="
python -m clarimol train \
    --model "$MODEL" \
    --data-dir "$COD_DATA" \
    --output-dir "$BASE_OUT/cod_finetuned" \
    --no-unsloth --max-length 512 \
    --batch-size 16 --grad-accum 1 \
    --lr 1e-4 --epochs 1 \
    --lora-r 64 --lora-alpha 16 \
    --bf16 --no-4bit --no-wandb \
    --save-steps 500 --seed "$SEED"

# ─── Step 4: Evaluate COD-finetuned model ────────────────────────────────────
echo ""
echo "=== COD-finetuned on COD ==="
python -m clarimol evaluate \
    --model-path "$BASE_OUT/cod_finetuned/final" \
    --data-dir "$COD_DATA" \
    --output-file "$BASE_OUT/cod_on_cod.json" \
    --no-unsloth --batch-size 16

echo ""
echo "=== COD-finetuned on ZINC (regression check) ==="
python -m clarimol evaluate \
    --model-path "$BASE_OUT/cod_finetuned/final" \
    --data-dir data/test \
    --output-file "$BASE_OUT/cod_on_zinc.json" \
    --no-unsloth --batch-size 16

echo ""
echo "=== COD domain shift complete at $(date) ==="
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
