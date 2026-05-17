#!/bin/bash
#SBATCH --job-name=cm_cod_ada
#SBATCH --partition=scavenger-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:a6000:1
#SBATCH --time=12:00:00
#SBATCH --output=output/logs/cod_shift/cod_82k_shift_ada_%j.log
#SBATCH --error=output/logs/cod_shift/cod_82k_shift_ada_%j.err
#SBATCH --requeue
#
# COD 82K domain shift on Ada GPU
# PREREQUISITE: data/cod_82k/ uploaded, multi_seed llama model exists
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
COD_DATA="data/cod_82k"
BASE_OUT="output/cod_domain_shift_82k"
SEED=42

echo "=== COD 82K Domain Shift (Ada) ==="
nvidia-smi
echo "Start: $(date)"

if [ ! -d "$COD_DATA" ] || [ -z "$(ls -A "$COD_DATA" 2>/dev/null)" ]; then
    echo "ERROR: COD 82K data not found at $COD_DATA"
    exit 1
fi

mkdir -p "$BASE_OUT"

# Zero-shot baseline
echo ""
echo "=== Zero-shot on COD 82K ==="
python -m clarimol evaluate \
    --model-path "$MODEL" \
    --data-dir "$COD_DATA" \
    --output-file "$BASE_OUT/zero_shot_cod.json" \
    --no-unsloth --batch-size 4

# ZINC-trained on COD
if [ -d "$ZINC_MODEL" ]; then
    echo ""
    echo "=== ZINC-trained → COD 82K ==="
    python -m clarimol evaluate \
        --model-path "$ZINC_MODEL" \
        --data-dir "$COD_DATA" \
        --output-file "$BASE_OUT/zinc_on_cod.json" \
        --no-unsloth --batch-size 8
fi

# Fine-tune on COD
echo ""
echo "=== Fine-tuning on COD 82K ==="
python -m clarimol train \
    --model "$MODEL" \
    --data-dir "$COD_DATA" \
    --output-dir "$BASE_OUT/cod_finetuned" \
    --no-unsloth --max-length 512 \
    --batch-size 8 --grad-accum 2 \
    --lr 1e-4 --epochs 1 \
    --lora-r 64 --lora-alpha 16 \
    --no-fp16 --bf16 --no-wandb \
    --save-steps 500 --seed "$SEED"

# COD-finetuned on COD
echo ""
echo "=== COD-finetuned → COD 82K ==="
python -m clarimol evaluate \
    --model-path "$BASE_OUT/cod_finetuned/final" \
    --data-dir "$COD_DATA" \
    --output-file "$BASE_OUT/cod_on_cod.json" \
    --no-unsloth --batch-size 8

# COD-finetuned on ZINC (regression)
echo ""
echo "=== COD-finetuned → ZINC (regression) ==="
python -m clarimol evaluate \
    --model-path "$BASE_OUT/cod_finetuned/final" \
    --data-dir data/test \
    --output-file "$BASE_OUT/cod_on_zinc.json" \
    --no-unsloth --batch-size 8

echo ""
echo "=== COD 82K domain shift complete at $(date) ==="
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
