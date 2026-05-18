#!/bin/bash
#SBATCH --job-name=cm_qm9
#SBATCH -A scavenger-h200
#SBATCH -p scavenger-h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=24:00:00
#SBATCH --output=output/logs/domain_shift/qm9_shift_%j.log
#SBATCH --error=output/logs/domain_shift/qm9_shift_%j.err
#SBATCH --requeue
#SBATCH --exclude=dcc-h200-gpu-05
#
# QM9 domain shift: how does ZINC-trained model transfer to small molecules?
# PREREQUISITE: data/qm9/ must be uploaded, job_multi_seed_llama.slurm must have run
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
QM9_DATA="data/qm9"
BASE_OUT="output/qm9_domain_shift"
SEED=42

echo "=== QM9 Domain Shift Experiment ==="
nvidia-smi
echo "Start: $(date)"

if [ ! -d "$QM9_DATA" ] || [ -z "$(ls -A "$QM9_DATA" 2>/dev/null)" ]; then
    echo "ERROR: QM9 data not found at $QM9_DATA"
    exit 1
fi

mkdir -p "$BASE_OUT"

# Zero-shot baseline
echo ""
echo "=== Zero-shot on QM9 ==="
python -m clarimol evaluate \
    --model-path "$MODEL" \
    --data-dir "$QM9_DATA" \
    --output-file "$BASE_OUT/zero_shot_qm9.json" \
    --no-unsloth --batch-size 8

# ZINC-trained on QM9
if [ -d "$ZINC_MODEL" ]; then
    echo ""
    echo "=== ZINC-trained → QM9 ==="
    python -m clarimol evaluate \
        --model-path "$ZINC_MODEL" \
        --data-dir "$QM9_DATA" \
        --output-file "$BASE_OUT/zinc_on_qm9.json" \
        --no-unsloth --batch-size 16
fi

# Fine-tune on QM9
echo ""
echo "=== Fine-tuning on QM9 ==="
python -m clarimol train \
    --model "$MODEL" \
    --data-dir "$QM9_DATA" \
    --output-dir "$BASE_OUT/qm9_finetuned" \
    --no-unsloth --max-length 512 \
    --batch-size 16 --grad-accum 1 \
    --lr 1e-4 --epochs 1 \
    --lora-r 64 --lora-alpha 16 \
    --bf16 --no-4bit --no-wandb \
    --save-steps 500 --seed "$SEED"

# QM9-finetuned on QM9
echo ""
echo "=== QM9-finetuned → QM9 ==="
python -m clarimol evaluate \
    --model-path "$BASE_OUT/qm9_finetuned/final" \
    --data-dir "$QM9_DATA" \
    --output-file "$BASE_OUT/qm9_on_qm9.json" \
    --no-unsloth --batch-size 16

# QM9-finetuned on ZINC (regression)
echo ""
echo "=== QM9-finetuned → ZINC (regression) ==="
python -m clarimol evaluate \
    --model-path "$BASE_OUT/qm9_finetuned/final" \
    --data-dir data/test \
    --output-file "$BASE_OUT/qm9_on_zinc.json" \
    --no-unsloth --batch-size 16

echo ""
echo "=== QM9 domain shift complete at $(date) ==="
for F in zero_shot_qm9 zinc_on_qm9 qm9_on_qm9 qm9_on_zinc; do
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
