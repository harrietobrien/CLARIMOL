#!/bin/bash
#SBATCH --job-name=cm_lo_fg
#SBATCH -A scavenger-h200
#SBATCH -p scavenger-h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=24:00:00
#SBATCH --output=output/logs/task_contrib/leave_out_fg_%j.log
#SBATCH --error=output/logs/task_contrib/leave_out_fg_%j.err
#SBATCH --requeue
#SBATCH --exclude=dcc-h200-gpu-05
#
# Rerun: leave-one-out functional_group (train + eval)
#

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p output/logs/task_contrib

export CONDARC=/work/gc237/.condarc
export HF_HOME=/work/gc237/.cache/huggingface
export HF_TOKEN=$(cat /work/gc237/.cache/huggingface/token)

module load Anaconda3 2>/dev/null || module load conda 2>/dev/null || true
eval "$(conda shell.bash hook)"
conda activate /work/gc237/conda_envs/clarimol

MODEL="meta-llama/Llama-3.1-8B-Instruct"
DATA_DIR="data/leave_out_functional_group"
OUT_DIR="output/ablation_tasks/leave_out_functional_group"
SEED=42

echo "=== Leave-one-out: excluding functional_group ==="
nvidia-smi
echo "Start: $(date)"

# Prepare dataset if needed
if [ ! -d "$DATA_DIR" ] || [ -z "$(ls -A "$DATA_DIR" 2>/dev/null)" ]; then
    python -m clarimol prepare \
        --tasks "ring_counting,chain_length,canonicalization,fragment_assembly" \
        --output-dir "$DATA_DIR" \
        --seed "$SEED"
fi

rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

python -m clarimol train \
    --model "$MODEL" \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUT_DIR" \
    --no-unsloth --max-length 512 \
    --batch-size 16 --grad-accum 1 \
    --lr 1e-4 --epochs 1 \
    --lora-r 64 --lora-alpha 16 \
    --bf16 --no-4bit --no-wandb \
    --save-steps 500 --seed "$SEED"

python -m clarimol evaluate \
    --model-path "$OUT_DIR/final" \
    --data-dir data/test \
    --output-file "$OUT_DIR/results.json" \
    --no-unsloth --batch-size 16

echo "Done: $(date)"

python -c "
import json
with open('$OUT_DIR/results.json') as f:
    d = json.load(f)
for task, v in d.items():
    print(f'  {task}: {v[\"accuracy\"]:.4f}')
accs = [v['accuracy'] for v in d.values()]
print(f'  mean: {sum(accs)/len(accs):.4f}')
"
