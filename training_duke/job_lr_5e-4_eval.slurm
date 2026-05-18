#!/bin/bash
#SBATCH --job-name=cm_lr5e4_ev
#SBATCH -A scavenger-h200
#SBATCH -p scavenger-h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=24:00:00
#SBATCH --output=output/logs/lr_lora/lr_5e-4_eval_%j.log
#SBATCH --error=output/logs/lr_lora/lr_5e-4_eval_%j.err
#SBATCH --requeue
#SBATCH --exclude=dcc-h200-gpu-05
#
# Eval only: lr_5e-4 (training already completed)
#

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p output/logs/lr_lora

export CONDARC=/work/gc237/.condarc
export HF_HOME=/work/gc237/.cache/huggingface
export HF_TOKEN=$(cat /work/gc237/.cache/huggingface/token)

module load Anaconda3 2>/dev/null || module load conda 2>/dev/null || true
eval "$(conda shell.bash hook)"
conda activate /work/gc237/conda_envs/clarimol

OUT_DIR="output/ablation_lr/lr_5e-4"

echo "=== LR=5e-4 Eval Only ==="
nvidia-smi
echo "Start: $(date)"

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
