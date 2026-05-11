#!/bin/bash
#SBATCH --job-name=cm_seed_olmo
#SBATCH -A scavenger-h200
#SBATCH -p scavenger-h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=24:00:00
#SBATCH --output=output/multi_seed_olmo_%j.log
#SBATCH --error=output/multi_seed_olmo_%j.err
#SBATCH --requeue
#SBATCH --exclude=dcc-h200-gpu-05
#
# Multi-seed OLMo 3 7B Instruct (seeds 42, 137, 2024)
# Fully open model (data + code + weights) from Allen AI
#
# Submit: sbatch training_duke/job_multi_seed_olmo.slurm

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

export CONDARC=/work/gc237/.condarc
export HF_HOME=/work/gc237/.cache/huggingface
export HF_TOKEN=$(cat /work/gc237/.cache/huggingface/token 2>/dev/null || cat ~/.cache/huggingface/token)

module load Anaconda3 2>/dev/null || module load conda 2>/dev/null || true
eval "$(conda shell.bash hook)"
conda activate /work/gc237/conda_envs/clarimol

MODEL="allenai/OLMo-3-7B-Instruct"
BASE_OUT="output/multi_seed/olmo-7b"
SEEDS=(42 137 2024)

echo "=== Multi-seed OLMo 3 7B Instruct ==="
nvidia-smi
echo "Start: $(date)"

for SEED in "${SEEDS[@]}"; do
    SEED_DIR="${BASE_OUT}/seed_${SEED}"
    mkdir -p "$SEED_DIR"

    RESUME_FLAG=""
    if ls "$SEED_DIR"/checkpoint-* 1>/dev/null 2>&1; then
        echo "Checkpoint(s) detected for seed=$SEED — resuming"
        RESUME_FLAG="--resume"
    fi

    echo "--- Seed $SEED: $(date) ---"
    python -m clarimol train \
        --model "$MODEL" \
        --data-dir data/clarimol \
        --output-dir "$SEED_DIR" \
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
        --seed "$SEED" \
        $RESUME_FLAG

    echo "Evaluating seed=$SEED..."
    python -m clarimol evaluate \
        --model-path "$SEED_DIR/final" \
        --data-dir data/test \
        --output-file "$SEED_DIR/results.json" \
        --no-unsloth \
        --batch-size 16

    echo "Seed $SEED done at $(date)"
done

echo ""
echo "=== Aggregating multi-seed results ==="
python -c "
import json, pathlib, statistics
base = pathlib.Path('$BASE_OUT')
seeds = [42, 137, 2024]
all_results = {}
for s in seeds:
    rp = base / f'seed_{s}' / 'results.json'
    if rp.exists():
        with open(rp) as f:
            all_results[s] = json.load(f)
if not all_results:
    print('No results found'); exit()
tasks = list(next(iter(all_results.values())).keys())
summary = {}
for t in tasks:
    accs = [all_results[s][t]['accuracy'] for s in all_results]
    m = statistics.mean(accs)
    sd = statistics.stdev(accs) if len(accs) > 1 else 0.0
    summary[t] = {'mean': m, 'std': sd, 'per_seed': {str(s): all_results[s][t]['accuracy'] for s in all_results}}
    print(f'{t:25s}  {m:.4f} +/- {sd:.4f}')
with open(base / 'summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print(f'Summary saved to {base}/summary.json')
"

echo "=== Complete at $(date) ==="
