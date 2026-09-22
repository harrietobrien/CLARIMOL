#!/bin/bash
#SBATCH --job-name=cm_head_ablation
#SBATCH --partition=scavenger-h200
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=output/logs/probing/head_ablation_%j.log
#SBATCH --error=output/logs/probing/head_ablation_%j.err
#SBATCH --requeue

source /opt/apps/rhel9/Anaconda3-2024.02/etc/profile.d/conda.sh
export CONDARC=/work/gc237/.condarc
conda activate /work/gc237/conda_envs/clarimol

cd ~/storage/CLARIMOL

mkdir -p output/logs/probing
mkdir -p output/ablation_heads

echo "Causal head ablation experiment"
echo "Adapter: output/multi_seed/llama-8b/seed_42/final"
echo "Test data: data/test"
echo "Output: output/ablation_heads/ablation_results.json"

python scripts/eval_causal_ablation.py \
    --adapter-path output/multi_seed/llama-8b/seed_42/final \
    --test-data data/test \
    --output-dir output/ablation_heads \
    --max-samples 500 \
    --batch-size 16 \
    --n-random-controls 5 \
    --seed 42

echo "Causal head ablation complete."
