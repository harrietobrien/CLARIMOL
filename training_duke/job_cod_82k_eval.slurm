#!/bin/bash
#SBATCH --job-name=cm_cod82k_ev
#SBATCH -A scavenger-h200
#SBATCH -p scavenger-h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=24:00:00
#SBATCH --output=output/logs/cod_shift/cod_82k_eval_%j.log
#SBATCH --error=output/logs/cod_shift/cod_82k_eval_%j.err
#SBATCH --requeue
#
# Eval-only: COD 82K finetuned model on COD and ZINC test sets.
# Training already complete — this produces the missing result JSONs.
#
# Submit: sbatch training_duke/job_cod_82k_eval.slurm

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p output/logs/cod_shift

export CONDARC=/work/gc237/.condarc
export HF_HOME=/work/gc237/.cache/huggingface
export HF_TOKEN=$(cat /work/gc237/.cache/huggingface/token 2>/dev/null || cat ~/.cache/huggingface/token)

module load Anaconda3 2>/dev/null || module load conda 2>/dev/null || true
eval "$(conda shell.bash hook)"
conda activate /work/gc237/conda_envs/clarimol

MODEL="output/cod_domain_shift_82k/cod_finetuned/final"
COD_DATA="data/cod_82k"
BASE_OUT="output/cod_domain_shift_82k"

echo "=== COD 82K eval-only ==="
nvidia-smi
echo "Start: $(date)"

if [ ! -d "$MODEL" ]; then
    echo "ERROR: finetuned model not found at $MODEL"
    exit 1
fi

if [ ! -f "$BASE_OUT/cod_on_cod.json" ]; then
    echo "=== COD-finetuned -> COD 82K ==="
    python -m clarimol evaluate \
        --model-path "$MODEL" \
        --data-dir "$COD_DATA" \
        --output-file "$BASE_OUT/cod_on_cod.json" \
        --no-unsloth --batch-size 8
else
    echo "SKIP: cod_on_cod.json exists"
fi

if [ ! -f "$BASE_OUT/cod_on_zinc.json" ]; then
    echo "=== COD-finetuned -> ZINC ==="
    python -m clarimol evaluate \
        --model-path "$MODEL" \
        --data-dir data/test \
        --output-file "$BASE_OUT/cod_on_zinc.json" \
        --no-unsloth --batch-size 8
else
    echo "SKIP: cod_on_zinc.json exists"
fi

echo ""
echo "=== Results ==="
for F in cod_on_cod cod_on_zinc; do
    echo -n "  $F: "
    python3 -c "
import json, pathlib
p = pathlib.Path('$BASE_OUT/${F}.json')
if p.exists():
    d = json.load(open(p))
    accs = [d[t]['accuracy'] for t in sorted(d) if 'accuracy' in d[t]]
    print(f'mean={sum(accs)/len(accs):.4f}')
else:
    print('MISSING')
"
done

echo "=== Complete at $(date) ==="
