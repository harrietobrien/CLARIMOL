#!/usr/bin/env bash
: <<'COMMENT'
--no-unsloth: Unsloth Triton kernels don't compile on P100 (sm_60)
     SDPA patch also crashes; use standard HF+PEFT instead
--max-length 256: Samples average ~100 tokens with chat template
     p95 < 150 tokens. 256 gives headroom while keeping VRAM in budget
     (512 caused OOM with batch=8 due to 128K vocab cross-entropy tensor)
--packing: Packs ~2-3 short sequences per 256-token slot. Saves padding
--select-sample 10000: 10K samples/task (50K total). Sufficient for LoRA SFT
     Set to 2000 for fastest run (10K total), or remove for full 50K/task
--batch-size 8: Fits P100 16GB with 256 max_length + grad checkpointing
--grad-accum 2: Effective batch = 16
COMMENT

set -euo pipefail
cd "$(dirname "$0")/.."

OUTPUT_DIR="output/parsing_pretrain"
LOG_FILE="output/train_p100_fast.log"
mkdir -p output

# Optional: start GPU monitor in background
if [[ -f scripts/gpu_monitor.py ]]; then
    python3 scripts/gpu_monitor.py --interval 30 --log-file output/gpu_temps.log &
    MONITOR_PID=$!
    trap "kill $MONITOR_PID 2>/dev/null" EXIT
    echo "GPU monitor started (PID $MONITOR_PID) — logging to output/gpu_temps.log"
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Starting optimized P100 training at $(date)"
echo "Logs: $LOG_FILE"

python -m clarimol train \
    --model unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit \
    --data-dir data/clarimol \
    --output-dir "$OUTPUT_DIR" \
    --no-unsloth \
    --packing \
    --max-length 256 \
    --select-sample 2000 \
    --batch-size 4 \
    --grad-accum 4 \
    --lr 5e-4 \
    --epochs 1 \
    --lora-r 64 \
    --lora-alpha 16 \
    --no-wandb \
    --seed 42 \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "Training complete at $(date) :)"
echo "Model saved to: $OUTPUT_DIR/final"
echo ""
echo "To evaluate:"
echo "  bash scripts/evaluate_all.sh $OUTPUT_DIR/final"
