#!/bin/bash
#SBATCH --job-name=cm_moleculariq
#SBATCH -A scavenger-h200
#SBATCH -p scavenger-h200
#SBATCH --gres=gpu:h200:1
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=output/logs/benchmarks/moleculariq_%j.log
#SBATCH --error=output/logs/benchmarks/moleculariq_%j.err
#
# MolecularIQ benchmark evaluation (Bartmann et al., ICLR 2026).
# 5,111 questions across 849 molecules, 30 molecular features.
# Tests ring counting, longest chain, functional groups -- direct
# overlap with CLARIMOL SMILES parsing tasks.
#
# Evaluates base vs CLARIMOL-trained LLaMA-8B and Mistral-7B using
# the lm-eval-harness integration from moleculariq-eval.
#
# Submit: sbatch scripts/eval_benchmark_moleculariq.sh

set -euo pipefail
cd ~/storage/CLARIMOL
mkdir -p output/logs/benchmarks
mkdir -p output/benchmarks/moleculariq

export CONDARC=/work/gc237/.condarc
export HF_HOME=/work/gc237/.cache/huggingface
export HF_TOKEN=$(cat /work/gc237/.cache/huggingface/token 2>/dev/null || cat ~/.cache/huggingface/token 2>/dev/null || true)

source /opt/apps/rhel9/Anaconda3-2024.02/etc/profile.d/conda.sh
conda activate /work/gc237/conda_envs/clarimol

echo "MolecularIQ Benchmark Evaluation"
nvidia-smi
echo "Start: $(date)"

OUT_DIR="output/benchmarks/moleculariq"

# Model definitions
# Each entry: "label|base_model_id|adapter_path_or_NONE"
MODELS=(
    "llama-8b-base|meta-llama/Llama-3.1-8B-Instruct|NONE"
    "llama-8b-clarimol|meta-llama/Llama-3.1-8B-Instruct|output/multi_seed/llama-8b/seed_42/final"
    "mistral-7b-base|mistralai/Mistral-7B-Instruct-v0.3|NONE"
    "mistral-7b-clarimol|mistralai/Mistral-7B-Instruct-v0.3|output/multi_seed/mistral-7b/seed_42/final"
)

# Step 1: Install moleculariq-eval if not present
if ! python3 -c "import lm_eval" 2>/dev/null; then
    echo "Installing lm-eval..."
    pip install lm-eval
fi

MOLECULARIQ_DIR="$HOME/storage/repos/moleculariq-eval"
if [ ! -d "$MOLECULARIQ_DIR" ]; then
    echo "Cloning moleculariq-eval..."
    mkdir -p "$HOME/storage/repos"
    git clone https://github.com/ml-jku/moleculariq-eval.git "$MOLECULARIQ_DIR"
    pip install -e "$MOLECULARIQ_DIR"
    pip install moleculariq-core rdkit 2>/dev/null || pip install moleculariq-core 2>/dev/null || true
else
    echo "moleculariq-eval already cloned at $MOLECULARIQ_DIR"
fi

# Step 2: Try lm-eval-harness integration first.
#
# The moleculariq-eval repo registers task configs (moleculariq_pass_at_k,
# moleculariq_inline) with lm-eval. The --model hf backend supports PEFT
# via the peft= arg in model_args. If this fails, fall back to standalone
# evaluation in Step 3.
#
# NOTE: lm-eval supports --model hf for transformers-based inference.
# The peft= argument in model_args loads a LoRA adapter via PEFT.
# Syntax: --model_args pretrained=BASE,peft=ADAPTER_PATH
# Ref: https://github.com/EleutherAI/lm-evaluation-harness

LM_EVAL_WORKS=true

for entry in "${MODELS[@]}"; do
    IFS='|' read -r label base_model adapter_path <<< "$entry"
    results_file="$OUT_DIR/${label}_results.json"

    if [ -f "$results_file" ]; then
        echo "SKIP: $label results already exist at $results_file"
        continue
    fi

    echo "Evaluating $label ($base_model, adapter=$adapter_path)..."

    if [ "$adapter_path" = "NONE" ]; then
        MODEL_ARGS="pretrained=${base_model},dtype=bfloat16,trust_remote_code=True"
    else
        MODEL_ARGS="pretrained=${base_model},peft=${adapter_path},dtype=bfloat16,trust_remote_code=True"
    fi

    # Attempt lm-eval with --model hf (transformers backend, no vLLM needed).
    # moleculariq_pass_at_k is the standard task with system prompt support.
    # If the task is not found, the install of moleculariq-eval may need the
    # editable install to register the task yaml.
    if $LM_EVAL_WORKS; then
        lm_eval --model hf \
            --model_args "$MODEL_ARGS" \
            --tasks moleculariq_pass_at_k \
            --apply_chat_template \
            --batch_size 4 \
            --log_samples \
            --output_path "$OUT_DIR/$label" \
            2>&1 | tee "$OUT_DIR/${label}_lm_eval.log" && \
        {
            echo "lm-eval succeeded for $label"
            # Copy the results JSON to the standard location
            find "$OUT_DIR/$label" -name "results.json" -exec cp {} "$results_file" \;
            continue
        } || {
            echo "WARNING: lm-eval failed for $label, will try standalone"
            LM_EVAL_WORKS=false
        }
    fi
done

# Step 3: Standalone fallback if lm-eval integration does not work.
# This loads the MolecularIQ benchmark dataset directly and runs
# inference using the CLARIMOL model loading pattern.

if ! $LM_EVAL_WORKS; then
    echo ""
    echo "lm-eval integration failed. Running standalone MolecularIQ evaluation."
    echo ""

    python3 << 'PYEOF'
import json
import os
import sys
import re
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm

out_dir = "output/benchmarks/moleculariq"
os.makedirs(out_dir, exist_ok=True)

# Load MolecularIQ benchmark dataset.
# The benchmark is hosted on HuggingFace by ml-jku.
# TODO: Verify the exact HuggingFace dataset ID. Candidates:
#   - ml-jku/moleculariq
#   - ml-jku/moleculariq-benchmark
# If neither works, load from the cloned repo's data/ directory.
dataset = None
dataset_source = None

try:
    from datasets import load_dataset
    for ds_name in ["ml-jku/moleculariq-benchmark", "ml-jku/moleculariq"]:
        try:
            dataset = load_dataset(ds_name, split="test")
            dataset_source = ds_name
            print(f"Loaded MolecularIQ from HuggingFace: {ds_name} ({len(dataset)} samples)")
            break
        except Exception:
            continue
except ImportError:
    print("datasets library not available", file=sys.stderr)

if dataset is None:
    # Try loading from cloned repo
    repo_dir = os.path.expanduser("~/storage/repos/moleculariq-eval")
    data_candidates = [
        os.path.join(repo_dir, "data"),
        os.path.join(repo_dir, "benchmark"),
    ]
    for data_dir in data_candidates:
        if os.path.isdir(data_dir):
            jsonl_files = list(Path(data_dir).rglob("*.jsonl")) + list(Path(data_dir).rglob("*.json"))
            if jsonl_files:
                print(f"Found data files in {data_dir}: {[f.name for f in jsonl_files[:5]]}")
                # TODO: Parse the specific format used by moleculariq-benchmark
                dataset_source = str(data_dir)
                break

    if dataset is None:
        print("ERROR: Could not load MolecularIQ benchmark dataset.", file=sys.stderr)
        print("Manual steps needed:", file=sys.stderr)
        print("  1. Check https://github.com/ml-jku/moleculariq-benchmark for data", file=sys.stderr)
        print("  2. Or find the HuggingFace dataset ID on the MolecularIQ leaderboard", file=sys.stderr)
        sys.exit(1)


def load_model(base_model_name, adapter_path=None):
    """Load base model, optionally with LoRA adapter."""
    print(f"  Loading tokenizer from {adapter_path or base_model_name}...")
    tok_source = adapter_path if adapter_path else base_model_name
    tokenizer = AutoTokenizer.from_pretrained(tok_source, padding_side="left", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"  Loading base model {base_model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        quantization_config=bnb_config,
        trust_remote_code=True,
    )

    if adapter_path:
        print(f"  Loading LoRA adapter from {adapter_path}...")
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model, tokenizer


@torch.inference_mode()
def run_moleculariq_eval(model, tokenizer, dataset, max_samples=None):
    """Run MolecularIQ evaluation and return metrics."""
    samples = list(dataset)
    if max_samples:
        samples = samples[:max_samples]

    correct = 0
    total = 0
    results_by_task = {}

    for i, sample in enumerate(tqdm(samples, desc="MolecularIQ")):
        # TODO: Verify the column names in the MolecularIQ dataset.
        # Expected columns based on the benchmark description:
        #   - "question" or "prompt": the question text
        #   - "answer" or "ground_truth": the expected answer
        #   - "task_type" or "category": counting/indexing/constrained_generation
        question = sample.get("question", sample.get("prompt", sample.get("input", "")))
        ground_truth = str(sample.get("answer", sample.get("ground_truth", sample.get("output", ""))))
        task_type = sample.get("task_type", sample.get("category", sample.get("type", "unknown")))

        if not question:
            continue

        messages = [{"role": "user", "content": question}]
        try:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            prompt = f"<|user|>\n{question}\n<|assistant|>\n"

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1536)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        ).strip()

        # Exact match (case-insensitive, stripped)
        is_correct = response.strip().lower() == ground_truth.strip().lower()

        # Also try numeric extraction for counting tasks
        if not is_correct and task_type in ("counting", "Counting"):
            pred_nums = re.findall(r'\b(\d+)\b', response)
            gt_nums = re.findall(r'\b(\d+)\b', ground_truth)
            if pred_nums and gt_nums:
                is_correct = pred_nums[-1] == gt_nums[-1]

        correct += int(is_correct)
        total += 1

        if task_type not in results_by_task:
            results_by_task[task_type] = {"correct": 0, "total": 0}
        results_by_task[task_type]["correct"] += int(is_correct)
        results_by_task[task_type]["total"] += 1

    overall_acc = correct / total if total > 0 else 0.0
    for t in results_by_task:
        results_by_task[t]["accuracy"] = (
            results_by_task[t]["correct"] / results_by_task[t]["total"]
            if results_by_task[t]["total"] > 0 else 0.0
        )

    return {
        "overall_accuracy": overall_acc,
        "correct": correct,
        "total": total,
        "by_task_type": results_by_task,
    }


# Run evaluation for each model configuration
MODELS = [
    ("llama-8b-base", "meta-llama/Llama-3.1-8B-Instruct", None),
    ("llama-8b-clarimol", "meta-llama/Llama-3.1-8B-Instruct", "output/multi_seed/llama-8b/seed_42/final"),
    ("mistral-7b-base", "mistralai/Mistral-7B-Instruct-v0.3", None),
    ("mistral-7b-clarimol", "mistralai/Mistral-7B-Instruct-v0.3", "output/multi_seed/mistral-7b/seed_42/final"),
]

all_results = {}

for label, base_model, adapter_path in MODELS:
    results_file = os.path.join(out_dir, f"{label}_results.json")
    if os.path.exists(results_file):
        print(f"SKIP: {label} results already exist")
        with open(results_file) as f:
            all_results[label] = json.load(f)
        continue

    print(f"\nEvaluating {label}...")
    model, tokenizer = load_model(base_model, adapter_path)
    result = run_moleculariq_eval(model, tokenizer, dataset)
    all_results[label] = result

    with open(results_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  {label}: accuracy={result['overall_accuracy']:.4f} ({result['correct']}/{result['total']})")

    # Free GPU memory before loading next model
    del model, tokenizer
    torch.cuda.empty_cache()
    import gc; gc.collect()

# Summary comparison
print("\n\nMolecularIQ Results Summary")
print(f"{'Model':30s}  {'Accuracy':>10s}  {'Correct':>8s}  {'Total':>6s}")
print("-" * 60)
for label, res in all_results.items():
    print(f"{label:30s}  {res['overall_accuracy']:10.4f}  {res['correct']:8d}  {res['total']:6d}")

# Save combined results
combined_path = os.path.join(out_dir, "results.json")
with open(combined_path, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nCombined results saved to {combined_path}")
PYEOF
fi

echo "MolecularIQ evaluation complete at $(date)"
