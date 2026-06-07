#!/bin/bash
#SBATCH --job-name=cm_tdc_admet
#SBATCH -A scavenger-h200
#SBATCH -p scavenger-h200
#SBATCH --gres=gpu:h200:1
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=output/logs/tdc_admet/tdc_admet_%j.log
#SBATCH --error=output/logs/tdc_admet/tdc_admet_%j.err
#
# TDC ADMET evaluation: test whether SMILES parsing pre-training
# improves property prediction on Therapeutics Data Commons benchmarks.
#
# Evaluates ZINC-trained LLaMA on classification and regression
# ADMET tasks formatted as instruction-following prompts.
#
# Submit: sbatch scripts/eval_tdc_admet.sh

set -euo pipefail
cd ~/storage/CLARIMOL
mkdir -p output/logs/tdc_admet

export CONDARC=/work/gc237/.condarc
export HF_HOME=/work/gc237/.cache/huggingface
export HF_TOKEN=$(cat /work/gc237/.cache/huggingface/token 2>/dev/null || cat ~/.cache/huggingface/token)

source /opt/apps/rhel9/Anaconda3-2024.02/etc/profile.d/conda.sh
conda activate /work/gc237/conda_envs/clarimol

echo "=== TDC ADMET Evaluation ==="
nvidia-smi
echo "Start: $(date)"

ZINC_MODEL="output/multi_seed/llama-8b/seed_42/final"
BASE_MODEL="meta-llama/Llama-3.1-8B-Instruct"
OUT_DIR="output/tdc_admet"
mkdir -p "$OUT_DIR"

if [ ! -d "$ZINC_MODEL" ]; then
    echo "ERROR: ZINC-trained model not found at $ZINC_MODEL"
    exit 1
fi

# Step 1: Download TDC datasets and format as instruction-following prompts
if [ ! -f "$OUT_DIR/tdc_prepared.flag" ]; then
    echo "=== Preparing TDC ADMET datasets ==="
    python3 << 'PYEOF'
import json
import os
import sys

out_dir = "output/tdc_admet"
os.makedirs(out_dir, exist_ok=True)

try:
    from tdc.benchmark_group import admet_group
except ImportError:
    print("ERROR: PyTDC not installed. Run: pip install PyTDC", file=sys.stderr)
    sys.exit(1)

group = admet_group(path=os.path.join(out_dir, "tdc_data"))
benchmark_names = group.dataset_names

# Classification tasks (binary prediction)
CLASSIFICATION_TASKS = [
    "caco2_wang", "hia_hou", "pgp_broccatelli", "bioavailability_ma",
    "bbb_martins", "cyp2d6_veith", "cyp3a4_veith", "cyp2c9_veith",
    "cyp2d6_substrate_carbonmangels", "cyp3a4_substrate_carbonmangels",
    "herg", "ames", "dili",
]

# Regression tasks (continuous value prediction)
REGRESSION_TASKS = [
    "lipophilicity_astrazeneca", "solubility_aqsoldb",
    "half_life_obach", "clearance_hepatocyte_az", "clearance_microsome_az",
    "ld50_zhu", "ppbr_az", "vdss_lombardo",
]

all_results = {}
for name in benchmark_names:
    try:
        benchmark = group.get(name)
        train_val, test = benchmark["train_val"], benchmark["test"]

        is_classification = name.lower() in [t.lower() for t in CLASSIFICATION_TASKS]
        task_type = "classification" if is_classification else "regression"

        # Format test samples as instruction-following prompts
        prompts = []
        for _, row in test.iterrows():
            smiles = row["Drug"]
            label = row["Y"]

            if is_classification:
                instruction = (
                    f"Given the molecule with SMILES notation: {smiles}\n"
                    f"Predict whether this molecule is positive or negative "
                    f"for the {name} ADMET property.\n"
                    f"Answer with exactly 'positive' or 'negative'."
                )
            else:
                instruction = (
                    f"Given the molecule with SMILES notation: {smiles}\n"
                    f"Predict the numeric value for the {name} ADMET property.\n"
                    f"Answer with a single number."
                )

            prompts.append({
                "instruction": instruction,
                "ground_truth": label,
                "smiles": smiles,
                "task_type": task_type,
            })

        output_path = os.path.join(out_dir, f"tdc_{name}.json")
        with open(output_path, "w") as f:
            json.dump(prompts, f, indent=2, default=str)

        all_results[name] = {
            "task_type": task_type,
            "n_test": len(prompts),
            "file": output_path,
        }
        print(f"  {name}: {len(prompts)} test samples ({task_type})")

    except Exception as e:
        print(f"  {name}: FAILED - {e}", file=sys.stderr)
        continue

meta_path = os.path.join(out_dir, "tdc_meta.json")
with open(meta_path, "w") as f:
    json.dump(all_results, f, indent=2)

print(f"Prepared {len(all_results)} TDC ADMET tasks")
PYEOF

    if [ $? -eq 0 ]; then
        touch "$OUT_DIR/tdc_prepared.flag"
    else
        echo "ERROR: TDC preparation failed"
        exit 1
    fi
else
    echo "SKIP: TDC datasets already prepared"
fi

# Step 2: Evaluate ZINC-trained model on TDC tasks
if [ ! -f "$OUT_DIR/zinc_model_tdc_results.json" ]; then
    echo "=== Evaluating ZINC-trained LLaMA on TDC ADMET ==="
    python3 << 'PYEOF'
import json
import os
import sys
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

out_dir = "output/tdc_admet"
zinc_model = "output/multi_seed/llama-8b/seed_42/final"
base_model = "meta-llama/Llama-3.1-8B-Instruct"

# Load metadata
with open(os.path.join(out_dir, "tdc_meta.json")) as f:
    meta = json.load(f)

print("Loading base model...")
tokenizer = AutoTokenizer.from_pretrained(base_model)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(model, zinc_model)
model.eval()

MAX_SAMPLES_PER_TASK = 500
all_results = {}

for task_name, task_info in meta.items():
    task_file = task_info["file"]
    task_type = task_info["task_type"]

    if not os.path.exists(task_file):
        print(f"  {task_name}: data file missing, skipping")
        continue

    with open(task_file) as f:
        samples = json.load(f)

    # Limit samples for tractability
    samples = samples[:MAX_SAMPLES_PER_TASK]
    print(f"  Evaluating {task_name} ({task_type}, {len(samples)} samples)...")

    correct = 0
    total = 0
    predictions = []
    errors = []
    abs_errors = []

    for sample in samples:
        prompt = sample["instruction"]
        ground_truth = sample["ground_truth"]

        # Format as chat
        messages = [{"role": "user", "content": prompt}]
        chat_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(chat_text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
                temperature=None,
                top_p=None,
            )

        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        ).strip().lower()

        if task_type == "classification":
            gt_label = "positive" if float(ground_truth) >= 0.5 else "negative"
            pred_label = "positive" if "positive" in response else "negative"
            is_correct = pred_label == gt_label
            correct += int(is_correct)
            predictions.append({
                "ground_truth": gt_label,
                "prediction": pred_label,
                "raw_response": response,
                "correct": is_correct,
            })
        else:
            # Extract numeric prediction
            numbers = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", response)
            if numbers:
                pred_val = float(numbers[0])
                gt_val = float(ground_truth)
                abs_err = abs(pred_val - gt_val)
                abs_errors.append(abs_err)
                predictions.append({
                    "ground_truth": gt_val,
                    "prediction": pred_val,
                    "raw_response": response,
                    "abs_error": abs_err,
                })
            else:
                errors.append(response)
                predictions.append({
                    "ground_truth": float(ground_truth),
                    "prediction": None,
                    "raw_response": response,
                    "parse_error": True,
                })

        total += 1

    if task_type == "classification":
        accuracy = correct / total if total > 0 else 0.0
        result = {
            "task_type": task_type,
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "predictions": predictions,
        }
        print(f"    accuracy: {accuracy:.4f} ({correct}/{total})")
    else:
        mae = sum(abs_errors) / len(abs_errors) if abs_errors else float("inf")
        parse_failures = len(errors)
        result = {
            "task_type": task_type,
            "mae": mae,
            "n_parsed": len(abs_errors),
            "n_parse_errors": parse_failures,
            "total": total,
            "predictions": predictions,
        }
        print(f"    MAE: {mae:.4f}, parsed: {len(abs_errors)}/{total}")

    all_results[task_name] = result

# Save results
output_path = os.path.join(out_dir, "zinc_model_tdc_results.json")
with open(output_path, "w") as f:
    json.dump(all_results, f, indent=2, default=str)
print(f"Results saved to {output_path}")

# Summary
print("\n=== TDC ADMET Summary (ZINC-trained LLaMA) ===")
for name, res in sorted(all_results.items()):
    if res["task_type"] == "classification":
        print(f"  {name:40s}  acc={res['accuracy']:.4f}")
    else:
        print(f"  {name:40s}  MAE={res['mae']:.4f}  (parsed {res['n_parsed']}/{res['total']})")
PYEOF
else
    echo "SKIP: TDC results already exist"
fi

# Step 3: Evaluate base model (no LoRA) for comparison
if [ ! -f "$OUT_DIR/base_model_tdc_results.json" ]; then
    echo "=== Evaluating base LLaMA (no LoRA) on TDC ADMET ==="
    python3 << 'PYEOF'
import json
import os
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

out_dir = "output/tdc_admet"
base_model = "meta-llama/Llama-3.1-8B-Instruct"

with open(os.path.join(out_dir, "tdc_meta.json")) as f:
    meta = json.load(f)

print("Loading base model (no LoRA)...")
tokenizer = AutoTokenizer.from_pretrained(base_model)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.eval()

MAX_SAMPLES_PER_TASK = 500
all_results = {}

for task_name, task_info in meta.items():
    task_file = task_info["file"]
    task_type = task_info["task_type"]

    if not os.path.exists(task_file):
        continue

    with open(task_file) as f:
        samples = json.load(f)[:MAX_SAMPLES_PER_TASK]

    print(f"  Evaluating {task_name} ({task_type}, {len(samples)} samples)...")
    correct = 0
    total = 0
    abs_errors = []
    errors = []

    for sample in samples:
        messages = [{"role": "user", "content": sample["instruction"]}]
        chat_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(chat_text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=32, do_sample=False,
                temperature=None, top_p=None,
            )

        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip().lower()

        if task_type == "classification":
            gt_label = "positive" if float(sample["ground_truth"]) >= 0.5 else "negative"
            pred_label = "positive" if "positive" in response else "negative"
            correct += int(pred_label == gt_label)
        else:
            numbers = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", response)
            if numbers:
                abs_errors.append(abs(float(numbers[0]) - float(sample["ground_truth"])))
            else:
                errors.append(response)
        total += 1

    if task_type == "classification":
        acc = correct / total if total > 0 else 0.0
        all_results[task_name] = {"task_type": task_type, "accuracy": acc, "correct": correct, "total": total}
        print(f"    accuracy: {acc:.4f}")
    else:
        mae = sum(abs_errors) / len(abs_errors) if abs_errors else float("inf")
        all_results[task_name] = {"task_type": task_type, "mae": mae, "n_parsed": len(abs_errors), "total": total}
        print(f"    MAE: {mae:.4f}")

output_path = os.path.join(out_dir, "base_model_tdc_results.json")
with open(output_path, "w") as f:
    json.dump(all_results, f, indent=2, default=str)
print(f"Base model results saved to {output_path}")
PYEOF
else
    echo "SKIP: base model TDC results already exist"
fi

# Step 4: Compare ZINC-trained vs base
echo ""
echo "=== Comparison: ZINC-trained vs Base LLaMA on TDC ADMET ==="
python3 << 'PYEOF'
import json, os

out_dir = "output/tdc_admet"
zinc_path = os.path.join(out_dir, "zinc_model_tdc_results.json")
base_path = os.path.join(out_dir, "base_model_tdc_results.json")

if not os.path.exists(zinc_path) or not os.path.exists(base_path):
    print("Missing result files, cannot compare.")
    exit(0)

with open(zinc_path) as f:
    zinc = json.load(f)
with open(base_path) as f:
    base = json.load(f)

print(f"{'Task':40s}  {'Base':>10s}  {'ZINC':>10s}  {'Delta':>10s}")
print("-" * 75)

for name in sorted(set(list(zinc.keys()) + list(base.keys()))):
    z = zinc.get(name, {})
    b = base.get(name, {})
    task_type = z.get("task_type", b.get("task_type", "?"))

    if task_type == "classification":
        z_val = z.get("accuracy", 0.0)
        b_val = b.get("accuracy", 0.0)
        delta = z_val - b_val
        print(f"{name:40s}  {b_val:10.4f}  {z_val:10.4f}  {delta:+10.4f}")
    else:
        z_val = z.get("mae", float("inf"))
        b_val = b.get("mae", float("inf"))
        delta = z_val - b_val
        marker = " (lower=better)" if delta < 0 else ""
        print(f"{name:40s}  {b_val:10.4f}  {z_val:10.4f}  {delta:+10.4f}{marker}")
PYEOF

echo "=== Complete at $(date) ==="
