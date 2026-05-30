#!/bin/bash
#SBATCH --job-name=cm_robustness
#SBATCH -A scavenger-h200
#SBATCH -p scavenger-h200
#SBATCH --gres=gpu:h200:1
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --exclude=dcc-h200-gpu-05
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=output/logs/robustness/robustness_%j.log
#SBATCH --error=output/logs/robustness/robustness_%j.err
#
# SMILES robustness evaluation:
# For each molecule in the test set, generate 5 random non-canonical SMILES
# (via RDKit randomization), run all 5 parsing tasks on randomized inputs,
# and measure accuracy drop vs canonical SMILES.
#
# Tests whether parsing pre-training makes models robust to SMILES
# representation choice -- a key practical concern since the same molecule
# can be written as many different valid SMILES strings.
#
# Submit: sbatch scripts/eval_robustness.sh

set -euo pipefail
cd ~/storage/CLARIMOL
mkdir -p output/logs/robustness

export CONDARC=/work/gc237/.condarc
export HF_HOME=/work/gc237/.cache/huggingface
export HF_TOKEN=$(cat /work/gc237/.cache/huggingface/token 2>/dev/null || cat ~/.cache/huggingface/token)

source /opt/apps/rhel9/Anaconda3-2024.02/etc/profile.d/conda.sh
conda activate /work/gc237/conda_envs/clarimol

echo "=== SMILES Robustness Evaluation ==="
nvidia-smi
echo "Start: $(date)"

ZINC_MODEL="output/multi_seed/llama-8b/seed_42/final"
BASE_MODEL="meta-llama/Llama-3.1-8B-Instruct"
OUT_DIR="output/robustness"
TEST_DIR="data/test"
N_RANDOM=5
MAX_MOLECULES=2000

mkdir -p "$OUT_DIR"

if [ ! -d "$ZINC_MODEL" ]; then
    echo "ERROR: ZINC-trained model not found at $ZINC_MODEL"
    exit 1
fi

# Step 1: Generate randomized SMILES test sets
if [ ! -f "$OUT_DIR/randomized_data.flag" ]; then
    echo "=== Generating randomized SMILES variants ==="
    python3 << PYEOF
import json
import os
import random
from rdkit import Chem

random.seed(42)
out_dir = "output/robustness"
test_dir = "data/test"
n_random = $N_RANDOM
max_molecules = $MAX_MOLECULES

tasks = [
    "functional_group", "ring_counting", "chain_length",
    "canonicalization", "fragment_assembly",
]

def randomize_smiles(mol, n=5):
    """Generate n random SMILES strings for a molecule."""
    results = []
    for _ in range(n * 10):
        if len(results) >= n:
            break
        try:
            smi = Chem.MolToSmiles(mol, doRandom=True, canonical=False)
            if smi and smi not in results:
                results.append(smi)
        except Exception:
            continue
    return results

for task in tasks:
    task_file = os.path.join(test_dir, f"{task}.json")
    if not os.path.exists(task_file):
        print(f"  {task}: test file not found, skipping")
        continue

    with open(task_file) as f:
        samples = json.load(f)

    # Subsample for tractability
    if len(samples) > max_molecules:
        samples = random.sample(samples, max_molecules)

    canonical_samples = []
    randomized_samples = []
    n_skipped = 0

    for sample in samples:
        # Extract SMILES from the instruction text
        instruction = sample.get("instruction", sample.get("input", ""))
        output = sample.get("output", sample.get("response", ""))

        # Find SMILES in instruction (typically after "SMILES:" or similar)
        smiles = None
        for line in instruction.split("\n"):
            line = line.strip()
            if "SMILES" in line.upper() and ":" in line:
                smiles = line.split(":", 1)[1].strip()
                break

        if not smiles:
            # Try extracting directly if instruction contains SMILES
            tokens = instruction.split()
            for tok in tokens:
                mol_check = Chem.MolFromSmiles(tok)
                if mol_check and len(tok) > 5:
                    smiles = tok
                    break

        if not smiles:
            n_skipped += 1
            continue

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            n_skipped += 1
            continue

        canonical = Chem.MolToSmiles(mol, canonical=True)
        randoms = randomize_smiles(mol, n=n_random)

        if len(randoms) < 1:
            n_skipped += 1
            continue

        # Keep canonical version
        canonical_samples.append({
            "instruction": instruction,
            "output": output,
            "smiles": canonical,
            "molecule_id": len(canonical_samples),
        })

        # Create randomized versions
        for i, rsmi in enumerate(randoms):
            randomized_instruction = instruction.replace(smiles, rsmi)
            if canonical in randomized_instruction:
                randomized_instruction = randomized_instruction.replace(canonical, rsmi)
            randomized_samples.append({
                "instruction": randomized_instruction,
                "output": output,
                "smiles": rsmi,
                "canonical_smiles": canonical,
                "molecule_id": len(canonical_samples) - 1,
                "variant_id": i,
            })

    # Save canonical test set
    canon_path = os.path.join(out_dir, f"canonical_{task}.json")
    with open(canon_path, "w") as f:
        json.dump(canonical_samples, f, indent=2)

    # Save randomized test set
    rand_path = os.path.join(out_dir, f"randomized_{task}.json")
    with open(rand_path, "w") as f:
        json.dump(randomized_samples, f, indent=2)

    print(f"  {task}: {len(canonical_samples)} canonical, "
          f"{len(randomized_samples)} randomized ({n_skipped} skipped)")

print("Randomized data generation complete.")
PYEOF

    if [ $? -eq 0 ]; then
        touch "$OUT_DIR/randomized_data.flag"
    else
        echo "ERROR: randomized data generation failed"
        exit 1
    fi
else
    echo "SKIP: randomized data already generated"
fi

# Step 2: Evaluate ZINC-trained model on canonical + randomized inputs
if [ ! -f "$OUT_DIR/robustness_results.json" ]; then
    echo "=== Evaluating robustness ==="
    python3 << 'PYEOF'
import json
import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

out_dir = "output/robustness"
zinc_model = "output/multi_seed/llama-8b/seed_42/final"
base_model = "meta-llama/Llama-3.1-8B-Instruct"

tasks = [
    "functional_group", "ring_counting", "chain_length",
    "canonicalization", "fragment_assembly",
]

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(base_model)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    base_model, torch_dtype=torch.bfloat16, device_map="auto",
)
model = PeftModel.from_pretrained(model, zinc_model)
model.eval()

BATCH_SIZE = 16

def evaluate_samples(samples, tokenizer, model):
    """Evaluate a list of instruction/output samples and return per-sample correctness."""
    results = []
    for i in range(0, len(samples), BATCH_SIZE):
        batch = samples[i:i+BATCH_SIZE]
        for sample in batch:
            instruction = sample["instruction"]
            expected = sample["output"].strip().lower()

            messages = [{"role": "user", "content": instruction}]
            chat_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(
                chat_text, return_tensors="pt", truncation=True, max_length=512
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens=128, do_sample=False,
                    temperature=None, top_p=None,
                )

            response = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            ).strip().lower()

            correct = response == expected or expected in response
            results.append({
                "correct": correct,
                "molecule_id": sample.get("molecule_id"),
                "variant_id": sample.get("variant_id"),
            })

        if (i + BATCH_SIZE) % 200 == 0:
            print(f"    processed {min(i + BATCH_SIZE, len(samples))}/{len(samples)}")

    return results


all_results = {}
for task in tasks:
    canon_path = os.path.join(out_dir, f"canonical_{task}.json")
    rand_path = os.path.join(out_dir, f"randomized_{task}.json")

    if not os.path.exists(canon_path) or not os.path.exists(rand_path):
        print(f"  {task}: data files missing, skipping")
        continue

    with open(canon_path) as f:
        canonical = json.load(f)
    with open(rand_path) as f:
        randomized = json.load(f)

    print(f"  {task}: evaluating {len(canonical)} canonical + {len(randomized)} randomized...")

    # Evaluate canonical
    canon_results = evaluate_samples(canonical, tokenizer, model)
    canon_acc = sum(r["correct"] for r in canon_results) / len(canon_results) if canon_results else 0.0

    # Evaluate randomized
    rand_results = evaluate_samples(randomized, tokenizer, model)
    rand_acc = sum(r["correct"] for r in rand_results) / len(rand_results) if rand_results else 0.0

    # Per-molecule consistency: fraction of molecules where ALL variants are correct
    mol_ids = set(r["molecule_id"] for r in rand_results if r["molecule_id"] is not None)
    consistent = 0
    any_correct = 0
    for mid in mol_ids:
        mol_results = [r["correct"] for r in rand_results if r["molecule_id"] == mid]
        if all(mol_results):
            consistent += 1
        if any(mol_results):
            any_correct += 1
    consistency_rate = consistent / len(mol_ids) if mol_ids else 0.0
    any_correct_rate = any_correct / len(mol_ids) if mol_ids else 0.0

    all_results[task] = {
        "canonical_accuracy": canon_acc,
        "randomized_accuracy": rand_acc,
        "accuracy_drop": canon_acc - rand_acc,
        "consistency_rate": consistency_rate,
        "any_correct_rate": any_correct_rate,
        "n_canonical": len(canonical),
        "n_randomized": len(randomized),
        "n_molecules": len(mol_ids),
    }

    print(f"    canonical:   {canon_acc:.4f}")
    print(f"    randomized:  {rand_acc:.4f}")
    print(f"    drop:        {canon_acc - rand_acc:+.4f}")
    print(f"    consistency: {consistency_rate:.4f} (all variants correct)")
    print(f"    any_correct: {any_correct_rate:.4f}")

# Save full results
output_path = os.path.join(out_dir, "robustness_results.json")
with open(output_path, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nResults saved to {output_path}")
PYEOF
else
    echo "SKIP: robustness results already exist"
fi

# Step 3: Also evaluate base model for comparison
if [ ! -f "$OUT_DIR/robustness_base_results.json" ]; then
    echo "=== Evaluating base LLaMA robustness (no LoRA) ==="
    python3 << 'PYEOF'
import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

out_dir = "output/robustness"
base_model = "meta-llama/Llama-3.1-8B-Instruct"

tasks = [
    "functional_group", "ring_counting", "chain_length",
    "canonicalization", "fragment_assembly",
]

print("Loading base model (no LoRA)...")
tokenizer = AutoTokenizer.from_pretrained(base_model)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    base_model, torch_dtype=torch.bfloat16, device_map="auto",
)
model.eval()

all_results = {}
for task in tasks:
    canon_path = os.path.join(out_dir, f"canonical_{task}.json")
    rand_path = os.path.join(out_dir, f"randomized_{task}.json")

    if not os.path.exists(canon_path) or not os.path.exists(rand_path):
        continue

    with open(canon_path) as f:
        canonical = json.load(f)
    with open(rand_path) as f:
        randomized = json.load(f)

    print(f"  {task}: {len(canonical)} canonical + {len(randomized)} randomized...")

    def eval_batch(samples):
        correct = 0
        for sample in samples:
            messages = [{"role": "user", "content": sample["instruction"]}]
            chat_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(chat_text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens=128, do_sample=False,
                    temperature=None, top_p=None,
                )
            response = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            ).strip().lower()
            expected = sample["output"].strip().lower()
            if response == expected or expected in response:
                correct += 1
        return correct / len(samples) if samples else 0.0

    canon_acc = eval_batch(canonical)
    rand_acc = eval_batch(randomized)

    all_results[task] = {
        "canonical_accuracy": canon_acc,
        "randomized_accuracy": rand_acc,
        "accuracy_drop": canon_acc - rand_acc,
    }
    print(f"    canonical: {canon_acc:.4f}, randomized: {rand_acc:.4f}, drop: {canon_acc - rand_acc:+.4f}")

output_path = os.path.join(out_dir, "robustness_base_results.json")
with open(output_path, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"Base results saved to {output_path}")
PYEOF
else
    echo "SKIP: base robustness results already exist"
fi

# Step 4: Print comparison
echo ""
echo "=== SMILES Robustness Comparison ==="
python3 << 'PYEOF'
import json, os

out_dir = "output/robustness"
zinc_path = os.path.join(out_dir, "robustness_results.json")
base_path = os.path.join(out_dir, "robustness_base_results.json")

if not os.path.exists(zinc_path):
    print("No ZINC-trained results found.")
    exit(0)

with open(zinc_path) as f:
    zinc = json.load(f)

base = {}
if os.path.exists(base_path):
    with open(base_path) as f:
        base = json.load(f)

print(f"{'Task':25s}  {'Canon':>7s}  {'Random':>7s}  {'Drop':>7s}  {'Consist':>8s}  {'Base Drop':>10s}")
print("-" * 75)

for task in sorted(zinc.keys()):
    z = zinc[task]
    b = base.get(task, {})
    base_drop = b.get("accuracy_drop", float("nan"))
    print(f"{task:25s}  "
          f"{z['canonical_accuracy']:7.4f}  "
          f"{z['randomized_accuracy']:7.4f}  "
          f"{z['accuracy_drop']:+7.4f}  "
          f"{z.get('consistency_rate', 0):8.4f}  "
          f"{base_drop:+10.4f}")

print("\nConsistency = fraction of molecules where ALL 5 random SMILES variants are correct")
print("Lower drop + higher consistency = more robust to SMILES representation choice")
PYEOF

echo "=== Complete at $(date) ==="
