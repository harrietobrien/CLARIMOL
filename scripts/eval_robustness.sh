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
# (via RDKit randomization), run all 5 parsing tasks on both canonical and
# randomized inputs, and measure accuracy drop + consistency.
#
# Tests whether parsing pre-training makes models robust to SMILES
# representation choice.
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

# Clear stale results from previous failed run
rm -f "$OUT_DIR/robustness_results.json" "$OUT_DIR/robustness_base_results.json"
rm -f "$OUT_DIR/randomized_data.flag"
rm -f "$OUT_DIR"/canonical_*.json "$OUT_DIR"/randomized_*.json

# Step 1: Generate randomized SMILES test sets using actual test data format
echo "=== Generating randomized SMILES variants ==="
python3 << 'PYEOF'
import json
import os
import random
from rdkit import Chem

random.seed(42)
out_dir = "output/robustness"
test_dir = "data/test"
n_random = 5
max_molecules = 2000

tasks = [
    "functional_group", "ring_counting", "chain_length",
    "canonicalization", "fragment_assembly",
]

def randomize_smiles(mol, n=5):
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

    if len(samples) > max_molecules:
        samples = random.sample(samples, max_molecules)

    canonical_samples = []
    randomized_samples = []
    n_skipped = 0

    for sample in samples:
        smiles = sample.get("smiles", "")
        question = sample.get("question", "")
        answer = sample.get("answer", "")
        task_name = sample.get("task", task)
        metadata = sample.get("metadata", {})

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

        mol_id = len(canonical_samples)

        canonical_samples.append({
            "smiles": canonical,
            "question": question,
            "answer": answer,
            "task": task_name,
            "metadata": metadata,
            "molecule_id": mol_id,
        })

        for i, rsmi in enumerate(randoms):
            randomized_samples.append({
                "smiles": rsmi,
                "question": question,
                "answer": answer,
                "task": task_name,
                "metadata": metadata,
                "molecule_id": mol_id,
                "variant_id": i,
            })

    canon_path = os.path.join(out_dir, f"canonical_{task}.json")
    with open(canon_path, "w") as f:
        json.dump(canonical_samples, f)

    rand_path = os.path.join(out_dir, f"randomized_{task}.json")
    with open(rand_path, "w") as f:
        json.dump(randomized_samples, f)

    print(f"  {task}: {len(canonical_samples)} canonical, "
          f"{len(randomized_samples)} randomized ({n_skipped} skipped)")

print("Randomized data generation complete.")
PYEOF

echo "=== Data generation done ==="

# Step 2: Evaluate ZINC-trained model on canonical + randomized inputs
echo "=== Evaluating ZINC-trained model ==="
python3 << 'PYEOF'
import json
import os
import sys
import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

sys.path.insert(0, ".")
from src.clarimol.tasks.prompts import build_messages

out_dir = "output/robustness"
zinc_model = "output/multi_seed/llama-8b/seed_42/final"
base_model = "meta-llama/Llama-3.1-8B-Instruct"

tasks = [
    "functional_group", "ring_counting", "chain_length",
    "canonicalization", "fragment_assembly",
]

print("Loading ZINC-trained model...")
tokenizer = AutoTokenizer.from_pretrained(base_model)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    base_model, torch_dtype=torch.bfloat16, device_map="auto",
)
model = PeftModel.from_pretrained(model, zinc_model)
model.eval()

rng = random.Random(42)

def evaluate_samples(samples, tokenizer, model):
    results = []
    for i, sample in enumerate(samples):
        messages = build_messages(sample, rng=rng, use_system_prompt=True)
        messages_no_answer = [m for m in messages if m["role"] != "assistant"]

        try:
            prompt = tokenizer.apply_chat_template(
                messages_no_answer, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            prompt = "\n".join(
                f"<|{m['role']}|>\n{m['content']}" for m in messages_no_answer
            ) + "\n<|assistant|>\n"

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=128, do_sample=False,
                temperature=None, top_p=None,
            )

        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()

        expected = sample["answer"].strip()
        correct = response.lower() == expected.lower() or expected.lower() in response.lower()

        results.append({
            "correct": correct,
            "molecule_id": sample.get("molecule_id"),
            "variant_id": sample.get("variant_id"),
        })

        if (i + 1) % 500 == 0:
            print(f"    processed {i+1}/{len(samples)}")

    return results


# Need Sample-like objects for build_messages
class SampleLike:
    def __init__(self, d):
        self.smiles = d["smiles"]
        self.task = d["task"]
        self.question = d["question"]
        self.answer = d["answer"]
        self.metadata = d.get("metadata", {})
        self.difficulty = d.get("difficulty", 0)

all_results = {}
for task in tasks:
    canon_path = os.path.join(out_dir, f"canonical_{task}.json")
    rand_path = os.path.join(out_dir, f"randomized_{task}.json")

    if not os.path.exists(canon_path) or not os.path.exists(rand_path):
        print(f"  {task}: data files missing, skipping")
        continue

    with open(canon_path) as f:
        canonical_raw = json.load(f)
    with open(rand_path) as f:
        randomized_raw = json.load(f)

    if not canonical_raw:
        print(f"  {task}: no canonical samples, skipping")
        continue

    canonical = [SampleLike(s) for s in canonical_raw]
    randomized = [SampleLike(s) for s in randomized_raw]

    print(f"  {task}: evaluating {len(canonical)} canonical + {len(randomized)} randomized...")

    canon_results = evaluate_samples(canonical, tokenizer, model)
    canon_acc = sum(r["correct"] for r in canon_results) / len(canon_results)

    rand_results = evaluate_samples(randomized, tokenizer, model)
    rand_acc = sum(r["correct"] for r in rand_results) / len(rand_results)

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
        "canonical_accuracy": round(canon_acc, 4),
        "randomized_accuracy": round(rand_acc, 4),
        "accuracy_drop": round(canon_acc - rand_acc, 4),
        "consistency_rate": round(consistency_rate, 4),
        "any_correct_rate": round(any_correct_rate, 4),
        "n_canonical": len(canonical),
        "n_randomized": len(randomized),
        "n_molecules": len(mol_ids),
    }

    print(f"    canonical:   {canon_acc:.4f}")
    print(f"    randomized:  {rand_acc:.4f}")
    print(f"    drop:        {canon_acc - rand_acc:+.4f}")
    print(f"    consistency: {consistency_rate:.4f}")

with open(os.path.join(out_dir, "robustness_results.json"), "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nZINC-trained results saved.")
PYEOF

# Step 3: Evaluate base model for comparison
echo "=== Evaluating base LLaMA (no LoRA) ==="
python3 << 'PYEOF'
import json
import os
import sys
import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, ".")
from src.clarimol.tasks.prompts import build_messages

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

rng = random.Random(42)

class SampleLike:
    def __init__(self, d):
        self.smiles = d["smiles"]
        self.task = d["task"]
        self.question = d["question"]
        self.answer = d["answer"]
        self.metadata = d.get("metadata", {})
        self.difficulty = d.get("difficulty", 0)

all_results = {}
for task in tasks:
    canon_path = os.path.join(out_dir, f"canonical_{task}.json")
    rand_path = os.path.join(out_dir, f"randomized_{task}.json")

    if not os.path.exists(canon_path) or not os.path.exists(rand_path):
        continue

    with open(canon_path) as f:
        canonical = [SampleLike(s) for s in json.load(f)]
    with open(rand_path) as f:
        randomized = [SampleLike(s) for s in json.load(f)]

    if not canonical:
        continue

    print(f"  {task}: {len(canonical)} canonical + {len(randomized)} randomized...")

    def eval_samples(samples):
        correct = 0
        for i, sample in enumerate(samples):
            messages = build_messages(sample, rng=rng, use_system_prompt=True)
            messages_no_answer = [m for m in messages if m["role"] != "assistant"]
            try:
                prompt = tokenizer.apply_chat_template(
                    messages_no_answer, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                prompt = "\n".join(
                    f"<|{m['role']}|>\n{m['content']}" for m in messages_no_answer
                ) + "\n<|assistant|>\n"
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens=128, do_sample=False,
                    temperature=None, top_p=None,
                )
            response = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            ).strip()
            expected = sample.answer.strip()
            if response.lower() == expected.lower() or expected.lower() in response.lower():
                correct += 1
            if (i + 1) % 500 == 0:
                print(f"    processed {i+1}/{len(samples)}")
        return correct / len(samples) if samples else 0.0

    canon_acc = eval_samples(canonical)
    rand_acc = eval_samples(randomized)

    all_results[task] = {
        "canonical_accuracy": round(canon_acc, 4),
        "randomized_accuracy": round(rand_acc, 4),
        "accuracy_drop": round(canon_acc - rand_acc, 4),
    }
    print(f"    canonical: {canon_acc:.4f}, randomized: {rand_acc:.4f}, drop: {canon_acc - rand_acc:+.4f}")

with open(os.path.join(out_dir, "robustness_base_results.json"), "w") as f:
    json.dump(all_results, f, indent=2)
print("Base results saved.")
PYEOF

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
