#!/bin/bash
#SBATCH --job-name=cm_robustness
#SBATCH -A scavenger-h200
#SBATCH -p scavenger-h200
#SBATCH --gres=gpu:h200:1
#SBATCH --time=24:00:00
#SBATCH --requeue
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


def canonicalize_fragment_smiles(smiles_str):
    """Canonicalize a dot-separated fragment SMILES by processing each fragment independently."""
    parts = smiles_str.split(" . ")
    canon_parts = []
    for part in parts:
        mol = Chem.MolFromSmiles(part)
        if mol is not None:
            canon_parts.append(Chem.MolToSmiles(mol, canonical=True))
        else:
            canon_parts.append(part)
    return " . ".join(canon_parts)


def randomize_fragment_smiles(smiles_str, n=5):
    """Generate n random SMILES for a dot-separated fragment string."""
    parts = smiles_str.split(" . ")
    mols = []
    for part in parts:
        mol = Chem.MolFromSmiles(part)
        if mol is None:
            return []
        mols.append(mol)

    results = []
    for _ in range(n * 10):
        if len(results) >= n:
            break
        try:
            rand_parts = []
            for mol in mols:
                rand_parts.append(Chem.MolToSmiles(mol, doRandom=True, canonical=False))
            combined = " . ".join(rand_parts)
            if combined not in results:
                results.append(combined)
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
    is_fragment = (task == "fragment_assembly")

    for sample in samples:
        smiles = sample.get("smiles", "")
        question = sample.get("question", "")
        answer = sample.get("answer", "")
        task_name = sample.get("task", task)
        metadata = sample.get("metadata", {})

        if not smiles:
            n_skipped += 1
            continue

        if is_fragment:
            canonical = canonicalize_fragment_smiles(smiles)
            randoms = randomize_fragment_smiles(smiles, n=n_random)
        else:
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
import random
import sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.path.insert(0, ".")
from clarimol.data.sample import Sample
from clarimol.tasks.prompts import build_messages
from clarimol.eval.metrics import evaluate_parsing

TASKS = ["functional_group", "ring_counting", "chain_length", "canonicalization", "fragment_assembly"]
OUT_DIR = "output/robustness"
ADAPTER = "output/multi_seed/llama-8b/seed_42/final"
BATCH_SIZE = 16

print("Loading ZINC-trained model...")
adapter_config = json.load(open(f"{ADAPTER}/adapter_config.json"))
base_name = adapter_config["base_model_name_or_path"]
tokenizer = AutoTokenizer.from_pretrained(ADAPTER, padding_side="left", trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
base_model = AutoModelForCausalLM.from_pretrained(base_name, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
model = PeftModel.from_pretrained(base_model, ADAPTER)
model.eval()

def run_inference(model, tokenizer, samples, batch_size=16):
    rng = random.Random(0)
    predictions = []
    for i in range(0, len(samples), batch_size):
        batch = samples[i:i+batch_size]
        prompts = []
        for s in batch:
            msgs = build_messages(s, rng=rng, use_system_prompt=True)
            msgs_no_answer = [m for m in msgs if m["role"] != "assistant"]
            try:
                prompt = tokenizer.apply_chat_template(msgs_no_answer, tokenize=False, add_generation_prompt=True)
            except Exception:
                prompt = "\n".join(f"<|{m['role']}|>\n{m['content']}" for m in msgs_no_answer) + "\n<|assistant|>\n"
            prompts.append(prompt)
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1920).to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=128, do_sample=False, pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        for j, output in enumerate(outputs):
            input_len = inputs["input_ids"][j].shape[0]
            text = tokenizer.decode(output[input_len:], skip_special_tokens=True).strip()
            predictions.append(text)
    return predictions

all_results = {}
for task in TASKS:
    canon_path = f"{OUT_DIR}/canonical_{task}.json"
    rand_path = f"{OUT_DIR}/randomized_{task}.json"
    if not Path(canon_path).exists():
        print(f"Skipping {task}: no data")
        continue

    with open(canon_path) as f:
        canon_data = json.load(f)
    with open(rand_path) as f:
        rand_data = json.load(f)

    canon_samples = [Sample(smiles=d["smiles"], task=task, question=d["question"], answer=d["answer"], metadata=d.get("metadata", {})) for d in canon_data]
    rand_samples = [Sample(smiles=d["smiles"], task=task, question=d["question"], answer=d["answer"], metadata=d.get("metadata", {})) for d in rand_data]

    print(f"  {task}: {len(canon_samples)} canonical, {len(rand_samples)} randomized")

    canon_preds = run_inference(model, tokenizer, canon_samples)
    canon_result = evaluate_parsing(canon_preds, [s.answer for s in canon_samples], task, [s.metadata for s in canon_samples])

    rand_preds = run_inference(model, tokenizer, rand_samples)
    rand_result = evaluate_parsing(rand_preds, [s.answer for s in rand_samples], task, [s.metadata for s in rand_samples])

    # Consistency: for each molecule, check if all N_RANDOM variants are correct
    n_molecules = len(canon_data)
    n_random = 5
    consistent = 0
    for mol_id in range(n_molecules):
        mol_preds = rand_preds[mol_id * n_random:(mol_id + 1) * n_random]
        mol_refs = [rand_samples[mol_id * n_random + k].answer for k in range(n_random)]
        mol_result = evaluate_parsing(mol_preds, mol_refs, task)
        if mol_result.correct == n_random:
            consistent += 1

    all_results[task] = {
        "canonical_accuracy": round(canon_result.accuracy, 4),
        "randomized_accuracy": round(rand_result.accuracy, 4),
        "drop_pp": round((rand_result.accuracy - canon_result.accuracy) * 100, 1),
        "consistency": round(consistent / n_molecules, 4) if n_molecules > 0 else 0.0,
        "n_molecules": n_molecules,
    }
    print(f"    canon={canon_result.accuracy:.4f}, rand={rand_result.accuracy:.4f}, drop={all_results[task]['drop_pp']:.1f}pp, consistency={all_results[task]['consistency']:.1%}")

with open(f"{OUT_DIR}/robustness_results.json", "w") as f:
    json.dump(all_results, f, indent=2)
print(f"Results saved to {OUT_DIR}/robustness_results.json")
PYEOF

# Step 3: Evaluate base model for comparison
echo "=== Evaluating base LLaMA (no LoRA) ==="
python3 << 'PYEOF'
import json
import random
import sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, ".")
from clarimol.data.sample import Sample
from clarimol.tasks.prompts import build_messages
from clarimol.eval.metrics import evaluate_parsing

TASKS = ["functional_group", "ring_counting", "chain_length", "canonicalization", "fragment_assembly"]
OUT_DIR = "output/robustness"
BASE_NAME = "meta-llama/Llama-3.1-8B-Instruct"
BATCH_SIZE = 16

print("Loading base model (no LoRA)...")
tokenizer = AutoTokenizer.from_pretrained(BASE_NAME, padding_side="left", trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(BASE_NAME, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
model.eval()

def run_inference(model, tokenizer, samples, batch_size=16):
    rng = random.Random(0)
    predictions = []
    for i in range(0, len(samples), batch_size):
        batch = samples[i:i+batch_size]
        prompts = []
        for s in batch:
            msgs = build_messages(s, rng=rng, use_system_prompt=True)
            msgs_no_answer = [m for m in msgs if m["role"] != "assistant"]
            try:
                prompt = tokenizer.apply_chat_template(msgs_no_answer, tokenize=False, add_generation_prompt=True)
            except Exception:
                prompt = "\n".join(f"<|{m['role']}|>\n{m['content']}" for m in msgs_no_answer) + "\n<|assistant|>\n"
            prompts.append(prompt)
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1920).to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=128, do_sample=False, pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        for j, output in enumerate(outputs):
            input_len = inputs["input_ids"][j].shape[0]
            text = tokenizer.decode(output[input_len:], skip_special_tokens=True).strip()
            predictions.append(text)
    return predictions

all_results = {}
for task in TASKS:
    canon_path = f"{OUT_DIR}/canonical_{task}.json"
    rand_path = f"{OUT_DIR}/randomized_{task}.json"
    if not Path(canon_path).exists():
        print(f"Skipping {task}: no data")
        continue

    with open(canon_path) as f:
        canon_data = json.load(f)
    with open(rand_path) as f:
        rand_data = json.load(f)

    canon_samples = [Sample(smiles=d["smiles"], task=task, question=d["question"], answer=d["answer"], metadata=d.get("metadata", {})) for d in canon_data]
    rand_samples = [Sample(smiles=d["smiles"], task=task, question=d["question"], answer=d["answer"], metadata=d.get("metadata", {})) for d in rand_data]

    print(f"  {task}: {len(canon_samples)} canonical, {len(rand_samples)} randomized")

    canon_preds = run_inference(model, tokenizer, canon_samples)
    canon_result = evaluate_parsing(canon_preds, [s.answer for s in canon_samples], task, [s.metadata for s in canon_samples])

    rand_preds = run_inference(model, tokenizer, rand_samples)
    rand_result = evaluate_parsing(rand_preds, [s.answer for s in rand_samples], task, [s.metadata for s in rand_samples])

    all_results[task] = {
        "canonical_accuracy": round(canon_result.accuracy, 4),
        "randomized_accuracy": round(rand_result.accuracy, 4),
        "drop_pp": round((rand_result.accuracy - canon_result.accuracy) * 100, 1),
    }
    print(f"    canon={canon_result.accuracy:.4f}, rand={rand_result.accuracy:.4f}, drop={all_results[task]['drop_pp']:.1f}pp")

with open(f"{OUT_DIR}/robustness_base_results.json", "w") as f:
    json.dump(all_results, f, indent=2)
print(f"Base results saved to {OUT_DIR}/robustness_base_results.json")
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

print(f"{'Task':25s}  {'Canon':>7s}  {'Random':>7s}  {'Drop(pp)':>9s}  {'Consist':>8s}  {'Base Drop(pp)':>14s}")
print("-" * 80)

for task in sorted(zinc.keys()):
    z = zinc[task]
    b = base.get(task, {})
    base_drop = b.get("drop_pp", float("nan"))
    print(f"{task:25s}  "
          f"{z['canonical_accuracy']:7.4f}  "
          f"{z['randomized_accuracy']:7.4f}  "
          f"{z.get('drop_pp', 0):+9.1f}  "
          f"{z.get('consistency', 0):8.4f}  "
          f"{base_drop:+14.1f}")

print("\nConsistency = fraction of molecules where ALL 5 random SMILES variants are correct")
print("Lower drop + higher consistency = more robust to SMILES representation choice")
PYEOF

echo "=== Complete at $(date) ==="
