#!/usr/bin/env python3
"""Debug Qwen3 seed_42 outputs on canonicalization to diagnose 0.000 accuracy."""
import json
import sys
sys.path.insert(0, "src")
from clarimol.eval.inference import load_model_and_tokenizer, batch_generate

MODEL_PATH = "output/multi_seed/qwen3-8b/seed_42/final"
DATA_PATH = "data/test/canonicalization.json"
N_SAMPLES = 5

print("Loading model...")
model, tokenizer = load_model_and_tokenizer(MODEL_PATH, no_unsloth=True)

print("Loading test samples...")
samples = json.load(open(DATA_PATH))[:N_SAMPLES]

prompts = []
refs = []
for s in samples:
    if "prompt" in s:
        prompts.append(s["prompt"])
    elif "instruction" in s:
        prompts.append(s["instruction"])
    else:
        prompts.append(str(s))
    refs.append(s.get("output", s.get("answer", "")))

print("Generating outputs...")
outputs = batch_generate(model, tokenizer, prompts, batch_size=1)

for i in range(len(prompts)):
    print("=" * 60)
    print("SAMPLE", i)
    print("PROMPT:", prompts[i][:150])
    print("REFERENCE:", refs[i][:150])
    print("RAW OUTPUT:", repr(outputs[i][:300]))
    print()
