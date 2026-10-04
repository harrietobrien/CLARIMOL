"""
FGBench evaluation (Liu et al., NeurIPS 2025 D&B).

Evaluates base vs CLARIMOL-trained models on the FGBench benchmark:
7K curated questions testing functional-group-level molecular property
reasoning. Each question presents a molecule modification (adding or
removing a functional group) and asks whether a property increases.

Dataset: https://huggingface.co/datasets/xuan-liu/FGBench
Paper: arXiv:2508.01055

The test split contains questions in this format:
  "For a molecule whose SMILES with atom number is [CH3:0]...,
   its label for the property of <property> is <value>.
   After modifying the molecule by <operation>...
   Does the property of the modified molecule increase?
   Your final answer should be 'True' or 'False'."

Ground truth answers are "True" or "False".
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import re
import sys
from pathlib import Path

import torch
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def load_fgbench(max_samples: int | None = None) -> list[dict]:
    """Load FGBench test split from HuggingFace."""
    from datasets import load_dataset

    logger.info("Loading FGBench test split from xuan-liu/FGBench...")
    ds = load_dataset("xuan-liu/FGBench", split="test")
    logger.info("FGBench test split: %d samples", len(ds))

    samples = [dict(row) for row in ds]
    if max_samples and max_samples < len(samples):
        samples = samples[:max_samples]
        logger.info("Capped to %d samples", max_samples)

    return samples


def load_model(base_model_name: str, adapter_path: str | None = None):
    """Load a model with optional LoRA adapter, matching CLARIMOL conventions."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    tok_source = adapter_path if adapter_path else base_model_name
    logger.info("Loading tokenizer from %s", tok_source)
    tokenizer = AutoTokenizer.from_pretrained(
        tok_source, padding_side="left", trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    logger.info("Loading base model %s (4-bit quantized)", base_model_name)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        quantization_config=bnb_config,
        trust_remote_code=True,
    )

    if adapter_path:
        logger.info("Loading LoRA adapter from %s", adapter_path)
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model, tokenizer


@torch.inference_mode()
def evaluate_fgbench(
    model,
    tokenizer,
    samples: list[dict],
    batch_size: int = 1,
) -> dict:
    """
    Run FGBench evaluation.

    FGBench questions ask "Does the property of the modified molecule
    increase? Your final answer should be 'True' or 'False'."

    Accuracy = fraction where extracted True/False matches ground truth.
    """
    correct = 0
    total = 0
    parse_failures = 0
    results_by_property = {}
    results_by_type = {}
    predictions = []

    for sample in tqdm(samples, desc="FGBench inference"):
        question = sample["question"]
        ground_truth = sample["answer"].strip()
        property_name = sample.get("property_name", "unknown")
        q_type = sample.get("type", "unknown")

        # Format as chat prompt
        messages = [{"role": "user", "content": question.strip()}]
        try:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            prompt = f"<|user|>\n{question.strip()}\n<|assistant|>\n"

        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=1536
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        outputs = model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()

        # Extract True/False from response
        response_lower = response.lower().strip()

        # Look for explicit True/False
        extracted = None
        if re.search(r"\btrue\b", response_lower):
            extracted = "True"
        elif re.search(r"\bfalse\b", response_lower):
            extracted = "False"
        elif re.search(r"\byes\b", response_lower):
            extracted = "True"
        elif re.search(r"\bno\b", response_lower):
            extracted = "False"
        elif re.search(r"\bincrease\b", response_lower):
            extracted = "True"
        elif re.search(r"\bdecrease\b", response_lower):
            extracted = "False"

        if extracted is None:
            parse_failures += 1
            is_correct = False
        else:
            is_correct = extracted == ground_truth

        correct += int(is_correct)
        total += 1

        # Track per-property and per-type accuracy
        for group_name, group_key in [
            (property_name, "by_property"),
            (q_type, "by_type"),
        ]:
            store = results_by_property if group_key == "by_property" else results_by_type
            if group_name not in store:
                store[group_name] = {"correct": 0, "total": 0}
            store[group_name]["correct"] += int(is_correct)
            store[group_name]["total"] += 1

        predictions.append({
            "question": question[:200],
            "ground_truth": ground_truth,
            "raw_response": response[:200],
            "extracted": extracted,
            "correct": is_correct,
            "property_name": property_name,
            "type": q_type,
            "target_smiles": sample.get("target_smiles", ""),
            "ref_smiles": sample.get("ref_smiles", ""),
        })

    # Compute per-group accuracies
    for store in [results_by_property, results_by_type]:
        for k in store:
            store[k]["accuracy"] = (
                store[k]["correct"] / store[k]["total"]
                if store[k]["total"] > 0
                else 0.0
            )

    overall_acc = correct / total if total > 0 else 0.0

    return {
        "overall_accuracy": overall_acc,
        "correct": correct,
        "total": total,
        "parse_failures": parse_failures,
        "by_property": results_by_property,
        "by_type": results_by_type,
        "predictions": predictions,
    }


def main():
    parser = argparse.ArgumentParser(description="FGBench evaluation")
    parser.add_argument(
        "--output-dir",
        default="output/benchmarks/fgbench",
        help="Directory for results",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Cap number of test samples (for debugging)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Inference batch size",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset once
    samples = load_fgbench(max_samples=args.max_samples)

    # Model configurations: (label, base_model, adapter_path_or_None)
    MODELS = [
        (
            "llama-8b-base",
            "meta-llama/Llama-3.1-8B-Instruct",
            None,
        ),
        (
            "llama-8b-clarimol",
            "meta-llama/Llama-3.1-8B-Instruct",
            "output/multi_seed/llama-8b/seed_42/final",
        ),
        (
            "mistral-7b-base",
            "mistralai/Mistral-7B-Instruct-v0.3",
            None,
        ),
        (
            "mistral-7b-clarimol",
            "mistralai/Mistral-7B-Instruct-v0.3",
            "output/multi_seed/mistral-7b/seed_42/final",
        ),
    ]

    all_results = {}

    for label, base_model, adapter_path in MODELS:
        results_file = os.path.join(args.output_dir, f"{label}_results.json")
        if os.path.exists(results_file):
            logger.info("SKIP: %s results already exist at %s", label, results_file)
            with open(results_file) as f:
                all_results[label] = json.load(f)
            continue

        logger.info("Evaluating %s...", label)
        model, tokenizer = load_model(base_model, adapter_path)
        result = evaluate_fgbench(model, tokenizer, samples, batch_size=args.batch_size)

        # Save per-model results (without bulky predictions for the combined file)
        with open(results_file, "w") as f:
            json.dump(result, f, indent=2)

        # Save predictions separately
        pred_file = os.path.join(args.output_dir, f"{label}_predictions.jsonl")
        with open(pred_file, "w") as f:
            for p in result.get("predictions", []):
                f.write(json.dumps(p) + "\n")

        logger.info(
            "  %s: accuracy=%.4f (%d/%d), parse_failures=%d",
            label,
            result["overall_accuracy"],
            result["correct"],
            result["total"],
            result["parse_failures"],
        )

        # Store summary (without per-sample predictions) for combined output
        summary = {k: v for k, v in result.items() if k != "predictions"}
        all_results[label] = summary

        # Free GPU memory
        del model, tokenizer
        torch.cuda.empty_cache()
        gc.collect()

    # Print comparison
    print("\n\nFGBench Results Summary")
    print(f"{'Model':30s}  {'Accuracy':>10s}  {'Correct':>8s}  {'Total':>6s}  {'ParseFail':>10s}")
    print("-" * 70)
    for label, res in all_results.items():
        print(
            f"{label:30s}  {res['overall_accuracy']:10.4f}  "
            f"{res['correct']:8d}  {res['total']:6d}  "
            f"{res.get('parse_failures', 0):10d}"
        )

    # Show per-property breakdown for CLARIMOL vs base (LLaMA)
    for model_pair in [("llama-8b-base", "llama-8b-clarimol"), ("mistral-7b-base", "mistral-7b-clarimol")]:
        base_label, trained_label = model_pair
        if base_label in all_results and trained_label in all_results:
            print(f"\nPer-property comparison: {base_label} vs {trained_label}")
            print(f"{'Property':50s}  {'Base':>8s}  {'Trained':>8s}  {'Delta':>8s}")
            print("-" * 80)
            base_props = all_results[base_label].get("by_property", {})
            trained_props = all_results[trained_label].get("by_property", {})
            all_props = sorted(set(list(base_props.keys()) + list(trained_props.keys())))
            for prop in all_props:
                b_acc = base_props.get(prop, {}).get("accuracy", 0.0)
                t_acc = trained_props.get(prop, {}).get("accuracy", 0.0)
                delta = t_acc - b_acc
                print(f"{prop:50s}  {b_acc:8.4f}  {t_acc:8.4f}  {delta:+8.4f}")

    # Save combined results
    combined_path = os.path.join(args.output_dir, "results.json")
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Combined results saved to %s", combined_path)


if __name__ == "__main__":
    main()
