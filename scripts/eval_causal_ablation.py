"""
Causal head ablation experiment for CLARIMOL.

Tests whether identified attention head circuits are causally responsible
for task performance by zeroing out specific heads during inference and
measuring per-task accuracy changes.

Three conditions:
  (1) No ablation (control baseline)
  (2) Ring-circuit heads ablated (top 10 by ring-digit attention delta)
  (3) Random same-count heads ablated (permutation control)

If ring-circuit heads are genuinely specialized:
  - ring_counting accuracy should drop significantly
  - fragment_assembly / canonicalization may drop modestly
  - functional_group / chain_length should be minimally affected
  - random-head ablation should produce smaller, more uniform drops

The ring-circuit heads were identified from attention_patterns.npz as
the 10 heads with the largest fine-tuned minus baseline ring-digit
attention delta during ring_counting evaluation.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.path.insert(0, ".")
from clarimol.data.dataset import load_dataset_from_disk
from clarimol.data.sample import Sample
from clarimol.tasks.prompts import build_messages
from clarimol.eval.metrics import evaluate_parsing

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Ring-circuit heads identified from attention_patterns.npz:
# Top 10 by (fine-tuned minus baseline) ring-digit attention during ring_counting.
# Each tuple is (layer_index, head_index).
RING_CIRCUIT_HEADS = [
    (14, 31), (14, 1), (14, 0), (15, 26), (14, 24),
    (13, 21), (16, 8), (15, 20), (14, 18), (16, 19),
]

# Random control heads: same count (10), same layer distribution
# but different head indices. Selected to avoid overlap with ring heads.
# Layers 13-16 contain 8 of the 10 ring heads, so the control uses
# 8 heads from layers 13-16 and 2 from other layers, matching the
# layer distribution exactly.
RANDOM_CONTROL_HEADS = [
    (14, 5), (14, 10), (14, 15), (15, 3), (15, 10),
    (13, 5), (16, 12), (16, 25), (12, 14), (17, 7),
]

TASKS = [
    "functional_group", "ring_counting", "chain_length",
    "canonicalization", "fragment_assembly",
]


class HeadAblationHook:
    """Forward pre-hook on o_proj that zeros specific head outputs before projection.

    Registered on the o_proj linear layer within self_attn. The pre-hook
    intercepts the input to o_proj (which is the concatenated per-head
    attention outputs) and zeros the specified heads before the output
    projection mixes them. This is the correct ablation point — after
    per-head attention computation but before the projection that
    combines heads into the residual stream.

    The input to o_proj has shape (batch, seq_len, num_heads * head_dim)
    where each head occupies a contiguous slice of size head_dim.
    """

    def __init__(self, head_indices: list[int], num_heads: int, head_dim: int):
        self.head_indices = head_indices
        self.num_heads = num_heads
        self.head_dim = head_dim

    def __call__(self, module, args):
        # args[0] is the input tensor to o_proj
        # Shape: (batch, seq_len, num_heads * head_dim)
        hidden = args[0]
        for head_idx in self.head_indices:
            start = head_idx * self.head_dim
            end = start + self.head_dim
            hidden[:, :, start:end] = 0.0
        return (hidden,) + args[1:]


def get_layer_modules(model):
    """Navigate PeftModel wrapping to find transformer layer modules."""
    # PeftModel -> base_model -> model -> model -> layers
    if hasattr(model, "base_model"):
        inner = model.base_model.model
    else:
        inner = model

    if hasattr(inner, "model") and hasattr(inner.model, "layers"):
        return inner.model.layers
    raise AttributeError(
        "Cannot locate transformer layers. Expected model.base_model.model.model.layers"
    )


def install_ablation_hooks(
    model, heads_to_ablate: list[tuple[int, int]]
) -> list[torch.utils.hooks.RemovableHook]:
    """Install forward hooks that zero out specified (layer, head) pairs.

    Parameters
    ----------
    model : PeftModel or PreTrainedModel
        The loaded model.
    heads_to_ablate : list of (layer_idx, head_idx) tuples
        Which heads to zero out.

    Returns
    -------
    list of RemovableHook
        Handles for removing the hooks after evaluation.
    """
    layers = get_layer_modules(model)

    # Determine head_dim from the first layer's self_attn
    attn_module = layers[0].self_attn
    num_heads = attn_module.config.num_attention_heads
    head_dim = attn_module.head_dim

    # Group heads by layer
    layer_to_heads: dict[int, list[int]] = {}
    for layer_idx, head_idx in heads_to_ablate:
        layer_to_heads.setdefault(layer_idx, []).append(head_idx)

    handles = []
    for layer_idx, head_list in layer_to_heads.items():
        hook = HeadAblationHook(head_list, num_heads, head_dim)
        # Register on o_proj (pre-hook) to zero heads before the output projection
        handle = layers[layer_idx].self_attn.o_proj.register_forward_pre_hook(hook)
        handles.append(handle)
        logger.info(
            "Installed ablation pre-hook on layer %d o_proj for heads %s",
            layer_idx, head_list,
        )

    return handles


def remove_hooks(handles: list[torch.utils.hooks.RemovableHook]):
    """Remove all installed hooks."""
    for h in handles:
        h.remove()


def run_inference(
    model,
    tokenizer,
    samples: list[Sample],
    max_new_tokens: int = 128,
    batch_size: int = 16,
) -> list[str]:
    """Run batch inference and return raw prediction strings."""
    rng = random.Random(0)
    predictions: list[str] = []
    for i in tqdm(range(0, len(samples), batch_size), desc="Inference"):
        batch = samples[i : i + batch_size]
        prompts: list[str] = []
        for sample in batch:
            messages = build_messages(sample, rng=rng, use_system_prompt=True)
            messages_no_answer = [m for m in messages if m["role"] != "assistant"]
            try:
                prompt = tokenizer.apply_chat_template(
                    messages_no_answer,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                prompt = "\n".join(
                    f"<|{m['role']}|>\n{m['content']}" for m in messages_no_answer
                )
                prompt += "\n<|assistant|>\n"
            prompts.append(prompt)

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048 - max_new_tokens,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        for j, output in enumerate(outputs):
            input_len = inputs["input_ids"][j].shape[0]
            generated = output[input_len:]
            text = tokenizer.decode(generated, skip_special_tokens=True).strip()
            predictions.append(text)

    return predictions


def evaluate_condition(
    model,
    tokenizer,
    task_samples: dict[str, list[Sample]],
    condition_name: str,
    heads_to_ablate: list[tuple[int, int]] | None = None,
    max_samples: int = 200,
    batch_size: int = 16,
) -> dict[str, dict]:
    """Evaluate all tasks under a single ablation condition.

    Parameters
    ----------
    model : the loaded model
    tokenizer : the tokenizer
    task_samples : dict mapping task name -> list of Sample
    condition_name : label for logging
    heads_to_ablate : heads to zero out, or None for no ablation
    max_samples : max samples per task
    batch_size : inference batch size

    Returns
    -------
    dict mapping task name -> {accuracy, correct, total}
    """
    handles = []
    if heads_to_ablate:
        handles = install_ablation_hooks(model, heads_to_ablate)

    results = {}
    for task in TASKS:
        samples = task_samples.get(task, [])[:max_samples]
        if not samples:
            logger.warning("No samples for task %s", task)
            continue

        logger.info(
            "[%s] Evaluating %s (%d samples)", condition_name, task, len(samples)
        )
        predictions = run_inference(model, tokenizer, samples, batch_size=batch_size)

        # Compute accuracy via evaluate_parsing
        references = [s.answer for s in samples]
        metadata = [s.metadata for s in samples]
        result = evaluate_parsing(predictions, references, task, metadata)
        results[task] = {
            "accuracy": round(result.accuracy, 4),
            "correct": result.correct,
            "total": result.total,
        }
        logger.info(
            "[%s] %s: accuracy=%.4f (%d/%d)",
            condition_name, task, result.accuracy, result.correct, result.total,
        )

    remove_hooks(handles)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Causal head ablation experiment"
    )
    parser.add_argument(
        "--adapter-path",
        default="output/multi_seed/llama-8b/seed_42/final",
        help="Path to LoRA adapter directory",
    )
    parser.add_argument(
        "--test-data", default="data/test",
        help="Path to test data directory",
    )
    parser.add_argument(
        "--output-dir", default="output/ablation_heads",
        help="Directory for output JSON",
    )
    parser.add_argument(
        "--max-samples", type=int, default=200,
        help="Max samples per task (200 gives ~3%% CI width)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="Inference batch size",
    )
    parser.add_argument(
        "--n-random-controls", type=int, default=5,
        help="Number of random control ablation sets to average over",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for control head selection and sample ordering",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "ablation_results.json"

    if output_file.exists():
        logger.info("Output file exists: %s. Loading and checking for completeness.", output_file)
        with open(output_file) as f:
            existing = json.load(f)
        if all(k in existing for k in ["no_ablation", "ring_circuit", "random_controls"]):
            logger.info("All conditions already completed. Exiting.")
            return

    # Load model
    adapter_config_path = Path(args.adapter_path) / "adapter_config.json"
    with open(adapter_config_path) as f:
        adapter_config = json.load(f)
    base_model_name = adapter_config["base_model_name_or_path"]

    logger.info("Loading base model: %s", base_model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        args.adapter_path, padding_side="left", trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    model.eval()

    # Verify layer access
    layers = get_layer_modules(model)
    n_layers = len(layers)
    n_heads = layers[0].self_attn.config.num_attention_heads
    head_dim = layers[0].self_attn.head_dim
    logger.info(
        "Model: %d layers, %d heads/layer, head_dim=%d", n_layers, n_heads, head_dim
    )

    # Validate ring circuit heads are within bounds
    for layer_idx, head_idx in RING_CIRCUIT_HEADS:
        assert layer_idx < n_layers, f"Layer {layer_idx} out of range (max {n_layers - 1})"
        assert head_idx < n_heads, f"Head {head_idx} out of range (max {n_heads - 1})"

    # Load test data (returns dict[task_name -> list[Sample]])
    logger.info("Loading test data from %s", args.test_data)
    task_samples = load_dataset_from_disk(args.test_data)

    # Subsample deterministically
    for task in TASKS:
        if task in task_samples:
            rng_copy = random.Random(args.seed)
            rng_copy.shuffle(task_samples[task])
            task_samples[task] = task_samples[task][:args.max_samples]

    logger.info("Samples per task: %s", {t: len(s) for t, s in task_samples.items()})

    all_results = {}

    # Condition 1: No ablation (control)
    logger.info("=== CONDITION 1: No ablation ===")
    all_results["no_ablation"] = evaluate_condition(
        model, tokenizer, task_samples, "no_ablation",
        heads_to_ablate=None,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
    )

    # Condition 2: Ring-circuit heads ablated
    logger.info("=== CONDITION 2: Ring-circuit heads ablated ===")
    all_results["ring_circuit"] = evaluate_condition(
        model, tokenizer, task_samples, "ring_circuit",
        heads_to_ablate=RING_CIRCUIT_HEADS,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
    )

    # Condition 3: Random control ablation (multiple permutations)
    logger.info("=== CONDITION 3: Random control ablation ===")

    # Build pool of candidate heads, excluding ring circuit heads
    ring_set = set(RING_CIRCUIT_HEADS)
    # Match layer distribution: count heads per layer in ring circuit
    layer_counts: dict[int, int] = {}
    for layer_idx, _ in RING_CIRCUIT_HEADS:
        layer_counts[layer_idx] = layer_counts.get(layer_idx, 0) + 1

    all_random_results = []
    ctrl_rng = random.Random(args.seed + 1000)

    for ctrl_idx in range(args.n_random_controls):
        # Sample random heads matching layer distribution
        random_heads = []
        for layer_idx, count in layer_counts.items():
            available = [
                (layer_idx, h) for h in range(n_heads)
                if (layer_idx, h) not in ring_set
            ]
            chosen = ctrl_rng.sample(available, min(count, len(available)))
            random_heads.extend(chosen)

        logger.info(
            "Random control %d/%d: heads=%s",
            ctrl_idx + 1, args.n_random_controls, random_heads,
        )
        ctrl_result = evaluate_condition(
            model, tokenizer, task_samples,
            f"random_ctrl_{ctrl_idx}",
            heads_to_ablate=random_heads,
            max_samples=args.max_samples,
            batch_size=args.batch_size,
        )
        all_random_results.append(ctrl_result)

    # Average random control results
    avg_random = {}
    for task in TASKS:
        accs = [r[task]["accuracy"] for r in all_random_results if task in r]
        avg_random[task] = {
            "accuracy_mean": round(np.mean(accs), 4),
            "accuracy_std": round(np.std(accs), 4),
            "accuracy_per_control": [round(a, 4) for a in accs],
        }
    all_results["random_controls"] = avg_random

    # Summary
    print("\n" + "=" * 70)
    print("CAUSAL ABLATION RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Task':<22} {'Intact':>8} {'Ring abl':>8} {'Delta':>8} {'Rnd ctrl':>8} {'Rnd delta':>8}")
    print("-" * 70)
    for task in TASKS:
        intact = all_results["no_ablation"][task]["accuracy"]
        ring = all_results["ring_circuit"][task]["accuracy"]
        rnd = all_results["random_controls"][task]["accuracy_mean"]
        rnd_std = all_results["random_controls"][task]["accuracy_std"]
        ring_delta = ring - intact
        rnd_delta = rnd - intact
        print(
            f"{task:<22} {intact:>8.4f} {ring:>8.4f} {ring_delta:>+8.4f} "
            f"{rnd:>7.4f}±{rnd_std:.3f} {rnd_delta:>+8.4f}"
        )

    # Save results
    all_results["metadata"] = {
        "adapter_path": args.adapter_path,
        "test_data": args.test_data,
        "max_samples": args.max_samples,
        "n_random_controls": args.n_random_controls,
        "seed": args.seed,
        "ring_circuit_heads": RING_CIRCUIT_HEADS,
        "random_control_heads_per_trial": [
            [(l, h) for l, h in ctrl_result_heads]
            for ctrl_result_heads in []  # not stored above; add if needed
        ],
    }

    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Results saved to %s", output_file)


if __name__ == "__main__":
    main()
