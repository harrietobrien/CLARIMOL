"""
Layer-wise representation extraction for CLARIMOL probing analysis.

Extracts hidden states from multiple intermediate layers of a fine-tuned
LLaMA-8B model using forward hooks. Enables probing how task-specific
representations evolve across the depth of the transformer.

Target layers: [0, 4, 8, 12, 16, 20, 24, 28, 31] (9 layers spanning the
full 32-layer model).
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

TARGET_LAYERS = [0, 4, 8, 12, 16, 20, 24, 28, 31]


def load_model(model_path: str):
    """Load a LoRA-adapted model and tokenizer in bf16 without quantization."""
    adapter_config_path = Path(model_path) / "adapter_config.json"
    if adapter_config_path.exists():
        with open(adapter_config_path) as f:
            adapter_config = json.load(f)
        base_model_name = adapter_config["base_model_name_or_path"]
        logger.info("Loading base model %s with LoRA adapter from %s", base_model_name, model_path)
    else:
        base_model_name = model_path
        logger.info("Loading plain model from %s (no adapter found)", model_path)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path if adapter_config_path.exists() else base_model_name,
        padding_side="left",
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    if adapter_config_path.exists():
        model = PeftModel.from_pretrained(base_model, model_path)
    else:
        model = base_model

    model.eval()
    return model, tokenizer


def build_prompt(sample: Sample, tokenizer, rng: random.Random) -> str:
    """Construct the prompt string (without the assistant answer) for a sample."""
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
    return prompt


def extract_layerwise(
    model_path: str,
    data_dir: str,
    output_path: str,
    samples_per_task: int = 1000,
    seed: int = 42,
    max_seq_length: int = 1920,
):
    """
    Extract hidden states from multiple layers for each sample.

    Registers forward hooks on target transformer layers to capture
    intermediate activations at the last token position.

    Parameters
    ----------
    model_path : str
        Path to the trained model checkpoint (LoRA adapter directory).
    data_dir : str
        Directory containing task JSON files.
    output_path : str
        Destination path for the output .npz file.
    samples_per_task : int
        Number of samples to extract per task.
    seed : int
        Random seed for sample selection reproducibility.
    max_seq_length : int
        Maximum token length for input prompts.
    """
    logger.info("Loading model from %s", model_path)
    model, tokenizer = load_model(model_path)

    logger.info("Loading test data from %s", data_dir)
    all_samples = load_dataset_from_disk(data_dir)

    rng = random.Random(seed)

    selected_samples: list[Sample] = []
    for task_name, samples in sorted(all_samples.items()):
        if len(samples) > samples_per_task:
            chosen = rng.sample(samples, samples_per_task)
        else:
            chosen = samples
        selected_samples.extend(chosen)
        logger.info("Selected %d samples for task %s", len(chosen), task_name)

    logger.info("Total samples to process: %d", len(selected_samples))
    logger.info("Target layers: %s", TARGET_LAYERS)

    # Determine the actual model reference for hook registration.
    # PeftModel wraps the base model; the transformer layers live
    # at model.base_model.model.model.layers for PeftModel or
    # model.model.layers for a plain HF model.
    if hasattr(model, "base_model"):
        transformer_layers = model.base_model.model.model.layers
    else:
        transformer_layers = model.model.layers

    # Register forward hooks on target layers
    activations: dict[int, np.ndarray] = {}

    def make_hook(layer_idx: int):
        def hook(module, input, output):
            # output is a tuple; first element is the hidden state tensor
            activations[layer_idx] = output[0][:, -1, :].detach().cpu().float().numpy()
        return hook

    hooks = []
    for layer_idx in TARGET_LAYERS:
        h = transformer_layers[layer_idx].register_forward_hook(make_hook(layer_idx))
        hooks.append(h)
        logger.info("Registered hook on layer %d", layer_idx)

    # Storage: one list per target layer
    layer_hidden_states: dict[int, list[np.ndarray]] = {li: [] for li in TARGET_LAYERS}
    smiles_list: list[str] = []
    tasks_list: list[str] = []
    correct_list: list[bool] = []
    smiles_lengths_list: list[int] = []

    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    prompt_rng = random.Random(seed + 1)

    for idx, sample in enumerate(tqdm(selected_samples, desc="Extracting layer-wise representations")):
        prompt = build_prompt(sample, tokenizer, prompt_rng)

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_seq_length,
        )
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs["attention_mask"].to(model.device)

        if input_ids.shape[1] < 1:
            logger.warning("Empty tokenization for sample %d, skipping", idx)
            continue

        activations.clear()

        with torch.no_grad():
            _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        for li in TARGET_LAYERS:
            layer_hidden_states[li].append(activations[li].squeeze(0))

        smiles_list.append(sample.smiles)
        tasks_list.append(sample.task)
        correct_list.append(False)  # No generation pass; correctness not evaluated here
        smiles_lengths_list.append(len(sample.smiles))

        if (idx + 1) % 100 == 0:
            logger.info("Processed %d / %d samples", idx + 1, len(selected_samples))

    # Remove hooks
    for h in hooks:
        h.remove()

    # Build save dict
    save_dict = {}
    for li in TARGET_LAYERS:
        arr = np.stack(layer_hidden_states[li], axis=0)
        save_dict[f"hidden_states_layer_{li}"] = arr
        logger.info("Layer %d hidden states shape: %s", li, arr.shape)

    save_dict["smiles"] = np.array(smiles_list, dtype=object)
    save_dict["tasks"] = np.array(tasks_list, dtype=object)
    save_dict["correct"] = np.array(correct_list, dtype=bool)
    save_dict["smiles_lengths"] = np.array(smiles_lengths_list, dtype=np.int32)

    np.savez_compressed(output_path, **save_dict)
    logger.info("Saved layer-wise representations to %s", output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract layer-wise hidden-state representations from a trained LLM."
    )
    parser.add_argument("--model-path", type=str, required=True, help="Path to the trained model checkpoint.")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory with task JSON files.")
    parser.add_argument("--output", type=str, required=True, help="Output .npz file path.")
    parser.add_argument("--samples-per-task", type=int, default=1000, help="Number of samples per task.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--max-seq-length", type=int, default=1920, help="Maximum input sequence length.")
    args = parser.parse_args()

    extract_layerwise(
        model_path=args.model_path,
        data_dir=args.data_dir,
        output_path=args.output,
        samples_per_task=args.samples_per_task,
        seed=args.seed,
        max_seq_length=args.max_seq_length,
    )
