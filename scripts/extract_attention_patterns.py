"""
Attention pattern extraction for CLARIMOL mechanistic analysis.

Extracts per-head attention weights from both the fine-tuned LLaMA-8B model
and the untrained base model. For each sample, attention from the last token
to all other tokens is aggregated by SMILES token category (ring-closure
digits, branch parentheses, bond symbols, atom characters, other).

This provides direct mechanistic evidence of how fine-tuning reshapes
attention allocation toward syntactically meaningful SMILES tokens.
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

CATEGORIES = ["ring_digits", "branches", "bonds", "atoms", "other"]
RING_DIGITS = set("123456789")
BRANCH_CHARS = set("()")
BOND_CHARS = set("=#")
ATOM_CHARS = set("CNOSPFIBrcnospfib")


def classify_token(token_str: str) -> int:
    """
    Classify a token string into a SMILES category index.

    Returns the index into CATEGORIES based on the dominant character type
    in the token string. Tokens containing ring-closure digits are prioritized,
    followed by branch parentheses, bond symbols, and atom characters.

    Parameters
    ----------
    token_str : str
        The decoded token string (may contain special characters from
        the tokenizer such as leading spaces or subword markers).

    Returns
    -------
    int
        Index into the CATEGORIES list.
    """
    clean = token_str.replace("▁", "").replace("Ġ", "").replace(" ", "").strip()
    if not clean:
        return 4  # other

    if any(c in RING_DIGITS for c in clean):
        non_digit = clean.translate(str.maketrans("", "", "123456789"))
        if len(non_digit) < len(clean):
            return 0  # ring_digits
    if any(c in BRANCH_CHARS for c in clean):
        return 1  # branches
    if any(c in BOND_CHARS for c in clean):
        return 2  # bonds
    if any(c in ATOM_CHARS for c in clean):
        return 3  # atoms
    return 4  # other


def load_finetuned_model(adapter_path: str):
    """Load the fine-tuned LoRA model in bf16 for attention extraction."""
    adapter_config_path = Path(adapter_path) / "adapter_config.json"
    with open(adapter_config_path) as f:
        adapter_config = json.load(f)
    base_model_name = adapter_config["base_model_name_or_path"]
    logger.info("Loading base model %s with LoRA adapter from %s", base_model_name, adapter_path)

    tokenizer = AutoTokenizer.from_pretrained(adapter_path, padding_side="left", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    return model, tokenizer, base_model_name


def load_base_model(model_name: str):
    """Load the untrained base model in bf16."""
    logger.info("Loading base model %s (no adapter)", model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
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


def extract_attention_for_sample(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> np.ndarray:
    """
    Run a forward pass with output_attentions=True and aggregate attention
    from the last token to each token category.

    Parameters
    ----------
    model : transformers model
        The causal LM to extract attention from.
    tokenizer : transformers tokenizer
        Tokenizer used for decoding token IDs to strings.
    input_ids : torch.Tensor
        Token IDs, shape (1, seq_len).
    attention_mask : torch.Tensor
        Attention mask, shape (1, seq_len).

    Returns
    -------
    np.ndarray
        Aggregated attention per category, shape (n_layers, n_heads, n_categories).
    """
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
        )

    # outputs.attentions is a tuple of (batch, heads, seq, seq) per layer
    token_ids = input_ids[0].cpu().tolist()
    token_strings = tokenizer.convert_ids_to_tokens(token_ids)

    # Classify each token position
    seq_len = len(token_strings)
    token_categories = np.array([classify_token(t) for t in token_strings], dtype=np.int32)

    n_layers = len(outputs.attentions)
    n_heads = outputs.attentions[0].shape[1]
    n_categories = len(CATEGORIES)

    result = np.zeros((n_layers, n_heads, n_categories), dtype=np.float32)

    for layer_idx, attn_tensor in enumerate(outputs.attentions):
        # attn_tensor shape: (1, n_heads, seq_len, seq_len)
        # Extract attention from the last token to all positions
        last_token_attn = attn_tensor[0, :, -1, :].cpu().float().numpy()
        # last_token_attn shape: (n_heads, seq_len)

        for cat_idx in range(n_categories):
            mask = token_categories == cat_idx
            if mask.any():
                result[layer_idx, :, cat_idx] = last_token_attn[:, mask].mean(axis=1)

    return result


def extract_attention_patterns(
    adapter_path: str,
    data_dir: str,
    output_path: str,
    samples_per_task: int = 200,
    seed: int = 42,
    max_seq_length: int = 1920,
):
    """
    Extract attention patterns from both fine-tuned and base models.

    Parameters
    ----------
    adapter_path : str
        Path to the fine-tuned LoRA adapter directory.
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

    # Load test data and select samples
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

    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    ft_checkpoint_path = Path(output_path).with_suffix(".ft_checkpoint.npz")
    bl_checkpoint_path = Path(output_path).with_suffix(".bl_checkpoint.npz")

    n_total = len(selected_samples)

    def run_phase(model, tokenizer, checkpoint_path, desc, phase_seed):
        """Extract attention for one model, with periodic checkpointing."""
        start_idx = 0
        attention_list: list[np.ndarray] = []

        if checkpoint_path.exists():
            logger.info("Found checkpoint at %s, resuming...", checkpoint_path)
            ckpt = np.load(checkpoint_path, allow_pickle=True)
            start_idx = int(ckpt["completed"])
            attention_list = list(ckpt["attention"])
            logger.info("Resuming from sample %d / %d", start_idx, n_total)

        # Each sample gets a deterministic per-sample rng so resume is exact
        for idx in tqdm(range(start_idx, n_total), desc=desc, initial=start_idx, total=n_total):
            sample = selected_samples[idx]
            sample_rng = random.Random(phase_seed + idx)
            prompt = build_prompt(sample, tokenizer, sample_rng)

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

            attn = extract_attention_for_sample(model, tokenizer, input_ids, attention_mask)
            attention_list.append(attn)

            if (idx + 1) % 50 == 0:
                logger.info("%s: processed %d / %d samples", desc, idx + 1, n_total)

            # Checkpoint every 200 samples
            if (idx + 1) % 200 == 0:
                np.savez_compressed(
                    checkpoint_path,
                    attention=np.stack(attention_list, axis=0),
                    completed=np.array(idx + 1),
                )
                logger.info("Checkpoint saved at sample %d", idx + 1)

        return attention_list

    # Phase 1: Fine-tuned model
    logger.info("=== Phase 1: Fine-tuned model ===")
    ft_model, ft_tokenizer, base_model_name = load_finetuned_model(adapter_path)

    smiles_list = [s.smiles for s in selected_samples]
    tasks_list = [s.task for s in selected_samples]

    ft_attention_list = run_phase(ft_model, ft_tokenizer, ft_checkpoint_path, "Fine-tuned attention", seed + 1)

    del ft_model
    torch.cuda.empty_cache()

    # Phase 2: Base model
    logger.info("=== Phase 2: Base (untrained) model ===")
    bl_model, bl_tokenizer = load_base_model(base_model_name)

    bl_attention_list = run_phase(bl_model, bl_tokenizer, bl_checkpoint_path, "Base attention", seed + 1)

    del bl_model
    torch.cuda.empty_cache()

    # Stack and save
    ft_attention = np.stack(ft_attention_list, axis=0)
    bl_attention = np.stack(bl_attention_list, axis=0)

    logger.info(
        "Extraction complete. Fine-tuned shape: %s, Base shape: %s",
        ft_attention.shape, bl_attention.shape,
    )

    np.savez_compressed(
        output_path,
        ft_attention=ft_attention,
        bl_attention=bl_attention,
        categories=np.array(CATEGORIES, dtype=object),
        tasks=np.array(tasks_list, dtype=object),
        smiles=np.array(smiles_list, dtype=object),
    )
    logger.info("Saved attention patterns to %s", output_path)

    # Clean up checkpoints on success
    for ckpt_path in [ft_checkpoint_path, bl_checkpoint_path]:
        if ckpt_path.exists():
            ckpt_path.unlink()
            logger.info("Removed checkpoint file: %s", ckpt_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract attention patterns from fine-tuned and base LLaMA-8B models."
    )
    parser.add_argument(
        "--adapter-path", type=str, required=True,
        help="Path to the fine-tuned LoRA adapter directory.",
    )
    parser.add_argument("--data-dir", type=str, required=True, help="Directory with task JSON files.")
    parser.add_argument("--output", type=str, required=True, help="Output .npz file path.")
    parser.add_argument("--samples-per-task", type=int, default=200, help="Samples per task.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--max-seq-length", type=int, default=1920, help="Maximum input sequence length.")
    args = parser.parse_args()

    extract_attention_patterns(
        adapter_path=args.adapter_path,
        data_dir=args.data_dir,
        output_path=args.output,
        samples_per_task=args.samples_per_task,
        seed=args.seed,
        max_seq_length=args.max_seq_length,
    )
