"""
Baseline representation extraction for CLARIMOL probing analysis.

Extracts final-layer hidden states from the UNTRAINED base LLaMA-8B model
(no LoRA adapter) on the same 5000 test samples used for the fine-tuned
extraction. Enables direct comparison of representation structure before
and after fine-tuning.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, ".")
from clarimol.data.dataset import load_dataset_from_disk
from clarimol.data.sample import Sample
from clarimol.tasks.prompts import build_messages

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_base_model(model_name: str):
    """Load the base model (no adapter) in bf16."""
    logger.info("Loading base model %s (no LoRA adapter)", model_name)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side="left",
        trust_remote_code=True,
    )
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


def generate_prediction(model, tokenizer, input_ids: torch.Tensor, max_new_tokens: int = 128) -> str:
    """Generate a short prediction from the model for correctness scoring."""
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    generated_tokens = outputs[0, input_ids.shape[1]:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


def check_correctness(prediction: str, reference: str, task: str) -> bool:
    """Determine whether a prediction matches the reference answer."""
    from rdkit import Chem

    pred = prediction.strip()
    ref = reference.strip()

    # Strip <think>...</think> blocks from prediction
    pred = re.sub(r"<think>.*?</think>\s*", "", pred, flags=re.DOTALL).strip()

    if task in ("canonicalization", "fragment_assembly"):
        smiles_match = re.search(r"[A-Za-z0-9@+\-\[\]\(\)=#%/\\.]+", pred)
        if not smiles_match:
            return False
        pred_smi = smiles_match.group(0)
        pred_mol = Chem.MolFromSmiles(pred_smi)
        ref_mol = Chem.MolFromSmiles(ref)
        if pred_mol is None or ref_mol is None:
            return False
        return Chem.MolToSmiles(pred_mol, canonical=True) == Chem.MolToSmiles(ref_mol, canonical=True)
    elif task == "functional_group":
        yes_no = re.search(r"\b(yes|no)\b", pred, re.IGNORECASE)
        if yes_no is None:
            return False
        return yes_no.group(1).lower() == ref.lower()
    elif task in ("ring_counting", "chain_length"):
        integer_match = re.search(r"\b(\d+)\b", pred)
        if integer_match is None:
            return False
        return integer_match.group(1) == ref.strip()
    else:
        return pred.lower() == ref.lower()


def extract_baseline(
    base_model_name: str,
    data_dir: str,
    output_path: str,
    samples_per_task: int = 1000,
    seed: int = 42,
    max_seq_length: int = 1920,
):
    """
    Extract final-layer hidden states from the untrained base model.

    Uses the same sample selection (seed=42) and prompt construction as
    the fine-tuned extraction to ensure identical inputs for comparison.

    Parameters
    ----------
    base_model_name : str
        HuggingFace model identifier for the base model.
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
    logger.info("Loading base model %s", base_model_name)
    model, tokenizer = load_base_model(base_model_name)

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

    hidden_states_list: list[np.ndarray] = []
    smiles_list: list[str] = []
    tasks_list: list[str] = []
    correct_list: list[bool] = []
    difficulty_list: list[float] = []
    predictions_list: list[str] = []
    answers_list: list[str] = []
    smiles_lengths_list: list[int] = []

    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    prompt_rng = random.Random(seed + 1)

    for idx, sample in enumerate(tqdm(selected_samples, desc="Extracting baseline representations")):
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

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        # Final layer hidden state at the last token position
        final_hidden = outputs.hidden_states[-1]
        last_token_hidden = final_hidden[0, -1, :].cpu().float().numpy()

        # Generate prediction for correctness labeling
        prediction = generate_prediction(model, tokenizer, input_ids, max_new_tokens=128)
        is_correct = check_correctness(prediction, sample.answer, sample.task)

        hidden_states_list.append(last_token_hidden)
        smiles_list.append(sample.smiles)
        tasks_list.append(sample.task)
        correct_list.append(is_correct)
        difficulty_list.append(sample.difficulty)
        predictions_list.append(prediction)
        answers_list.append(sample.answer)
        smiles_lengths_list.append(len(sample.smiles))

        if (idx + 1) % 100 == 0:
            logger.info(
                "Processed %d / %d samples. Correct so far: %d / %d",
                idx + 1, len(selected_samples),
                sum(correct_list), len(correct_list),
            )

    hidden_states_array = np.stack(hidden_states_list, axis=0)
    logger.info(
        "Extraction complete. Shape: %s. Overall accuracy: %.4f",
        hidden_states_array.shape,
        np.mean(correct_list),
    )

    np.savez_compressed(
        output_path,
        hidden_states=hidden_states_array,
        smiles=np.array(smiles_list, dtype=object),
        tasks=np.array(tasks_list, dtype=object),
        correct=np.array(correct_list, dtype=bool),
        difficulties=np.array(difficulty_list, dtype=np.float32),
        predictions=np.array(predictions_list, dtype=object),
        answers=np.array(answers_list, dtype=object),
        smiles_lengths=np.array(smiles_lengths_list, dtype=np.int32),
    )
    logger.info("Saved baseline representations to %s", output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract hidden-state representations from the untrained base LLM."
    )
    parser.add_argument(
        "--base-model", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
        help="HuggingFace model identifier for the base model.",
    )
    parser.add_argument("--data-dir", type=str, required=True, help="Directory with task JSON files.")
    parser.add_argument("--output", type=str, required=True, help="Output .npz file path.")
    parser.add_argument("--samples-per-task", type=int, default=1000, help="Number of samples per task.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--max-seq-length", type=int, default=1920, help="Maximum input sequence length.")
    args = parser.parse_args()

    extract_baseline(
        base_model_name=args.base_model,
        data_dir=args.data_dir,
        output_path=args.output,
        samples_per_task=args.samples_per_task,
        seed=args.seed,
        max_seq_length=args.max_seq_length,
    )
