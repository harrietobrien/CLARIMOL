"""
Representation extraction pipeline for CLARIMOL probing analysis.

Loads a trained LoRA model, runs forward passes on test samples,
and extracts the final hidden state at the last input token position.
These representations enable downstream probing experiments
(linear probes, dimensionality reduction, cluster analysis).
"""
from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from clarimol.data.dataset import load_dataset_from_disk
from clarimol.data.sample import Sample
from clarimol.tasks.prompts import build_messages

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_model(model_path: str):
    """Load a LoRA-adapted model and tokenizer for forward-pass extraction."""
    adapter_config_path = Path(model_path) / "adapter_config.json"
    if adapter_config_path.exists():
        with open(adapter_config_path) as f:
            adapter_config = json.load(f)
        base_model_name = adapter_config["base_model_name_or_path"]
        logger.info("Loading base model %s with LoRA adapter from %s", base_model_name, model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left", trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map="auto",
            quantization_config=bnb_config,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base_model, model_path)
        model.eval()
        return model, tokenizer
    else:
        # Plain model (no adapter)
        tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left", trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16,
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
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    generated_tokens = outputs[0, input_ids.shape[1]:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


def check_correctness(prediction: str, reference: str, task: str) -> bool:
    """Determine whether a prediction matches the reference answer."""
    import re
    from rdkit import Chem

    pred = prediction.strip()
    ref = reference.strip()

    # Strip <think>...</think> blocks from prediction
    pred = re.sub(r"<think>.*?</think>\s*", "", pred, flags=re.DOTALL).strip()

    if task in ("canonicalization", "fragment_assembly"):
        # Extract first SMILES-like token from prediction
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


def extract_representations(
    model_path: str,
    data_dir: str,
    output_path: str,
    samples_per_task: int = 1000,
    seed: int = 42,
    max_seq_length: int = 1920,
):
    """
    Main extraction loop.

    For each sample: tokenize prompt, run a forward pass with
    output_hidden_states=True, capture the hidden state at the last
    input token from the final transformer layer. Also generate a
    short prediction for correctness labeling.

    Parameters
    ----------
    model_path : str
        Path to the trained model checkpoint (LoRA adapter directory).
    data_dir : str
        Directory containing task JSON files for evaluation.
    output_path : str
        Destination path for the output .npz file.
    samples_per_task : int
        Number of samples to extract per task.
    seed : int
        Random seed for sample selection reproducibility.
    max_seq_length : int
        Maximum token length for input prompts. Longer prompts are skipped.
    """
    logger.info("Loading model from %s", model_path)
    model, tokenizer = load_model(model_path)

    logger.info("Loading test data from %s", data_dir)
    all_samples = load_dataset_from_disk(data_dir)

    rng = random.Random(seed)

    # Select samples_per_task random samples from each task
    selected_samples: list[Sample] = []
    for task_name, samples in sorted(all_samples.items()):
        if len(samples) > samples_per_task:
            chosen = rng.sample(samples, samples_per_task)
        else:
            chosen = samples
        selected_samples.extend(chosen)
        logger.info("Selected %d samples for task %s", len(chosen), task_name)

    logger.info("Total samples to process: %d", len(selected_samples))

    # Storage lists
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

    for idx, sample in enumerate(tqdm(selected_samples, desc="Extracting representations")):
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

        # Forward pass to extract hidden states
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        # Extract hidden state from the final layer at the last input token
        # outputs.hidden_states is a tuple of (n_layers + 1) tensors, each (batch, seq, hidden_dim)
        final_hidden = outputs.hidden_states[-1]  # (1, seq_len, hidden_dim)
        last_token_hidden = final_hidden[0, -1, :].cpu().float().numpy()  # (hidden_dim,)

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

        if (idx + 1) % 500 == 0:
            logger.info(
                "Processed %d / %d samples. Correct so far: %d / %d",
                idx + 1, len(selected_samples),
                sum(correct_list), len(correct_list),
            )

    # Stack and save
    hidden_states_array = np.stack(hidden_states_list, axis=0)  # (N, hidden_dim)
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
    logger.info("Saved representations to %s", output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract hidden-state representations from a trained LLM.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the trained model checkpoint.")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory with task JSON files.")
    parser.add_argument("--output", type=str, required=True, help="Output .npz file path.")
    parser.add_argument("--samples-per-task", type=int, default=1000, help="Number of samples per task.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--max-seq-length", type=int, default=1920, help="Maximum input sequence length.")
    args = parser.parse_args()

    extract_representations(
        model_path=args.model_path,
        data_dir=args.data_dir,
        output_path=args.output,
        samples_per_task=args.samples_per_task,
        seed=args.seed,
        max_seq_length=args.max_seq_length,
    )
