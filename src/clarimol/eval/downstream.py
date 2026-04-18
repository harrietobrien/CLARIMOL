"""
Downstream task evaluation for molecular generation.

Evaluates on Mol-Instructions test splits using the metrics from Table 4:
    Exact match, BLEU, Levenshtein, MACCS FTS, RDK FTS, Morgan FTS, Validity
"""

from __future__ import annotations
import json
import logging
import random
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, PreTrainedModel, PreTrainedTokenizerBase
from peft import PeftModel
import torch
from tqdm import tqdm
from clarimol.data.downstream import (
    DownstreamSample,
    load_mol_instructions,
    build_downstream_messages,
)
from clarimol.eval.metrics import GenerationResult, evaluate_generation

logger = logging.getLogger(__name__)


def _load_model_for_inference(model_path: str, use_unsloth: bool = True):
    """Load a trained model + tokenizer for inference."""
    if use_unsloth:
        try:
            from unsloth import FastLanguageModel
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_path,
                max_seq_length=2048,
                load_in_4bit=True,
            )
            FastLanguageModel.for_inference(model)
            return model, tokenizer
        except (ImportError, Exception) as e:
            logger.warning("Unsloth inference failed (%s), falling back to HF", e)

    # Check if this is a LoRA adapter checkpoint
    adapter_config_path = Path(model_path) / "adapter_config.json"
    if adapter_config_path.exists():
        with open(adapter_config_path) as f:
            adapter_config = json.load(f)
        base_model_name = adapter_config["base_model_name_or_path"]
        logger.info("Loading base model %s with LoRA adapter from %s", base_model_name, model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left", trust_remote_code=True)
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

    # Plain model (no adapter)
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


@torch.inference_mode()
def generate_downstream_predictions(
    model,
    tokenizer,
    samples: list[DownstreamSample],
    max_new_tokens: int = 256,
    temperature: float = 0.2,
    batch_size: int = 4,
) -> list[str]:
    """Run batch inference on downstream samples."""
    predictions: list[str] = []
    for i in tqdm(range(0, len(samples), batch_size), desc="Inference"):
        batch = samples[i : i + batch_size]
        prompts: list[str] = []
        for sample in batch:
            messages = build_downstream_messages(sample, use_system_prompt=True)
            # Remove assistant turn
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


def evaluate_downstream(
    model_path: str,
    data_dir: str,
    tasks: list[str] | None = None,
    use_unsloth: bool = True,
    max_samples: int | None = None,
    batch_size: int = 4,
) -> dict[str, GenerationResult]:
    """
    Evaluate a model on downstream molecular generation tasks.

    :param model_path: Path to fine-tuned model checkpoint
    :param data_dir: Directory with Mol-Instructions JSON files
    :param tasks: Which tasks to evaluate (default: all three)
    :param use_unsloth: Try Unsloth for faster inference
    :param max_samples: Cap samples per task
    :param batch_size: Inference batch size
    :return: Dict mapping task name to GenerationResult
    """
    logger.info("Loading model from %s", model_path)
    model, tokenizer = _load_model_for_inference(model_path, use_unsloth)

    all_data = load_mol_instructions(data_dir, tasks=tasks, split="test")
    results: dict[str, GenerationResult] = {}

    for task_name, samples in all_data.items():
        if max_samples:
            samples = samples[:max_samples]
        logger.info("Evaluating %s (%d test samples)", task_name, len(samples))
        predictions = generate_downstream_predictions(
            model, tokenizer, samples, batch_size=batch_size,
        )
        references = [s.output for s in samples]
        result = evaluate_generation(predictions, references)
        results[task_name] = result
        logger.info(
            "  %s: exact=%.3f  bleu=%.3f  lev=%.2f  maccs=%.3f  rdk=%.3f  morgan=%.3f  valid=%.3f",
            task_name,
            result.exact_match, result.bleu, result.levenshtein,
            result.maccs_fps, result.rdk_fps, result.morgan_fps, result.validity,
        )
    return results
