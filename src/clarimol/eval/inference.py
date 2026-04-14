from __future__ import annotations
import logging
import random
from pathlib import Path
# get version that matches CUDA driver on desktop
from unsloth import FastLanguageModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
from clarimol.data.dataset import load_dataset_from_disk
from clarimol.data.sample import Sample
from clarimol.eval.metrics import ParsingResult, evaluate_parsing
from clarimol.tasks.prompts import build_messages

logger = logging.getLogger(__name__)


def _load_model_for_inference(model_path: str, use_unsloth: bool = True):
    """Load a trained model + tokenizer for inference"""
    if use_unsloth:
        try:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_path,
                max_seq_length=2048,
                load_in_4bit=True,
            )
            FastLanguageModel.for_inference(model)
            return model, tokenizer
        except (ImportError, Exception) as e:
            logger.warning("Unsloth inference failed (%s), falling back to HF", e)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


@torch.inference_mode()
def generate_predictions(
    model,
    tokenizer,
    samples: list[Sample],
    max_new_tokens: int = 128,
    temperature: float = 0.2,
    batch_size: int = 8,
) -> list[str]:
    """Run batch inference over samples + return raw model outputs"""
    rng = random.Random(0)
    predictions: list[str] = []
    for i in tqdm(range(0, len(samples), batch_size), desc="Inference"):
        batch = samples[i : i + batch_size]
        prompts: list[str] = []
        for sample in batch:
            messages = build_messages(sample, rng=rng, use_system_prompt=True)
            # Remove the assistant turn (what we want the model to generate)
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
        # Tokenize
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048 - max_new_tokens,
        ).to(model.device)
        # Generate
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        # Decode only the newly generated tokens
        for j, output in enumerate(outputs):
            input_len = inputs["input_ids"][j].shape[0]
            generated = output[input_len:]
            text = tokenizer.decode(generated, skip_special_tokens=True).strip()
            predictions.append(text)
    return predictions


def evaluate_model(
    model_path: str,
    data_dir: str,
    use_unsloth: bool = True,
    max_samples: int | None = None,
    batch_size: int = 8,
) -> dict[str, ParsingResult]:
    """
    Evaluation Pipeline: load model, run inference, compute metrics.

    :param: model_path: Path to saved model checkpoint
    :param: data_dir: Directory with task JSON files
    :param: use_unsloth: Try Unsloth for faster inference
    :param: max_samples: Cap samples per task (for quick eval)
    :param: batch_size: Inference batch size

    :return: Dict mapping task name to ParsingResult
    """
    logger.info("Loading model from %s", model_path)
    model, tokenizer = _load_model_for_inference(model_path, use_unsloth)
    logger.info("Loading test data from %s", data_dir)
    all_samples = load_dataset_from_disk(data_dir)
    results: dict[str, ParsingResult] = {}
    for task_name, samples in all_samples.items():
        if max_samples:
            samples = samples[:max_samples]
        logger.info("Evaluating %s (%d samples)", task_name, len(samples))
        predictions = generate_predictions(
            model, tokenizer, samples,
            batch_size=batch_size,
        )
        references = [s.answer for s in samples]
        result = evaluate_parsing(predictions, references, task_name)
        results[task_name] = result
        logger.info(
            "  %s: accuracy=%.4f (%d/%d)",
            task_name, result.accuracy, result.correct, result.total,
        )
    return results