"""
Training Pipeline for CLARIMOL:
    1. Pre-train on SMILES parsing tasks (this module)
    2. Fine-tune on downstream tasks (same pipeline, different data)

(Supports both Unsloth-accelerated and standard HuggingFace workflows)
"""

from __future__ import annotations
import logging
import random
from pathlib import Path
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch
from trl import SFTTrainer, SFTConfig
from clarimol.data.dataset import load_dataset_from_disk
from clarimol.data.sample import Sample
from clarimol.tasks.prompts import build_messages
from clarimol.train.config import TrainConfig


logger = logging.getLogger(__name__)


def _load_model_tokenizer(config: TrainConfig):
    """Load model + tokenizer, applying LoRA. Returns (model, tokenizer)"""
    if config.use_unsloth:
        try:
            from unsloth import FastLanguageModel
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=config.model_name,
                max_seq_length=config.model_max_length,
                load_in_4bit=config.load_in_4bit,
            )
            model = FastLanguageModel.get_peft_model(
                model,
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=config.lora_target_modules,
                use_gradient_checkpointing=config.gradient_checkpointing,
            )
            return model, tokenizer
        except ImportError:
            logger.warning("Unsloth not available, falling back to standard HF + PEFT")
    bnb_config = None
    if config.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        model_max_length=config.model_max_length,
        padding_side="right",
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    if config.load_in_4bit:
        model = prepare_model_for_kbit_training(model)
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer


def _build_hf_dataset(
    samples: dict[str, list[Sample]],
    tokenizer,
    config: TrainConfig,
) -> Dataset:
    """Convert task samples into a HuggingFace Dataset w/ formatted text"""
    rng = random.Random(config.seed)
    all_texts: list[str] = list()
    for task_name, task_samples in samples.items():
        for sample in task_samples:
            messages = build_messages(sample, rng=rng, use_system_prompt=True)
            try:
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
            except Exception:
                # Fallback: simple concatenation
                text = ""
                for msg in messages:
                    text += f"<|{msg['role']}|>\n{msg['content']}\n"
            all_texts.append(text)
    # Shuffle to interleave tasks
    rng_shuffle = random.Random(config.seed)
    rng_shuffle.shuffle(all_texts)
    return Dataset.from_dict({"text": all_texts})


def run_training(config: TrainConfig) -> str:
    """
    Execute the full training pipeline

    :return: the path to the saved model
    """
    logger.info("Loading model: %s", config.model_name)
    model, tokenizer = _load_model_tokenizer(config)
    logger.info("Loading dataset from: %s", config.data_dir)
    samples = load_dataset_from_disk(config.data_dir)
    logger.info("Building HF dataset (%d tasks)", len(samples))
    train_ds = _build_hf_dataset(samples, tokenizer, config)
    logger.info("Training dataset: %d examples", len(train_ds))
    # Configure SFT
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    training_args = SFTConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        num_train_epochs=config.epochs,
        lr_scheduler_type=config.lr_scheduler,
        max_grad_norm=config.max_grad_norm,
        fp16=config.fp16,
        bf16=config.bf16,
        logging_steps=config.logging_steps,
        eval_strategy="no",
        save_strategy="steps",
        save_steps=config.save_steps,
        seed=config.seed,
        max_length=config.model_max_length,
        dataset_text_field="text",
        report_to="wandb" if config.use_wandb else "none",
        run_name=config.wandb_project,
    )
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        args=training_args,
    )
    logger.info("Starting training...")
    trainer.train()
    # Save final model
    final_path = str(output_dir / "final")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    logger.info("Model saved to %s", final_path)
    return final_path