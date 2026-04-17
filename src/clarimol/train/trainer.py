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
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, PreTrainedModel, PreTrainedTokenizerBase
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch
import trl
from transformers.models.auto.configuration_auto import replace_list_option_in_docstrings
from transformers.models.auto.tokenization_auto import TOKENIZER_MAPPING_NAMES
from clarimol.data.dataset import load_dataset_from_disk
from clarimol.data.sample import Sample
from clarimol.tasks.prompts import build_messages
from clarimol.train.config import TrainConfig


logger = logging.getLogger(__name__)

@replace_list_option_in_docstrings(TOKENIZER_MAPPING_NAMES)
def _load_model_tokenizer(config: TrainConfig):
    """Load model + tokenizer, applying LoRA. Returns (model, tokenizer)"""
    if config.use_unsloth:
        try:
            # Save originals BEFORE Unsloth import (it monkey-patches them)
            from transformers.integrations.sdpa_attention import sdpa_attention_forward as _orig_sdpa
            from transformers.loss.loss_utils import LOSS_MAPPING as _loss_map
            from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS as _attn_fns
            _orig_loss = _loss_map.get("ForCausalLM")

            # Import unsloth first; must be imported before model loading
            from unsloth import FastLanguageModel
            # Undo unsloth Triton-based patches that fail on P100 (sm_60):
            #   SDPA attention: probes mask on-GPU with .item() → CUDA error
            #   Loss function: uses Triton cross-entropy → PTX codegen failure
            # keep unsloth model loading, LoRA setup, and padding-free batching
            _attn_fns["sdpa"] = _orig_sdpa
            if _orig_loss is not None:
                _loss_map["ForCausalLM"] = _orig_loss

            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=config.model_name,
                max_seq_length=config.model_max_length,
                load_in_4bit=config.load_in_4bit,
            )
            model = FastLanguageModel.get_peft_model(
                model,
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=0,  # unsloth fast patching requires dropout=0
                target_modules=config.lora_target_modules,
                use_gradient_checkpointing=config.gradient_checkpointing,
            )
            # Fix eos_token if unsloth set a placeholder not in vocab
            if tokenizer.eos_token and tokenizer.eos_token not in tokenizer.get_vocab():
                tokenizer.eos_token = tokenizer.decode([tokenizer.eos_token_id])
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
    # noinspection PyTypeChecker
    tokenizer = AutoTokenizer.from_pretrained(  # type: PreTrainedTokenizerBase
        config.model_name,
        model_max_length=config.model_max_length,
        padding_side="right",
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # noinspection PyTypeChecker
    model = AutoModelForCausalLM.from_pretrained(  # type: PreTrainedModel
        config.model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
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
    # Cast any bf16 params/buffers to fp16 (P100/GTX lack bf16 support)
    for param in model.parameters():
        if param.dtype == torch.bfloat16:
            param.data = param.data.to(torch.float16)
    for buf in model.buffers():
        if buf.dtype == torch.bfloat16:
            buf.data = buf.data.to(torch.float16)
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
    if config.select_sample is not None:
        rng_select = random.Random(config.seed)
        for task_name in samples:
            if len(samples[task_name]) > config.select_sample:
                samples[task_name] = rng_select.sample(samples[task_name], config.select_sample)
                logger.info("Subsampled %s to %d", task_name, config.select_sample)
    logger.info("Building HF dataset (%d tasks)", len(samples))
    train_ds = _build_hf_dataset(samples, tokenizer, config)
    logger.info("Training dataset: %d examples", len(train_ds))
    # Configure SFT
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    training_args = trl.SFTConfig(
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
        gradient_checkpointing=config.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if config.gradient_checkpointing else None,
        logging_steps=config.logging_steps,
        eval_strategy="no",
        save_strategy="steps",
        save_steps=config.save_steps,
        seed=config.seed,
        max_length=config.model_max_length,
        packing=config.packing,
        dataset_text_field="text",
        report_to="wandb" if config.use_wandb else "none",
        run_name=config.wandb_project,
    )
    trainer = trl.SFTTrainer(
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