"""
Stage 2: Fine-tune a (pre-trained) model on downstream molecular generation tasks.

This follows the two-stage training described in the CLEANMOL paper:
    1. Pre-train on SMILES parsing (train/trainer.py)
    2. Fine-tune on downstream tasks (this module)

Uses the same LoRA SFT approach with hyperparameters from Table 6 of the paper.
"""

from __future__ import annotations
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, PreTrainedModel, PreTrainedTokenizerBase
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
import torch
from trl import SFTTrainer, SFTConfig
from clarimol.data.downstream import (
    DownstreamSample,
    load_mol_instructions,
    build_downstream_messages,
)

logger = logging.getLogger(__name__)


@dataclass
class DownstreamConfig:
    """Hyperparameters for downstream fine-tuning (Table 6 of paper)."""
    # Model — can be a pre-trained CLARIMOL checkpoint or a base model
    model_name: str = "output/parsing_pretrain/final"
    model_max_length: int = 2048
    load_in_4bit: bool = True
    # LoRA
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )
    # Training (Table 6)
    batch_size: int = 8
    gradient_accumulation_steps: int = 2  # effective batch = 16
    learning_rate: float = 5e-4
    weight_decay: float = 0.1
    warmup_ratio: float = 0.01
    epochs: int = 1
    lr_scheduler: str = "cosine"
    max_grad_norm: float = 1.0
    bf16: bool = False
    fp16: bool = True
    temperature: float = 0.2
    repetition_penalty: float = 1.0
    # Data
    data_dir: str = "data/mol_instructions"
    task: str = "retrosynthesis"  # single task per fine-tune run
    seed: int = 42
    # Output
    output_dir: str = "output/downstream"
    logging_steps: int = 50
    save_steps: int = 2000
    use_wandb: bool = True
    wandb_project: str = "clarimol-downstream"
    # Hardware
    gradient_checkpointing: bool = True
    use_unsloth: bool = True


def _load_model_tokenizer(config: DownstreamConfig):
    """
    Load model + tokenizer for downstream fine-tuning.

    If model_name points to a CLARIMOL pre-trained checkpoint (a PEFT model),
    we merge the adapter first, then apply a fresh LoRA for downstream training.
    If it points to a base model, we just apply LoRA directly.
    """
    if config.use_unsloth:
        try:
            from unsloth import FastLanguageModel, FastLlamaModel
            tokenizer: PreTrainedTokenizerBase
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

    # Check if model_name is a PEFT checkpoint (has adapter_config.json)
    model_path = Path(config.model_name)
    is_peft_checkpoint = (model_path / "adapter_config.json").exists()

    if is_peft_checkpoint:
        logger.info("Loading pre-trained PEFT model from %s", config.model_name)
        # Load the base model through the adapter, then merge
        # noinspection PyTypeChecker
        model = AutoModelForCausalLM.from_pretrained(  # type: PreTrainedModel
            config.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        # noinspection PyTypeChecker
        tokenizer = AutoTokenizer.from_pretrained(  # type: PreTrainedTokenizerBase
            config.model_name,
            model_max_length=config.model_max_length,
            padding_side="right",
            trust_remote_code=True,
        )
    else:
        logger.info("Loading base model from %s", config.model_name)
        # noinspection PyTypeChecker
        tokenizer = AutoTokenizer.from_pretrained(  # type: PreTrainedTokenizerBase
            config.model_name,
            model_max_length=config.model_max_length,
            padding_side="right",
            trust_remote_code=True,
        )
        # noinspection PyTypeChecker
        model = AutoModelForCausalLM.from_pretrained(  # type: PreTrainedModel
            config.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if config.load_in_4bit:
        model = prepare_model_for_kbit_training(model)
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Apply fresh LoRA for downstream
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
    samples: list[DownstreamSample],
    tokenizer,
    config: DownstreamConfig,
) -> Dataset:
    """Convert downstream samples into a HuggingFace Dataset."""
    rng = random.Random(config.seed)
    all_texts: list[str] = []
    for sample in samples:
        messages = build_downstream_messages(sample, use_system_prompt=True)
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        except Exception:
            text = ""
            for msg in messages:
                text += f"<|{msg['role']}|>\n{msg['content']}\n"
        all_texts.append(text)
    rng.shuffle(all_texts)
    return Dataset.from_dict({"text": all_texts})


def run_downstream_training(config: DownstreamConfig) -> str:
    """
    Fine-tune on a single downstream task.

    :return: Path to saved model
    """
    logger.info("Loading model: %s", config.model_name)
    model, tokenizer = _load_model_tokenizer(config)

    logger.info("Loading %s training data from %s", config.task, config.data_dir)
    all_data = load_mol_instructions(
        config.data_dir,
        tasks=[config.task],
        split="train",
    )
    samples = all_data.get(config.task, [])
    if not samples:
        raise ValueError(f"No training samples found for task: {config.task}")

    logger.info("Building HF dataset (%d samples)", len(samples))
    train_ds = _build_hf_dataset(samples, tokenizer, config)

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
        run_name=f"{config.wandb_project}-{config.task}",
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        args=training_args,
    )

    logger.info("Starting downstream fine-tuning on %s...", config.task)
    trainer.train()

    final_path = str(output_dir / "final")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    logger.info("Model saved to %s", final_path)
    return final_path
