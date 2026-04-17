"""Training Configuration dataclass"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class TrainConfig:
    """All hyperparameters for CLARIMOL training."""
    # Model
    model_name: str = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
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
    # Training
    batch_size: int = 8  # smaller than paper's 16 — P100 has 16GB vs A100's 80GB
    gradient_accumulation_steps: int = 2  # effective batch = 16
    learning_rate: float = 5e-4
    weight_decay: float = 0.1
    warmup_ratio: float = 0.01
    epochs: int = 1
    lr_scheduler: str = "cosine"
    max_grad_norm: float = 1.0
    bf16: bool = False  # Ada on work laptop does support bf16 but 8gb VRAM :(
    fp16: bool = True   # Tesla P100 cannot support bf16 but has 16gb VRAM :)
    # Generation (for eval during training)
    temperature: float = 0.2
    repetition_penalty: float = 1.0
    # Data
    task_names: list[str] = field(
        default_factory=lambda: [
            "functional_group",
            "ring_counting",
            "chain_length",
            "canonicalization",
            "fragment_assembly",
        ]
    )
    data_dir: str = "data/clarimol"
    select_sample: int | None = None
    test_val_sample: int = 10_000
    seed: int = 42
    # Output
    output_dir: str = "output/clarimol"
    logging_steps: int = 50
    eval_steps: int = 2000
    save_steps: int = 2000
    use_wandb: bool = True
    wandb_project: str = "clarimol"
    # Hardware
    gradient_checkpointing: bool = True  # saves VRAM on P100
    packing: bool = False  # pack multiple sequences per batch (big speedup)
    use_unsloth: bool = True