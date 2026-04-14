# CLARIMOL

Improved reproduction of [CLEANMOL](https://github.com/yunhuijang/CLEANMOL) (EMNLP 2025) — *Improving Chemical Understanding of LLMs via SMILES Parsing*.

CLARIMOL pre-trains LLMs on five deterministic SMILES parsing tasks to improve molecular structural understanding, then fine-tunes on downstream chemistry tasks.

> * Only LLaMA 3.1-8B-Instruct has been set up for (pre)training (default model is unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit); LLaMA 3.1-8B and Qwen2.5-7B to-be-implemented for LoRA SFT training as well.

## SMILES Parsing Tasks

| Task | Type | Input | Output |
|------|------|-------|--------|
| Functional group matching | Subgraph (binary) | SMILES + functional group | Yes / No |
| Ring counting | Subgraph (integer) | SMILES + ring size | Count |
| Chain length measurement | Subgraph (integer) | SMILES | Length |
| SMILES canonicalization | Global graph | Randomized SMILES | Canonical SMILES |
| Fragment assembly | Global graph | Two BRICS fragments | Complete SMILES |

## Workflow

```bash
# Install from build configuration file: pyproject.toml
pip install -e .

# Prepare Dataset from [ZINC250K](https://huggingface.co/datasets/yairschiff/zinc250k)
python -m clarimol prepare --output-dir data/clarimol

# Quick Test (for instance, 100 molecules)
python -m clarimol prepare --max-molecules 100 --output-dir data/dev

# Pre-train on parsing tasks
python -m clarimol train --data-dir data/clarimol --output-dir output/pretrain

# Evaluate
python -m clarimol evaluate --model-path output/pretrain/final --data-dir data/clarimol
```

## Project Structure

```
src/clarimol/
├── data/                        # Dataset construction
│   ├── sample.py                # Sample dataclass
│   ├── parsing.py               # Parsing abstract base class
│   ├── functional_group.py      # FunctionalGroup task
│   ├── ring_counting.py         # RingCounting task
│   ├── chain_length.py          # ChainLength task
│   ├── canonicalization.py      # Canonicalization task
│   ├── fragment_assembly.py     # FragmentAssembly task
│   ├── functional_groups.yaml   # SMARTS patterns (YAML config)
│   ├── tasks.py                 # Task registry
│   ├── dataset.py               # ZINC250K loading, generation, serialization
│   └── pruning.py               # Middle-difficulty pruning + curriculum ordering
├── tasks/
│   └── prompts.py               # Instruction paraphrases + chat-template builder
├── train/
│   ├── config.py                # TrainConfig (P100-tuned defaults)
│   └── trainer.py               # Model loading (Unsloth/HF+PEFT), SFTTrainer
├── eval/
│   ├── metrics.py               # Accuracy, validity, BLEU, Levenshtein, fingerprints
│   └── inference.py             # Batch inference pipeline
└── utils/
    ├── chem.py                  # SMILES helpers
    └── io.py                    # Logging setup
```

## Assumed Hardware

Currently tuned for NVIDIA Tesla P100 (16GB VRAM): batch_size=8, gradient_accumulation=2, fp16, gradient checkpointing, 4-bit quantization, but may be updated for adaptability.

