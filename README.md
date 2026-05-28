# CLARIMOL

**A Systematic Study of SMILES Pre-Training Objectives for Molecular Language Models**

CLARIMOL provides a comprehensive toolkit for pre-training LLMs on deterministic SMILES parsing tasks to improve molecular structural understanding. The package supports multi-model training, systematic ablation studies, cross-domain evaluation, and downstream transfer to molecular generation tasks.

## SMILES Parsing Tasks

| Task | Abbrev. | Type | Input | Output |
|------|---------|------|-------|--------|
| Functional Group | FG | Binary classification | SMILES + SMARTS pattern | Yes / No |
| Ring Counting | RC | Integer | SMILES + ring size | Count |
| Chain Length | CL | Integer | SMILES | Length |
| Canonicalization | CA | String generation | Randomized SMILES | Canonical SMILES |
| Fragment Assembly | FA | String generation | Two BRICS fragments | Complete SMILES |

## Quick Start

```bash
# Install
pip install -e .

# Prepare training data (250K samples from ZINC250K)
python -m clarimol prepare --output-dir data/clarimol

# Prepare test data
python -m clarimol prepare --split validation --output-dir data/test \
    --max-molecules 10000 --subsample random --no-curriculum

# Train (LoRA SFT, single GPU)
python -m clarimol train \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --data-dir data/clarimol \
    --output-dir output/pretrain \
    --lr 1e-4 --epochs 1 --lora-r 64 --lora-alpha 16 \
    --bf16 --no-4bit --batch-size 16

# Evaluate
python -m clarimol evaluate \
    --model-path output/pretrain/final \
    --data-dir data/test \
    --output-file output/pretrain/results.json \
    --no-unsloth

# Domain shift (COD crystal structures)
python -m clarimol prepare --source cod --max-molecules 1000 --output-dir data/cod
python -m clarimol evaluate --model-path output/pretrain/final --data-dir data/cod

# Downstream transfer (retrosynthesis, forward reaction, reagent prediction)
python -m clarimol downstream-train --model output/pretrain/final \
    --task retrosynthesis --data-dir data/mol_instructions
```

## Supported Models

| Model | HuggingFace ID | Parameters |
|-------|---------------|------------|
| LLaMA-8B | `meta-llama/Llama-3.1-8B-Instruct` | 8.0B |
| Mistral-7B | `mistralai/Mistral-7B-Instruct-v0.3` | 7.2B |
| OLMo-7B | `allenai/OLMo-7B-Instruct` | 6.9B |
| Qwen2.5-7B | `Qwen/Qwen2.5-7B-Instruct` | 7.6B |
| Qwen3-8B | `Qwen/Qwen3-8B` | 8.2B |

## Data Sources

| Source | Molecules | Usage |
|--------|-----------|-------|
| [ZINC250K](https://huggingface.co/datasets/yairschiff/zinc250k) | 250K | Training + test |
| [COD](https://www.crystallography.net) | 82,916 | Domain shift |
| [QM9](https://figshare.com/collections/Quantum_chemistry_structures_and_properties_of_134_kilo_molecules/978904) | 133,885 | Domain shift |
| [ChEMBL](https://www.ebi.ac.uk/chembl/) | 2.4M | Domain shift (zero-shot) |
| [Mol-Instructions](https://github.com/zjunlp/Mol-Instructions) | ~100K | Downstream transfer |

## License

MIT
