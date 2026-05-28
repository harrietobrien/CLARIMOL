# CLARIMOL

**A Systematic Study of SMILES Pre-Training Objectives for Molecular Language Models**

CLARIMOL provides a comprehensive toolkit for pre-training LLMs on deterministic SMILES parsing tasks to improve molecular structural understanding. The package supports multi-model training, systematic ablation studies, cross-domain evaluation, and downstream transfer to molecular generation tasks.

Submitted to EMNLP 2026 via ACL Rolling Review (May 2026 cycle).

## Key Results

- **Five LLM families** evaluated: LLaMA-8B, Mistral-7B, OLMo-7B, Qwen2.5-7B, Qwen3-8B
- All models exceed **0.88 mean accuracy** after a single epoch of LoRA fine-tuning
- **Mistral-7B** achieves the highest mean accuracy (0.941); optimal LoRA configuration reaches **0.948**
- Effective learning rate ($\text{lr} \times \alpha/r$) identified as the primary control parameter
- Multi-task training yields up to **+25.5 pp** gains over single-task training
- Cross-domain transfer evaluated on COD crystal structures (82K) and QM9 small molecules
- Downstream transfer transforms zero-capability baselines into functional molecular generation

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

## Project Structure

```
src/clarimol/
├── data/                        # Dataset construction
│   ├── sample.py                # Sample dataclass
│   ├── parsing.py               # Parsing abstract base class
│   ├── functional_group.py      # FG task (30 SMARTS patterns)
│   ├── ring_counting.py         # RC task
│   ├── chain_length.py          # CL task
│   ├── canonicalization.py      # CA task
│   ├── fragment_assembly.py     # FA task (BRICS fragmentation)
│   ├── atom_degree.py           # Atom degree sequence (experimental)
│   ├── stereocenter.py          # Stereocenter CIP assignment (experimental)
│   ├── topological_distance.py  # Topological distance (experimental)
│   ├── functional_groups.yaml   # SMARTS pattern definitions
│   ├── tasks.py                 # Task registry
│   ├── dataset.py               # ZINC250K / COD / QM9 / ChEMBL loading
│   ├── pruning.py               # Middle-difficulty pruning + curriculum ordering
│   ├── cod.py                   # COD API integration
│   └── downstream.py            # Mol-Instructions data loader
├── tasks/
│   ├── prompts.py               # Instruction paraphrases + chat-template builder
│   ├── instructions.yaml        # Task instruction templates
│   └── system_prompts.yaml      # System prompts per task
├── train/
│   ├── config.py                # TrainConfig with hardware-adaptive defaults
│   ├── trainer.py               # Model loading (HF+PEFT), SFTTrainer
│   └── downstream.py            # Downstream fine-tuning driver
├── eval/
│   ├── metrics.py               # Accuracy, validity, BLEU, per-sample analysis
│   ├── inference.py             # Batch inference + per-sample prediction export
│   └── downstream.py            # Downstream evaluation
└── utils/
    ├── chem.py                  # SMILES helpers
    └── io.py                    # Logging setup

figures/                         # Plotting scripts for all paper figures
scripts/                         # SLURM submission scripts
training_duke/                   # HPC cluster job scripts
```

## Supported Models

| Model | HuggingFace ID | Parameters |
|-------|---------------|------------|
| LLaMA-8B | `meta-llama/Llama-3.1-8B-Instruct` | 8.0B |
| Mistral-7B | `mistralai/Mistral-7B-Instruct-v0.3` | 7.2B |
| OLMo-7B | `allenai/OLMo-7B-Instruct` | 6.9B |
| Qwen2.5-7B | `Qwen/Qwen2.5-7B-Instruct` | 7.6B |
| Qwen3-8B | `Qwen/Qwen3-8B` | 8.2B |

## Ablation Studies

The following ablations are supported and have been evaluated:

- **Learning rate:** 1e-4, 2e-4, 5e-4 (across LLaMA, Mistral, OLMo)
- **LoRA configuration:** rank {16, 32, 64, 128} x alpha {16, 32, 64, 128, 256}
- **Dataset size:** 10K, 25K, 50K, 100K, 250K samples
- **Training epochs:** 1, 2, 3
- **Curriculum ordering:** easy-hard, hard-easy, random, none (null result)
- **Single-task vs. multi-task:** per-task isolation + leave-one-out
- **Cross-domain:** ZINC250K, COD (1K and 82K), QM9, ChEMBL

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
