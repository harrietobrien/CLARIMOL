## Training on NRP JupyterHub (RTX 2080 Ti)

### Environment
- **Platform**: NRP Nautilus JupyterHub (`ecu-nlp-202630.nrp-nautilus.io`)
- **Image**: `quay.io/jupyter/pytorch-notebook:cuda12-2024-04-22`
- **GPU**: NVIDIA GeForce RTX 2080 Ti (11GB VRAM)
- **CUDA**: 13.1

### Workflow

#### Open terminal in JupyterHub

Clone and Enter Repository:
```bash
git clone https://github.com/harrietobrien/CLARIMOL.git
cd CLARIMOL
```

#### Setup environment:
```bash
cd training_nrp
bash setup_env.sh
cd ..
```

#### Generate training and test data:
```bash
python -m clarimol prepare --output-dir data/clarimol
python -m clarimol prepare --split validation --output-dir data/test --max-molecules 20000 --subsample random --no-curriculum
bash scripts/download_mol_instructions.sh
```

#### Run Training Script
```bash
bash train.sh
```

#### Alternatively, use the notebook:

Open `train_nrp.ipynb` and run all cells.

### Files
- `train.sh` — Optimized training script for RTX 2080 Ti
- `train_nrp.ipynb` — Jupyter notebook version
- `setup_env.sh` — Install dependencies into JupyterHub environment
- `requirements.txt` — Dependencies

### Notes
- RTX 2080 Ti has 11GB VRAM
- Turing architecture: fp16 tensor cores, no bf16
- Batch size must be smaller: batch=8 with 512 max_length
- 4-bit quantization required for 8B models
