## Training on DCC (H200 GPUs)

### Resources

- **Duke Compute Cluster (DCC)**: 1,360 nodes, 980 GPUs, SLURM scheduler
- **Target hardware**: NVIDIA H200 (141GB HBM3e, NVLink 900 GB/s)

> H200 enables full-precision training, massive batches, and multi-GPU parallelism.

### Workflow

#### SSH Access
Login to the DCC:
```bash
ssh netid@dcc-login.oit.duke.edu
```

Clone and Enter Repository:
```bash
git clone https://github.com/harrietobrien/CLARIMOL.git
cd CLARIMOL
```

#### Setup environment:
```bash
cd training_duke
bash setup_env.sh
cd ..
```

#### Generate training and test data:
```bash
python -m clarimol prepare --keep-n 100000 --output-dir data/clarimol
python -m clarimol prepare --split validation --output-dir data/test --subsample random --no-curriculum
bash scripts/download_mol_instructions.sh
```

#### Single GPU (`job_h200_single.slurm`)

Requests 1 H200 GPU and runs standard PyTorch training. The model and data flow through one GPU.

  - Batch size 64, no special distributed framework needed
  - Full 8B model in bf16 (~16GB); works easily w/ H200 141GB VRAM

```bash
sbatch job_h200_single.slurm
```

#### Multi GPU (`job_h200_multi.slurm`)

Requests 4 H200 GPUs connected via NVLink and uses DeepSpeed ZeRO Stage 2 to split the work.                                                                                                                          
   
  - Parallelism: Each GPU processes a different slice of each batch simultaneously                                                                                                                                  
  - ZeRO-2 shards the optimizer states (Adam momentum + variance) across the 4 GPUs, so each GPU only stores 1/4 of the optimizer memory
  - Gradients are synchronized via NVLink all-reduce (900 GB/s bidirectional)                                                                                                                                           
  - Uses `accelerate launch --num_processes 4` to coordinate the GPUs                                                                                                                                                                 
          
```bash
sbatch sbatch job_h200_multi.slurm
```

### Files
- `job_h200_single.slurm` — Single H200 GPU, full 250K dataset
- `job_h200_multi.slurm` — Multi-GPU (4x H200) with DeepSpeed ZeRO-2
- `job_h200_full_precision.slurm` — Full bf16 training without quantization
- `setup_env.sh` — Conda environment setup
- `requirements.txt` — Dependencies
- `deepspeed_config.json` — DeepSpeed ZeRO-2 config for multi-GPU

### Notes
- H200 has 141GB VRAM — no need for 4-bit quantization
- Can train full 8B model in bf16 with batch=64+
- `/work` has 75-day purge policy — copy results to `/cwork` after training
