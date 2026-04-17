#!/bin/bash
python -m clarimol prepare --keep-n 100000 --output-dir data/clarimol
python -m clarimol prepare --split validation --output-dir data/test --subsample random --no-curriculum
bash scripts/download_mol_instructions.sh

