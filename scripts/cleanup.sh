#!/bin/bash

#SBATCH --job-name=FIND-cleanup
#SBATCH --output=joblogs/%x_%j.txt
#SBATCH --time=00:30:00
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND

module unload Python
module load miniconda

conda activate inductive

echo "Running script scripts/cleanup.sh"
echo ""

python cleanup.py
