#!/bin/bash

#SBATCH --job-name=SCAN-simple_original-transformer-EL04_DL04-seed01
#SBATCH --output=joblogs/%x_%j.txt
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --time=01-00:00:00
#SBATCH --mem=20GB
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND

module unload Python
module load CUDA
module load cuDNN
module load miniconda
source activate inductive

echo Running script: SCAN_simple_original_trfm_EL04_DL04_seed01.sh

python local_grid.py \
	--sweep=hyperparams/SCAN/transformer_EL04_DL04_seed01.json \
	--task=tasks/SCAN/simple_original/fpa/ \
	--n_workers=1
