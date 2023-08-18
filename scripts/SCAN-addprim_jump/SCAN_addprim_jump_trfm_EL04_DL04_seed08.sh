#!/bin/bash

#SBATCH --job-name=SCAN-addprim_jump-transformer-EL04_DL04-seed08
#SBATCH --output=joblogs/%x_%j.txt
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --time=14:00:00
#SBATCH --mem=20GB
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND

module unload Python
module load CUDA
module load cuDNN
module load miniconda
source activate inductive

echo Running script: SCAN_addprim_jump_trfm_EL04_DL04_seed08.sh

python local_grid.py \
	--sweep=hyperparams/SCAN/transformer_EL04_DL04_seed08.json \
	--task=tasks/SCAN/addprim_jump/fpa/ \
	--n_workers=1
