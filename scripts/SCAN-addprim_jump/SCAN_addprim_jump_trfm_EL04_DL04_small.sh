#!/bin/bash

#SBATCH --job-name=SCAN-addprim_jump-transformer-EL04_DL04-small
#SBATCH --output=joblogs/%x_%j.txt
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=a5000:1
#SBATCH --partition=rhel8_gpu
#SBATCH --time=14:00:00
#SBATCH --mem=20GB
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND

module unload Python
module load CUDA
module load cuDNN
module load miniconda
source activate inductive

echo Running script: SCAN_addprim_jump_trfm_EL04_DL04_small.sh

python local_grid.py \
	--sweep=hyperparams/SCAN/transformer_EL04_DL04_small.json \
	--task=tasks/SCAN/addprim_jump/fpa/ \
	--n_workers=1
