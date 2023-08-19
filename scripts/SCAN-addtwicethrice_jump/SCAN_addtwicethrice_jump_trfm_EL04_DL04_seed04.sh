#!/bin/bash

#SBATCH --job-name=SCAN-addtwicethrice_jump-transformer-EL04_DL04-seed04
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

echo Running script: SCAN_addtwicethrice_jump_trfm_EL04_DL04_seed04.sh

python local_grid.py \
	--sweep=hyperparams/SCAN/transformer_EL04_DL04_seed04.json \
	--task=tasks/SCAN/addtwicethrice_jump/fpa/ \
	--n_workers=1
