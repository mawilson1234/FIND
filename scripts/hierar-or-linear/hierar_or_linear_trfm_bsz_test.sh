#!/bin/bash

#SBATCH --job-name=hierar-or-linear-transformer-bsz-test
#SBATCH --output=joblogs/%x_%j.txt
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=v100:1
#SBATCH --partition=gpu
#SBATCH --time=03:30:00
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND

module unload Python
module load CUDA
module load cuDNN
module load miniconda
source activate inductive

echo Running script: hierar_or_linear_trfm_bsz_test.sh

python local_grid.py \
	--sweep=hyperparams/hierar-or-linear/transformer_small_bsz_test.json \
	--task=tasks/hierar-or-linear/4/fpa/ \
	--n_workers=1
