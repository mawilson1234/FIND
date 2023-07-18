#!/bin/bash

#SBATCH --job-name=oddone-or-linear-transformer-lm
#SBATCH --output=joblogs/%x_%j.txt
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=v100:2
#SBATCH --partition=gpu
#SBATCH --time=06:00:00
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND

module unload Python
module load CUDA
module load cuDNN
module load miniconda
source activate inductive

echo Running script: oddone_or_linear_trfm_sh.sh

python local_grid.py \
	--sweep=hyperparams/oddone-or-linear/transformer_lm.json \
	--task=tasks/oddone-or-linear/4/fpa/ \
	--n_workers=2
