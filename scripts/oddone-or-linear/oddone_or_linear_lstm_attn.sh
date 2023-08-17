#!/bin/bash

#SBATCH --job-name=oddone-or-linear-lstm-attn
#SBATCH --output=joblogs/%x_%j.txt
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=v100:4
#SBATCH --partition=gpu
#SBATCH --time=03:30:00
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND

module unload Python
module load CUDA
module load cuDNN
module load miniconda
source activate inductive

echo Running script: oddone_or_linear_lstm_attn.sh

python local_grid.py \
	--sweep=hyperparams/oddone-or-linear/lstm_attention.json \
	--task=tasks/oddone-or-linear/4/fpa/ \
	--n_workers=4
