#!/bin/bash

#SBATCH --job-name=oddone-or-linear-pcfg-bilstm-attn
#SBATCH --output=joblogs/%x_%j.txt
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=v100:2
#SBATCH --partition=gpu
#SBATCH --time=01-00:00:00
#SBATCH --mem=20GB
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND

module unload Python
module load CUDA
module load cuDNN
module load miniconda
source activate inductive

echo Running script: oddone_or_linear_pcfg_bilstm_attn.sh

python local_grid.py \
	--sweep=hyperparams/oddone-or-linear-pcfg/bilstm_attention.json \
	--task=tasks/oddone-or-linear-pcfg/4/fpa/ \
	--n_workers=2
