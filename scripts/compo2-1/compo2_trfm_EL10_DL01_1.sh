#!/bin/bash

#SBATCH --job-name=compo2-transformer-EL10_DL01_1
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

echo Running script: compo2_trfm_EL10_DL01_1.sh

python local_grid.py \
	--sweep=hyperparams/compo2-1/transformer_EL10_DL01.json \
	--task=tasks/compo2/1/fpa/ \
	--n_workers=2
