#!/bin/bash

#SBATCH --job-name=compo2-cnn-EL02_DL10
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

echo Running script: compo2_cnn_EL02_DL10.sh

python local_grid.py \
	--sweep=hyperparams/compo2-24/cnn_EL02_DL10.json \
	--task=tasks/compo2/24/fpa/ \
	--n_workers=2