#!/bin/bash

#SBATCH --job-name=compo2-cnn-EL02_DL10
#SBATCH --output=joblogs/%x_%j.txt
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=a5000:4
#SBATCH --partition=rhel8_gpu
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
	--sweep=hyperparams/SCAN/cnn_EL02_DL10.json \
	--task=tasks/SCAN/addtwicethricejump/fpa/ \
	--n_workers=4
