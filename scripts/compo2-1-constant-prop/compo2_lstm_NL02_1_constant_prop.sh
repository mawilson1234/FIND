#!/bin/bash

#SBATCH --job-name=compo2-lstm-NL02
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

echo Running script: compo2_lstm_NL02.sh

python local_grid.py \
	--sweep=hyperparams/compo2-constant-prop/lstm_noattention_NL02.json \
	--task=tasks/compo2-constant-prop/1/fpa/ \
	--n_workers=4
