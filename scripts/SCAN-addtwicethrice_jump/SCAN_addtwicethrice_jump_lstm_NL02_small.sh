#!/bin/bash

#SBATCH --job-name=SCAN-addtwicethrice_jump-lstm-NL02-small
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

echo Running script: SCAN_addtwicethrice_jump_lstm_NL02_small.sh

python local_grid.py \
	--sweep=hyperparams/SCAN/lstm_noattention_NL02_small.json \
	--task=tasks/SCAN/addtwicethrice_jump/fpa/ \
	--n_workers=1
