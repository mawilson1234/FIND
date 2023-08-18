#!/bin/bash

#SBATCH --job-name=SCAN-addtwicethrice_jump-lstm-attn-NL02-seed09
#SBATCH --output=joblogs/%x_%j.txt
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --time=14:00:00
#SBATCH --mem=20GB
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND

module unload Python
module load CUDA
module load cuDNN
module load miniconda
source activate inductive

echo Running script: SCAN_addtwicethrice_jump_lstm_attn_NL02_seed09.sh

python local_grid.py \
	--sweep=hyperparams/SCAN/lstm_attention_NL02_seed09.json \
	--task=tasks/SCAN/addtwicethrice_jump/fpa/ \
	--n_workers=1
