#!/bin/bash

#SBATCH --job-name=compo2-2funcs-lstm-attn-NL10
#SBATCH --output=joblogs/%x_%j.txt
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --time=01-00:00:00
#SBATCH --mem=20GB
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND

module unload Python
module load CUDA
module load cuDNN
module load miniconda
source activate inductive

echo Running script: compo2_2funcs_lstm_attn_NL10.sh

python local_grid.py \
	--sweep=hyperparams/compo2-2funcs-constant-prop/lstm_attention_NL10.json \
	--task=tasks/compo2-2funcs-constant-prop/1/fpa/ \
	--n_workers=4
