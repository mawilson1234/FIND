#!/bin/bash

#SBATCH --job-name=oddone-or-hierar-mirr-pcfg-lstm-attn-NL10
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

echo Running script: oddone_or_hierar_mirr_pcfg_lstm_attn_NL10.sh

python local_grid.py \
	--sweep=hyperparams/oddone-or-hierar-mirr-pcfg/lstm_attention_NL10.json \
	--task=tasks/oddone-or-hierar-mirr-pcfg/4/fpa/ \
	--n_workers=2