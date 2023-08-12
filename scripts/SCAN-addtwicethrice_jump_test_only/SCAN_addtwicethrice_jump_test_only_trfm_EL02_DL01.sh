#!/bin/bash

#SBATCH --job-name=SCAN-addtwicethrice_jump_test_only-transformer-EL02_DL01
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

echo Running script: SCAN_addtwicethrice_jump_test_only_trfm_EL02_DL01.sh

python local_grid.py \
	--sweep=hyperparams/SCAN/transformer_EL02_DL01.json \
	--task=tasks/SCAN/addtwicethrice_jump_test_only/fpa/ \
	--n_workers=4
