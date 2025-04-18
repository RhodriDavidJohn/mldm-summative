#!/bin/bash

#SBATCH --job-name=mldm_summative
#SBATCH --output=./slurm_logs/output.txt
#SBATCH --error=./slurm_logs/output.txt
#SBATCH --partition=teach_cpu
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --time=2:0:0
#SBATCH --mem=1G
#SBATCH --account=SSCM033324

mkdir -p slurm_logs

echo 'Activating conda environment'
source /user/work/hv23625/miniforge3/etc/profile.d/conda.sh
conda deactivate
conda activate mldm-env

echo 'Running python script'
python main.py

echo 'Slurm job finished'
