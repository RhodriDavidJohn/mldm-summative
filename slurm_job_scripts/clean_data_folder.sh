#!/bin/bash

#SBATCH --job-name=mldm_summative
#SBATCH --output=./slurm_logs/clean_data_output.txt
#SBATCH --error=./slurm_logs/clean_data_output.txt
#SBATCH --partition=teach_cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0:10:0
#SBATCH --mem=1M
#SBATCH --account=SSCM033324

date

echo 'Removing the data folder'
rm -r data
echo 'Finished removing the data folder'

date
