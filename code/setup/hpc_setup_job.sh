#!/bin/bash

#SBATCH --account=SSCM033324
#SBATCH --job-name=ahds_summative_setup
#SBATCH --partition=teach_cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:10
#SBATCH --mem=1K
#SBATCH --output=logs/slurm/setup.out

echo "Setting up HCP environment and pipeline slurm config"

# create the log directory
mkdir -p logs/slurm

cd code/setup

# save the slurm config document to home directory
mkdir -p ~/.config/snakemake/mldm_slurm_profile

cp slurm_config.yaml ~/.config/snakemake/mldm_slurm_profile/config.yaml

echo "Finished setup!"
