#!/bin/bash
#SBATCH --job-name="setup"
#SBATCH --partition=ai
#SBATCH --gpus=1
#SBATCH --output=../slurm-out/%x.%j.out   # %x = job-name, %j = job‑ID
#SBATCH --error=../slurm-err/%x.%j.err

module purge
module load nvidia/cuda-12.4.0
module load modules/miniconda3

eval "$(conda shell.bash hook)"

echo “Running on host $(hostname)”
echo “CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES”

conda env create -f environment.yml

module purge