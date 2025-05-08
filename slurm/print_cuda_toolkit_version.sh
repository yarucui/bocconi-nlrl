#!/bin/bash
#SBATCH --job-name="Print CUDA toolkit version"
#SBATCH --partition=stud
#SBATCH --gpus=1
#SBATCH --output=../slurm-out
#SBATCH --error=../slurm-err

echo “Running on host $(hostname)”
echo “CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES”
nvidia-smi