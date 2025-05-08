#!/bin/bash
#SBATCH --job-name="print-CUDA-toolkit-version"
#SBATCH --partition=ai
#SBATCH --gpus=1
#SBATCH --output=../slurm-out/%x.%j.out   # %x = job-name, %j = job‑ID
#SBATCH --error=../slurm-err/%x.%j.err

echo “Running on host $(hostname)”
echo “CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES”
nvidia-smi