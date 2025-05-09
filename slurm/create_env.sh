#!/bin/bash
#SBATCH --job-name="create_env"
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

# Create the environment
conda env create -f environment.yml

# Activate the environment
conda activate nlrl

# Show that we are using the right environment
conda info --envs

# Show which python is being used
echo "which python"
which python

# Check bitsandbytes install
echo "python -m bitsandbytes"
python -m bitsandbytes

# Check bitsandbytes install location
python -c "import bitsandbytes; print(bitsandbytes.__file__)"

module purge