#!/bin/bash
#SBATCH --job-name="build_bitsandbytes"
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

conda activate nlrl

cd $HOME
git clone https://github.com/bitsandbytes-foundation/bitsandbytes.git && cd bitsandbytes/
cmake -DCOMPUTE_BACKEND=cuda -S .
make
pip install -e .

python -m bitsandbytes

module purge