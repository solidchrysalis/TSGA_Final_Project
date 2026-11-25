#!/bin/bash
#SBATCH --job-name=tsga-setup
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/%u/tsga/logs/%x_%j.out
# Explorer: build the tsga uv venv in ~/venvs/tsga (installs must run in a job, not on the login node).
# Submit from the repo root: mkdir -p /scratch/$USER/tsga/logs && sbatch scripts/setup_env_uv.sh
set -e
UV=~/.local/bin/uv
$UV venv ~/venvs/tsga --python 3.10
$UV pip install --python ~/venvs/tsga/bin/python torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
$UV pip install --python ~/venvs/tsga/bin/python "hydra-core==1.3.2" omegaconf "numpy<2" pillow h5py tqdm pyyaml gdown ordered-set requests
~/venvs/tsga/bin/python -c "import torch, torchvision, hydra, h5py; print('torch', torch.__version__, 'tv', torchvision.__version__)"
echo SETUP_DONE
