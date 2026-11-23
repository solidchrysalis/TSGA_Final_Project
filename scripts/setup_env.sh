#!/bin/bash
# Build the tsga conda env (on a compute node). Env + pkg cache live on /scratch (regenerable).
set -e
module load miniconda3/25.9.1
export CONDA_PKGS_DIRS=/scratch/$USER/tsga/conda-pkgs
export PIP_CACHE_DIR=/scratch/$USER/tsga/pip-cache
mkdir -p $CONDA_PKGS_DIRS $PIP_CACHE_DIR
conda create -y -p /scratch/$USER/tsga/envs/tsga python=3.10
source activate /scratch/$USER/tsga/envs/tsga
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
pip install "hydra-core==1.3.2" omegaconf numpy pillow h5py tqdm datasets huggingface_hub pyyaml gdown ordered-set requests
echo SETUP_DONE
