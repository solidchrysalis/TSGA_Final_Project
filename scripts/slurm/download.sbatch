#!/bin/bash
#SBATCH --job-name=tsga-data
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/%u/tsga/logs/%x_%j.out
# Download datasets into $TSGA_DATA and build the hdf5 files. Args: dataset names (cifarfs fc100 omniglot miniimagenet cub).
# Submit from the repo root: sbatch scripts/slurm/download.sbatch cifarfs fc100 cub
set -e
export TSGA_DATA=${TSGA_DATA:-/projects/ML_Small_Data/$USER/tsga/data}
cd "$SLURM_SUBMIT_DIR"
~/venvs/tsga/bin/python scripts/download_datasets.py --root "$TSGA_DATA" "$@"
ls -la "$TSGA_DATA"/*
