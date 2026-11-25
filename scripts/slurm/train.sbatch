#!/bin/bash
#SBATCH --job-name=tsga
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:1
#SBATCH --exclude=d1025,c2204,c2205,c2206,c2207
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/%u/tsga/logs/%x_%j.out
# Generic TSGA run; all args are Hydra overrides. Examples (from the repo root):
#   sbatch -J cifar-k10 scripts/slurm/train.sbatch dataset=CIFAR n_support=10
#   sbatch -J fc100-k10 scripts/slurm/train.sbatch dataset=FC n_support=10
#   sbatch -J cub-k10   scripts/slurm/train.sbatch dataset=CUB image_height=84 image_width=84 n_support=10
set -e
export TSGA_DATA=${TSGA_DATA:-/projects/ML_Small_Data/$USER/tsga/data}
export TSGA_OUT=${TSGA_OUT:-/projects/ML_Small_Data/$USER/tsga/outputs}
cd "$SLURM_SUBMIT_DIR"
nvidia-smi -L
echo "commit $(git rev-parse --short HEAD)  job $SLURM_JOB_ID  args: $*"
~/venvs/tsga/bin/python tsga.py "$@" hydra.run.dir="$TSGA_OUT/\${dataset}/K\${n_support}_${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
