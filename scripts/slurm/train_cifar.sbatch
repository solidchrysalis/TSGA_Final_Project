#!/bin/bash
#SBATCH --job-name=tsga-cifar
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=/scratch/%u/tsga/runs/slurm-%x.out
# Full-length CIFAR-FS run with the conf/config.yaml defaults; extra args are Hydra overrides.
# Submit from the repo root: mkdir -p /scratch/$USER/tsga/runs && sbatch scripts/slurm/train_cifar.sbatch
source ~/.bashrc
module load miniconda3/25.9.1
source activate /scratch/$USER/tsga/envs/tsga
cd "$SLURM_SUBMIT_DIR"
nvidia-smi -L
python tsga.py dataset=CIFAR "$@"
