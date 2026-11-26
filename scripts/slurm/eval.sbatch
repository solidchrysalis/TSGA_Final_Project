#!/bin/bash
#SBATCH --job-name=tsga-eval
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:1
#SBATCH --exclude=d1025,c2204,c2205,c2206,c2207
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/%u/tsga/logs/%x_%j.out
# Re-evaluate saved runs (accuracy + AUROC on 600 fresh test tasks). Args: run directories.
#   sbatch scripts/slurm/eval.sbatch /projects/ML_Small_Data/$USER/tsga/outputs/CUB/K10_cub-tsga-k10_12126125
set -e
export TSGA_DATA=${TSGA_DATA:-/projects/ML_Small_Data/$USER/tsga/data}
cd "$SLURM_SUBMIT_DIR"
nvidia-smi -L
echo "commit $(git rev-parse --short HEAD)  job $SLURM_JOB_ID  runs: $#"
~/venvs/tsga/bin/python evaluate.py "$@"
