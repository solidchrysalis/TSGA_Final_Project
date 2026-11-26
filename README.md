# TSGA

Reimplementation of *Task-Specific Gradient Adaptation for Few-Shot One-Class Classification* (Li et al., CVPR 2025),
built on the authors' code (github.com/noragami2333/TSGA), extended with two datasets the paper does not use: **FC100** and **CUB-200-2011**.

## Datasets

| `dataset=` | Data | Image size overrides | In paper? |
|---|---|---|---|
| `CIFAR` | CIFAR-FS (64/16/20 classes) | defaults (32x32, 3 ch) | yes |
| `MIN` | miniImageNet (64/16/20) | `image_height=84 image_width=84` | yes |
| `OMN` | Omniglot (Vinyals split; the paper uses 25/5/20 alphabets) | `image_channels=1 image_height=28 image_width=28` | yes |
| `FC` | FC100 (CIFAR-100 split by superclass, 60/20/20 classes) | defaults (32x32) | **no** |
| `CUB` | CUB-200-2011 birds (100/50/50 species) | `image_height=84 image_width=84` | **no** |

## Setup (Explorer cluster)

```bash
mkdir -p /scratch/$USER/tsga/logs
sbatch scripts/setup_env_uv.sh                                    # uv venv at ~/venvs/tsga
sbatch scripts/slurm/download.sbatch cifarfs fc100 cub            # -> /projects/ML_Small_Data/$USER/tsga/data
```

Set `TSGA_DATA` / `TSGA_OUT` to use other data/output roots. `scripts/setup_env.sh` is the original conda alternative.

## Run

All arguments are Hydra overrides of `conf/config.yaml`:

```bash
python tsga.py dataset=CIFAR n_support=10
python tsga.py dataset=CUB image_height=84 image_width=84 n_support=2
sbatch -J fc100-k10 scripts/slurm/train.sbatch dataset=FC n_support=10    # on Explorer
```

Ablations: `loss=false` (no loss network), `adapter=false` (no adapter); `loss=false adapter=false mamlpp=false` is the OC-MAML baseline.

Full sweep (TSGA vs OC-MAML on CIFAR-FS, FC100, CUB, K=2/10): `bash scripts/slurm/submit_sweep.sh scripts/slurm/sweep_datasets.txt` (re-run until all are submitted).

## Results

Test accuracy (%) on 600 tasks, best-validation checkpoint, ± 95% CI. One run each (the paper averages 5 restarts).

| Dataset | K | TSGA | OC-MAML | Paper TSGA / OC-MAML |
|---|---|---|---|---|
| CIFAR-FS | 10 | **83.07 ± 0.85** | 78.82 ± 0.96 | 81.70 / 79.1 |
| CIFAR-FS | 2 | **75.91 ± 1.03** | 74.88 ± 1.03 | 76.84 / 70.0 |
| FC100 | 10 | **70.97 ± 1.06** | 67.93 ± 0.99 | — |
| FC100 | 2 | **65.51 ± 1.07** | 63.22 ± 1.01 | — |
| CUB | 10 | **85.83 ± 0.75** | 82.49 ± 0.86 | — |
| CUB | 2 | **82.01 ± 0.93** | 77.50 ± 1.05 | — |

Each run writes `train_summary_statistics.csv` and `test_summary_statistics.csv` (accuracy on 600 test tasks, with std and 95% CI) to its output folder.
