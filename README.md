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

**Pseudo-anomalies (our contribution):** `pseudo=cutpaste|shuffle|mix` (default `none`). The support set holds only normal
images, so the inner-loop loss is one-class. With `pseudo` set, each support image also yields a synthetic anomaly
(a colour-jittered pasted patch, or a 4x4 tile shuffle; see `pseudo.py`), labelled anomalous, and the inner loop adapts
on that two-class set: plain cross-entropy for OC-MAML, learned loss on the normals plus cross-entropy for TSGA.
Only the support set is used, so this is also valid at meta-test time. Sweep: `scripts/slurm/sweep_pseudo.txt`.

**AUROC:** test results include AUROC (anomaly score = predicted probability of "anomalous"). To add it to older runs
without retraining: `python evaluate.py <run_dir> ...` (or `sbatch scripts/slurm/eval.sbatch <run_dir> ...`), which writes
`eval_summary_statistics.csv` in each run folder.

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

### AUROC

Saved best checkpoints re-evaluated with `evaluate.py` on 600 fresh test tasks (so accuracy differs slightly from above).
TSGA leads OC-MAML in AUROC everywhere, by 0.2 (CIFAR-FS K=2) to 4.1 points (CUB K=10).

| Dataset | K | TSGA acc | TSGA AUROC | OC-MAML acc | OC-MAML AUROC |
|---|---|---|---|---|---|
| CIFAR-FS | 10 | 83.97 ± 0.84 | **90.26 ± 0.76** | 79.30 ± 0.95 | 86.88 ± 0.92 |
| CIFAR-FS | 2 | 76.48 ± 1.02 | **82.89 ± 1.02** | 75.73 ± 1.03 | 82.65 ± 1.06 |
| FC100 | 10 | 70.93 ± 1.01 | **76.72 ± 1.08** | 67.33 ± 0.99 | 73.72 ± 1.14 |
| FC100 | 2 | 65.15 ± 1.05 | **70.09 ± 1.19** | 63.29 ± 1.05 | 68.44 ± 1.20 |
| CUB | 10 | 85.73 ± 0.77 | **92.86 ± 0.56** | 81.28 ± 0.88 | 88.79 ± 0.82 |
| CUB | 2 | 81.47 ± 0.97 | **88.40 ± 0.87** | 77.46 ± 1.00 | 85.25 ± 0.94 |

### Pseudo-anomalies (contribution): a negative result

Accuracy / AUROC (%), ± 95% CI ≈ 1.0 for every entry. "none" = the re-evaluated baselines above.

| Inner-loop support | FC100 K=2 OC-MAML | CUB K=2 OC-MAML | FC100 K=2 TSGA | CUB K=2 TSGA |
|---|---|---|---|---|
| none (baseline) | 63.29 / 68.44 | 77.46 / 85.25 | 65.15 / 70.09 | 81.47 / 88.40 |
| `mix` | 63.44 / 68.33 | 75.47 / 83.14 | 63.69 / 67.44 | 81.53 / 88.73 |
| `cutpaste` | 64.22 / 69.23 | 75.13 / 81.78 | — | — |
| `shuffle` | 64.38 / 69.59 | 76.98 / 84.82 | — | — |
| `mix`, weight 0.25 | 63.02 / 67.96 | 78.00 / 85.18 | 64.00 / 69.53 | 80.90 / 87.82 |
| `mix`, meta-training only | 53.53 / 68.79 | 78.79 / 85.92 | 62.79 / 67.30 | 54.58 / 57.72 |

At K=10, `mix` was likewise neutral: FC100 TSGA 70.54 / 76.86, OC-MAML 67.42 / 74.26; CUB TSGA 85.38 / 92.70, OC-MAML 80.16 / 88.18.

Findings:
- **No variant reliably helps.** Every change is within about one CI of the baseline, except where it hurts.
- **`cutpaste` hurts on fine-grained CUB** (−3.5 AUROC for OC-MAML); `shuffle` is neutral. The network learns to spot paste
  artefacts, which says nothing about whether a bird is a different species. Meta-training accuracy with pseudo-anomalies
  rises to 85–92% while test accuracy does not move: a shortcut.
- **Down-weighting (0.25) moves results back towards the baseline**, as expected, without crossing above it.
- **Meta-training-only breaks the inner loop's train/test match.** The model is meta-learned to adapt on a two-class support,
  then adapts on a one-class support at test time. For FC100 OC-MAML, accuracy falls to 53.5 while AUROC is unchanged (68.8):
  the anomaly ranking survives but the decision threshold does not, which accuracy alone would have hidden. CUB TSGA
  collapses (best validation epoch 7).

Each run writes `train_summary_statistics.csv` and `test_summary_statistics.csv` (accuracy on 600 test tasks, with std and 95% CI) to its output folder.
