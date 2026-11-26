#!/bin/bash
# Submit runs from a sweep file, skipping names already submitted (tracked in /scratch/$USER/tsga/submitted_<sweep>.txt)
# and stopping when the user's queue holds MAX jobs (QOS: 8 queued per user). Safe to re-run until all are submitted.
# Usage (repo root, login node): bash scripts/slurm/submit_sweep.sh scripts/slurm/sweep_datasets.txt
SWEEP=$1; MAX=${MAX:-8}; TIME=${TIME:-05:00:00}   # TSGA takes ~2.5 h on a V100 (OC-MAML ~1.5 h); 3 h timed out once
DONE=/scratch/$USER/tsga/submitted_$(basename "$SWEEP" .txt).txt; touch "$DONE"
grep -v '^#' "$SWEEP" | grep -v '^\s*$' | while IFS='|' read -r name args; do
  name=$(echo $name); args=$(echo $args)
  grep -q "^$name " "$DONE" && continue
  [ "$(squeue -u $USER -h | wc -l)" -ge "$MAX" ] && { echo "queue full; rerun later"; break; }
  out=$(sbatch -p gpu -t "$TIME" -J "$name" scripts/slurm/train.sbatch $args) || { echo "FAILED $name: $out"; break; }
  echo "$name ${out##* }" | tee -a "$DONE"
done
echo "submitted so far: $(wc -l < "$DONE") / $(grep -v '^#' "$SWEEP" | grep -cv '^\s*$')"
