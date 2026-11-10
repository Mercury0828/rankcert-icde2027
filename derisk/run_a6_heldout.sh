#!/usr/bin/env bash
# Held-out A6 evaluation after A6-FREEZE (derisk/A6_FREEZE.json): frozen R-PAIR-S6 vs BL1 and R-CP, reps 0-9,
# N = 1e5 first for every instance given, then N = 1e4, then N = 1e3. Rows -> ./data/a6/heldout/.
# usage: bash derisk/run_a6_heldout.sh <shard-name> <instance> [<instance> ...]
export PATH=/usr/bin:/bin:$PATH
export PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1
cd "$(dirname "$0")/.."
shard=$1; shift
LOG=./data/a6/heldout_$shard.log
for N in 100000 10000 1000; do
  for inst in "$@"; do
    python -m derisk.a6_run "$inst" --Ns $N --reps 0-9 --methods R-CP,BL1,R-PAIR-S6 --sub heldout >> "$LOG" 2>&1 \
      || echo "$inst $N FAILED" >> "$LOG"
  done
done
echo "shard $shard done" >> "$LOG"
