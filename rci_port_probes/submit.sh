#!/bin/bash
# Submit an experiment to SLURM. The friendly front door for submit.py.
#
#   ./submit.sh gps_ablation_5house --time 1d
#   ./submit.sh method_compare --time 4h --mode bfs
#   ./submit.sh dreamer_ladder --time 3d --variations consec4   # one ladder cell
#   ./submit.sh dreamer_ladder --time 3d --dry-run
#
# You pick the experiment and the wall clock. Partition, --cpus-per-task,
# --gres, --mem and any split across CPU/GPU jobs are derived from the def and
# scheduler/machines/*.yaml -- see submit.py for the reasoning.
#
# Resume is the default: re-submit the same line and finished stages are
# skipped. `--restart` is the destructive opposite.
#
# ⚠️ Run this from login2/3/4, NOT login1. login1 is CentOS 7 / glibc 2.17 and
# cannot run the module-provided Python the venv symlinks (RCI_CLUSTER_PORT.md
# phase 3 #11) -- you get a misleading "libpython3.12.so.1.0: cannot open
# shared object file".
set -euo pipefail

. "${RCI_ENV:-$HOME/rci_env.sh}"
ratsim_activate

cd "$RATSIM_GIT_DIR/meta_ratsim/ratsim_experiments"
exec python submit.py "$@"
