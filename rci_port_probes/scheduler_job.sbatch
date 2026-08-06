#!/bin/bash
# Run the ratsim scheduler for one experiment inside a single SLURM job.
#
#   sbatch scheduler_job.sbatch <exp>              [extra scheduler_run.py args]
#   sbatch -p gpulong --time=2-00:00:00 scheduler_job.sbatch ortho_wells_adaptive
#
# One job hosts the scheduler, which then dispatches its runs as child processes
# inside this allocation. That is deliberate over a SLURM job array: the
# scheduler already owns port windows, resume-from-.done markers and RAM kills,
# and an array would need all of that duplicated in sbatch bookkeeping.
#
# The flip side is that the whole experiment shares one wall-clock limit, so
# pick the partition to fit the *experiment*, not one run:
#   gpufast 4h (smoke tests only) | gpu 1d | gpulong 3d | gpuextralong 21d
#
# --cpus-per-task/--mem here MUST match `resources` in
# scheduler/machines/rci.yaml (cpu_slot: 16, gpu: 1). See that file for why 16.
#SBATCH --job-name=ratsim-sched
#SBATCH --partition=gpufast
#SBATCH --gres=gpu:1
#SBATCH --time=03:30:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=/mnt/personal/%u/logs/sched-%j.out
set -uo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: sbatch $0 <exp> [scheduler_run.py args...]" >&2
  exit 2
fi

. "${RCI_ENV:-$HOME/rci_env.sh}"      # modules, paths, per-job TMPDIR, *_PYTHON_PATH
ratsim_activate

export RATSIM_SCHEDULER_MACHINE=rci
export RATSIM_XVFB=1
# No RATSIM_RUNDIR / base_port: the launcher derives both from $SLURM_JOB_ID.

# Per-RUN thread budget, not per-job. rci_env.sh defaults these to the whole
# allocation, which is right for one training but catastrophic here: N children
# would each take all 16 cores. Must match needs.cpu_slot in machines/rci.yaml.
# `needs.cpu_slot` is only scheduler bookkeeping -- it does not constrain the
# child's thread count, so this is the half that actually enforces it.
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

echo "=== $(date -Is) scheduler job $SLURM_JOB_ID on $(hostname) ==="
echo "cpus: ${SLURM_CPUS_PER_TASK:-?}   machine cfg: $RATSIM_SCHEDULER_MACHINE"
echo "ppo python:     $PPO_PYTHON_PATH"
echo "dreamer python: $DREAMER_PYTHON_PATH"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || echo "no GPU"
echo

cd "$RATSIM_GIT_DIR/meta_ratsim/ratsim_experiments"
rc=0
# -u matters: without it the scheduler's progress sits in a pipe buffer and the
# SLURM log looks dead for minutes at a time, which is the main way you watch a
# long run (`tail -f`). The scheduler already passes -u to its own children.
python -u scheduler_run.py "$@" || rc=$?

# The scheduler kills its own children, but not if it was scancel'd or died hard.
"$RATSIM_GIT_DIR/meta_ratsim/ratsim/scripts/stop_ratsim_headless.sh" --all || true

echo "=== $(date -Is) scheduler_run.py exit $rc ==="
echo "Monitor from login2 (NOT login1 -- glibc 2.17 there):"
echo "  . ~/rci_env.sh && ratsim_activate && cd \$RATSIM_GIT_DIR/meta_ratsim/ratsim_experiments"
echo "  python scheduler_status.py $1 --watch 5"
exit $rc
