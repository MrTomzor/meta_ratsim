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
#   amdfast 4h (smoke tests only) | amd 1d | amdlong 3d | amdextralong 21d
#
# --cpus-per-task/--mem/--gres here MUST match `resources` in the machine
# config you run against. The default below is the CPU shape:
#   scheduler/machines/rci.yaml — cpu_slot: 112 = 7 concurrent PPO runs x 16
#   threads. See that file for the measurements behind 16.
#
# The AMD nodes are the default because they are the *biggest* (126 allocatable
# threads vs 46/70 on the Intel ones), not because they are faster — per-run
# speed is within 5%. Intel-built venvs run there unmodified.
#
# ⚠️ `amd*` maps to a[01-16], which have NO GPUs — adding --gres here would be
# rejected outright. For dreamer / recurrent_ppo, switch node type AND machine
# config together:
#
#   sbatch -p amdgpu --gres=gpu:1 --cpus-per-task=31 --mem=100G \
#          scheduler_job.sbatch <exp> --machine rci_gpu
#
# 31 rather than 112 because a GPU node rations threads per GPU (124/4 on
# g[01-10]); taking more would strand A100s nobody else can schedule.
#SBATCH --job-name=ratsim-sched
#SBATCH --partition=amdfast
#SBATCH --time=03:30:00
#SBATCH --cpus-per-task=112
#SBATCH --mem=200G
#SBATCH --output=/mnt/personal/%u/logs/sched-%j.out
set -uo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: sbatch $0 <exp> [scheduler_run.py args...]" >&2
  exit 2
fi

. "${RCI_ENV:-$HOME/rci_env.sh}"      # modules, paths, per-job TMPDIR, *_PYTHON_PATH
ratsim_activate

# Overridable so the GPU variant can be selected at submit time:
#   RATSIM_SCHEDULER_MACHINE=rci_gpu sbatch -p amdgpu --gres=gpu:1 ... <exp>
# (a `--machine` arg after <exp> also wins — scheduler_run.py prefers the flag.)
export RATSIM_SCHEDULER_MACHINE="${RATSIM_SCHEDULER_MACHINE:-rci}"
export RATSIM_XVFB=1
# No RATSIM_RUNDIR / base_port: the launcher derives both from $SLURM_JOB_ID.

# BLAS/OpenMP threads are per-PROCESS, so this is a per-run budget and must stay
# small: with 7 children each would otherwise size its pool from the node's 128
# CPUs. 4 is the measured optimum for PPO's optimize phase, not a fraction of
# the allocation -- see the long comment in rci_env.sh, which sets the same
# default. This is separate from needs.cpu_slot in the machine config: that is
# only scheduler bookkeeping and does not constrain the child's thread count.
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
