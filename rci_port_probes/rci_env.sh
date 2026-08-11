# Source this from any RCI job script:  . ~/rci_env.sh
#
# Sets up modules + the ratsim venvs on an RCI compute node. Nothing here is
# ratsim-specific beyond the venv paths, and nothing requires root.
#
# Why this file exists instead of just writing `ml ...` in the job script:
# `module`/`ml` is a *shell function*, not a binary, and on RCI
# /etc/profile.d/lmod.sh REFUSES to define it inside a SLURM job:
#
#     # NOOP if running under known resource manager
#     if [ ! -z "$SLURM_NODELIST" ];then return; fi
#
# So `#!/bin/bash -l` does NOT get you modules in a job. It only ever appears to
# work when an interactive login shell already initialised Lmod and SLURM
# propagated the exported function -- which is why `srun --pty bash -i` and
# `bash -lc "srun ..."` work while a bare `srun ./job.sh` from a non-interactive
# ssh does not. Below we do what that profile script would have done past its
# guard: set MODULEPATH and source Lmod's own init directly.

# --- make `module` exist, however this shell was started ---------------------
if ! command -v module >/dev/null 2>&1; then
  # `:-` also covers the set-but-empty case seen inside jobs.
  export MODULEPATH="${MODULEPATH:-/opt/ohpc/pub/modulefiles}"
  export LMOD_SETTARG_CMD=":"
  for _f in "${LMOD_PKG:-/opt/ohpc/admin/lmod/lmod}/init/bash" \
            /opt/ohpc/admin/lmod/lmod/init/bash \
            /usr/share/lmod/lmod/init/bash; do
    # NOTE: do not source /etc/profile.d/lmod.sh here -- its early `return`
    # succeeds, so a `&& break` would exit this loop having defined nothing.
    # shellcheck disable=SC1090
    if [ -r "$_f" ]; then . "$_f" >/dev/null 2>&1; fi
    command -v module >/dev/null 2>&1 && break
  done
  unset _f
fi
command -v module >/dev/null 2>&1 || {
  echo "rci_env: FATAL - could not initialise Lmod" >&2; return 1 2>/dev/null || exit 1; }

# --- modules ----------------------------------------------------------------
# Keep these on one toolchain (GCCcore-13.3.0). Python 3.12.3 matches the laptop.
# NOTE: never pipe a module command -- the pipeline subshell discards the env.
module load Python/3.12.3-GCCcore-13.3.0
module load Xvfb/21.1.14-GCCcore-13.3.0
module load git/2.45.1-GCCcore-13.3.0

# --- PATH -------------------------------------------------------------------
# A SLURM batch job starts with PATH=/usr/local/bin:/usr/bin only -- no /sbin.
# An interactive `srun --pty bash -i` DOES have it, which is how a probe can
# report "ss present" for a tool the real job then cannot find. Same trap shape
# as the Lmod one above: interactive and batch shells are not the same shell.
case ":$PATH:" in
  *:/sbin:*) ;;
  *) export PATH="$PATH:/sbin:/usr/sbin" ;;
esac

# --- thread limits ----------------------------------------------------------
# CRITICAL on a shared node. /proc/cpuinfo reports the whole machine (32 CPUs on
# n23) while SLURM cgroup-limits you to --cpus-per-task. PyTorch/OpenMP read the
# machine count, so a job allocated 16 cores spawns ~32 OpenMP threads per
# process and thrashes. Measured, and it is brutal: two concurrent PPO runs went
# to opt_seconds=188 per iteration and fps 20, against opt_seconds~0.5 and
# fps 816 on the laptop -- ~390x slower per gradient step, while rollout_fps
# stayed a healthy ~300. The rollout looks fine and only the optimize phase
# collapses, which makes this easy to misread as "the cluster CPU is slow".
#
# The default is 4, NOT the allocation size -- do not "fix" this to
# $SLURM_CPUS_PER_TASK. More threads is actively harmful for this workload,
# because PPO's optimize phase is many tiny matmuls where thread
# synchronisation dominates the arithmetic. Measured on one node, same 20k-step
# PPO run, only OMP_NUM_THREADS differing:
#
#   threads=16 -> fps  39, opt_seconds 53.0
#   threads=4  -> fps 442, opt_seconds  0.98
#
# 11x throughput and 54x on the optimize phase, in the *small* direction. Note
# rollout_fps was ~530-560 in BOTH cases: Unity does its own threading and does
# not care, so this only ever shows up in the optimize phase.
#
# Raise it deliberately for a method that actually benefits (a large conv net,
# or dreamer's world model on CPU) -- and measure, don't assume. Capped by the
# allocation so a 2-core job doesn't ask for 4.
if [ -n "${SLURM_CPUS_PER_TASK:-}" ]; then
  _rat_threads=4
  [ "$SLURM_CPUS_PER_TASK" -lt "$_rat_threads" ] && _rat_threads="$SLURM_CPUS_PER_TASK"
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$_rat_threads}"
  export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$OMP_NUM_THREADS}"
  export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$OMP_NUM_THREADS}"
  export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-$OMP_NUM_THREADS}"
  unset _rat_threads
fi

# --- process/thread limit ---------------------------------------------------
# RLIMIT_NPROC counts THREADS, not processes, and is enforced per-user
# per-node -- so it has to be raised in every job, not once somewhere.
#
# One headless Unity instance spawns exactly 266 tasks (measured, dead
# consistent: 1 instance -> 270 tasks, 14 -> 3728, +266 each time). Unity sizes
# its worker pool from /proc/cpuinfo, i.e. the whole 128-thread node, not the
# cgroup -- the same "sees the machine, ignores the allocation" trap as
# OMP_NUM_THREADS above.
#
# Against the default soft limit of 4096 that caps you at 14 concurrent Unity
# instances per node. A 7-run job at n_envs=4 wants 28, and fails at the 15th
# with `fork: Resource temporarily unavailable`. That EAGAIN surfaces as a
# subprocess.TimeoutExpired in the launcher, which reads like a slow Unity boot
# and is not one -- boots were ~1s throughout. If you see a launch storm fail,
# check the launcher's own stderr for the fork error before believing the
# Python traceback.
#
# The hard limit is 4127387, so raising the soft limit needs no privileges.
# 32768 = 123 Unity instances, comfortably past anything one node can host.
if [ -n "${SLURM_JOB_ID:-}" ]; then
  ulimit -u 32768 2>/dev/null || echo "rci_env: WARNING - could not raise ulimit -u (now $(ulimit -u)); >14 concurrent Unity instances will fail to fork" >&2
fi

# --- paths ------------------------------------------------------------------
export RATSIM_GIT_DIR="${RATSIM_GIT_DIR:-/mnt/personal/$USER/git}"
export RATSIM_VENV_DIR="${RATSIM_VENV_DIR:-/mnt/personal/$USER/ratvenv}"
export RATSIM_UNITY_BIN="${RATSIM_UNITY_BIN:-/mnt/personal/$USER/ForagerSimBuildV1/ForagerSimBuildV1.x86_64}"

# --- experiment tracking ----------------------------------------------------
# Weights & Biases is OFF unless switched on, so without this every cluster run
# would quietly train with no tracking -- nothing errors, the runs just never
# appear. On by default here; unset or pass wandb=0 to opt a run out.
#
# Only the switch belongs in this file. The API key does NOT: rci_env.sh is
# tracked in git. Credentials live in ~/.netrc (mode 600), which the wandb
# client picks up automatically from any job.
export RATSIM_WANDB="${RATSIM_WANDB:-1}"

# Per-job scratch for pidfiles and Unity logs. $TMPDIR is NOT set by SLURM here,
# and bare /tmp is shared with other users' jobs on the same node.
if [ -n "${SLURM_JOB_ID:-}" ]; then
  if [ -d "/mnt/job-${SLURM_JOB_ID}" ]; then
    export TMPDIR="/mnt/job-${SLURM_JOB_ID}"
  else
    export TMPDIR="${TMPDIR:-/tmp}/ratsim-job-${SLURM_JOB_ID}"
    mkdir -p "$TMPDIR"
  fi
fi

# --- venv helpers -----------------------------------------------------------
# The venv symlinks the module interpreter, so the `module load` above is
# mandatory before activating -- otherwise you get the misleading
# "libpython3.12.so.1.0: cannot open shared object file".
ratsim_activate()        { . "$RATSIM_VENV_DIR/venv/bin/activate"; }
ratsim_activate_dreamer() { . "$RATSIM_VENV_DIR/dreamer_venv/bin/activate"; }

# The scheduler resolves each method's interpreter through an env var name
# (scheduler/config.py DEFAULT_PYTHON_ENV), not a path -- so these must be
# exported for `scheduler_run.py --machine rci` to dispatch anything. It picks
# the interpreter per method, which is how one scheduler process drives both the
# SB3 venv and the DreamerV3 venv.
export PPO_PYTHON_PATH="${PPO_PYTHON_PATH:-$RATSIM_VENV_DIR/venv/bin/python}"
export DREAMER_PYTHON_PATH="${DREAMER_PYTHON_PATH:-$RATSIM_VENV_DIR/dreamer_venv/bin/python}"
