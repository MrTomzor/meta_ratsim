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

# --- paths ------------------------------------------------------------------
export RATSIM_GIT_DIR="${RATSIM_GIT_DIR:-/mnt/personal/$USER/git}"
export RATSIM_VENV_DIR="${RATSIM_VENV_DIR:-/mnt/personal/$USER/ratvenv}"
export RATSIM_UNITY_BIN="${RATSIM_UNITY_BIN:-/mnt/personal/$USER/ForagerSimBuildV1/ForagerSimBuildV1.x86_64}"

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
