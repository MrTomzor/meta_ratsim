#!/bin/bash
# Run one ratsim training on an RCI compute node.
#
#   sbatch train_job.sbatch method=ppo world=maze_default total_steps=20000 n_stages=1
#   sbatch train_job.sbatch def=dreamers_maze_smoke method=dreamer
#
# Everything after the script name is passed through to train.py verbatim.
#
# Defaults to cpufast (no GPU) because PPO with n_envs=1 is faster on CPU
# anyway (see the meta-repo CLAUDE.md) and a GPU sitting idle is wasted. For a
# GPU method, override on the command line — sbatch flags beat #SBATCH lines:
#
#   sbatch -p gpufast --gres=gpu:1 --time=01:00:00 train_job.sbatch def=... method=dreamer
#
# NOTE: do NOT add `#!/bin/bash -l`. Lmod refuses to initialise inside a SLURM
# job (see rci_env.sh); the login-shell flag buys nothing and hides the reason.
#SBATCH --job-name=ratsim-train
#SBATCH --partition=cpufast
#SBATCH --time=03:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/mnt/personal/%u/logs/train-%j.out
#SBATCH --error=/mnt/personal/%u/logs/train-%j.out
set -euo pipefail

# shellcheck source=rci_env.sh
. "${RCI_ENV:-$HOME/rci_env.sh}"          # modules, RATSIM_* paths, per-job TMPDIR

ratsim_activate

# Force the fast headless path explicitly rather than relying on autodetection,
# so the job fails loudly if `ml Xvfb` ever stops providing xvfb.
export RATSIM_XVFB=1

# NOTE: no RATSIM_RUNDIR and no base_port are set here on purpose. Since phase 2
# the launcher derives both from $SLURM_JOB_ID itself -- per-job scratch for
# pidfiles, a job-derived port window, bind-tested. Setting base_port here would
# actually be *worse*: an explicit port switches the launcher to strict mode,
# which fails hard instead of probing past a foreign listener.

echo "=== $(date -Is) job $SLURM_JOB_ID on $(hostname) ==="
echo "python:     $(python -c 'import sys; print(sys.executable)')"
echo "unity bin:  $RATSIM_UNITY_BIN"
echo "rundir:     $(python -c 'from ratsim.unity_launcher import _rundir; print(_rundir())')"
echo "base_port:  $(python -c 'from ratsim.unity_launcher import default_base_port; print(default_base_port())')"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || echo "no GPU"
echo

cd "$RATSIM_GIT_DIR/meta_ratsim/ratsim_experiments"
# `|| rc=$?` and not a bare call: set -e would skip the cleanup below on failure,
# which is exactly when the cleanup matters.
rc=0
python train.py "$@" || rc=$?

# Reap anything the trainer left holding a port. train.py kills its own
# instances at exit, but not if it was cancelled or died hard.
"$RATSIM_GIT_DIR/meta_ratsim/ratsim/scripts/stop_ratsim_headless.sh" --all || true

echo "=== $(date -Is) train.py exit $rc ==="
exit $rc
