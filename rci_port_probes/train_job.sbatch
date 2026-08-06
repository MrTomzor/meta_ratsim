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

# Pidfiles and Unity logs go in the per-job scratch, never bare /tmp: /tmp is
# shared with other users' jobs on the same node, and a colliding pidfile means
# one job kills the other job's Unity.
export RATSIM_RUNDIR="$TMPDIR"

# Force the fast headless path explicitly rather than relying on autodetection,
# so the job fails loudly if `ml Xvfb` ever stops providing xvfb.
export RATSIM_XVFB=1

# Interim port separation until the launcher learns to do this itself
# (RCI_CLUSTER_PORT.md phase 2 #7). Stride 10 leaves room for n_envs<=10, and
# the launcher refuses to reuse a live port, so the worst case is a clean
# failure rather than two jobs sharing one Unity.
BASE_PORT=$(( 9100 + (SLURM_JOB_ID % 88) * 10 ))

echo "=== $(date -Is) job $SLURM_JOB_ID on $(hostname) ==="
echo "python:     $(python -c 'import sys; print(sys.executable)')"
echo "unity bin:  $RATSIM_UNITY_BIN"
echo "rundir:     $RATSIM_RUNDIR"
echo "base_port:  $BASE_PORT"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || echo "no GPU"
echo

cd "$RATSIM_GIT_DIR/meta_ratsim/ratsim_experiments"
# `|| rc=$?` and not a bare call: set -e would skip the cleanup below on failure,
# which is exactly when the cleanup matters.
rc=0
python train.py base_port="$BASE_PORT" "$@" || rc=$?

# Reap anything the trainer left holding a port. train.py kills its own
# instances at exit, but not if it was cancelled or died hard.
"$RATSIM_GIT_DIR/meta_ratsim/ratsim/scripts/stop_ratsim_headless.sh" --all || true

echo "=== $(date -Is) train.py exit $rc ==="
exit $rc
