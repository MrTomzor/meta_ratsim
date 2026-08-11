#!/bin/bash
# Does RecurrentPPO actually need an A100?
#
# `rci_gpu.yaml` / `rci_gpu2.yaml` give recurrent_ppo `needs: {gpu: 1}` with
# `method.device: cuda`. That was a reservation, never a measurement — and it
# costs a whole card out of a 2-card job, where it competes directly with
# dreamer (the method that genuinely saturates one). If CPU is close, the right
# move is a `recurrent_ppo` profile on `rci.yaml` instead, which drops it into
# the 112-thread CPU job with 7 slots and stops it touching the GPU pool at all.
#
# This runs BOTH arms back to back on the SAME node with the SAME allocation and
# the SAME metaseed, changing exactly one thing: method.device. Two separate
# jobs would confound the answer with a[01-16]-vs-g[01-12] CPU difference.
#
#   sbatch -p amdgpufast --gres=gpu:1 --cpus-per-task=16 --mem=64G \
#          --time=01:30:00 rci_port_probes/lstm_device_probe.sbatch
#
# The number that decides it is `[TIMING] opt:` — PhaseTimingCallback splits
# wall-clock into rollout (Unity-bound, identical across arms by construction)
# and optimization (the LSTM backward pass, the only thing the device changes).
# Comparing total wall-clock instead would dilute the effect by the rollout.
#
# NOTE: do NOT add `#!/bin/bash -l` — Lmod refuses to initialise inside a SLURM
# job (see rci_env.sh).
#SBATCH --job-name=lstm-device
#SBATCH --partition=amdgpufast
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=/mnt/personal/%u/logs/lstm-device-%j.out
#SBATCH --error=/mnt/personal/%u/logs/lstm-device-%j.out
set -uo pipefail

# shellcheck source=rci_env.sh
. "${RCI_ENV:-$HOME/rci_env.sh}"          # modules, RATSIM_* paths, per-job TMPDIR
ratsim_activate
export RATSIM_XVFB=1

# --- What is held constant -------------------------------------------------
# The task preset matters more than usual here: the `deep_memory` recurrent
# preset sets n_steps="auto" = episode_max_steps, so this task's 5000-step
# episodes give n_steps=5000, batch_size=5000, n_epochs=5 — i.e. 20 grad steps
# per iteration, each backpropagating through a 5000-step LSTM unroll. That
# unroll length IS the cost being measured, so the probe has to use the same
# world/task/agent as the experiments the decision is for (memory_orthomaze).
AGENT=sphereagent_2d_lidar
TASK=volumetric_exploration_5000_collision_penalty
WORLD=maze_sectored_orthogonal
METASEED=1234
# 20k env steps per iteration (n_steps 5000 x n_envs 4) -> 3 optimization
# phases per arm, enough to discard the first as warm-up and still have two.
STEPS=60000
STAMP=$(date +%Y%m%d_%H%M%S)

echo "=== $(date -Is) job $SLURM_JOB_ID on $(hostname) ==="
echo "cpus-per-task: ${SLURM_CPUS_PER_TASK:-?}   OMP_NUM_THREADS: ${OMP_NUM_THREADS:-unset}"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || echo "no GPU"
echo

cd "$RATSIM_GIT_DIR/meta_ratsim/ratsim_experiments"

ARMLOG_DIR="/mnt/personal/$USER/logs"

run_arm() {
  dev="$1"
  folder="lstm_device_probe_${dev}_${STAMP}"
  armlog="$ARMLOG_DIR/lstm-device-${SLURM_JOB_ID}-${dev}.log"
  echo
  echo "############################################################"
  echo "### ARM: method.device=$dev   -> results/$folder"
  echo "############################################################"
  t0=$SECONDS
  # tee, not plain stdout: the summary at the bottom greps the per-arm file, so
  # the two arms' timings stay separable without eyeballing the interleaved log.
  python train.py \
    method=recurrent_ppo \
    agent="$AGENT" task="$TASK" world="$WORLD" \
    total_steps="$STEPS" n_stages=1 n_envs=4 \
    metaseed="$METASEED" \
    method.device="$dev" \
    run_folder="$folder" 2>&1 | tee "$armlog"
  rc=$?
  echo "### ARM $dev finished: exit $rc, $((SECONDS - t0))s wall"
  # Between arms, not just at the end: a leaked Unity instance would make the
  # second arm fight for ports and pollute its rollout timings.
  "$RATSIM_GIT_DIR/meta_ratsim/ratsim/scripts/stop_ratsim_headless.sh" --all || true
  # Ports sit in TIME_WAIT for ~60 s and the launcher's bind probe waits 20 s.
  sleep 75
  return $rc
}

run_arm cuda; RC_CUDA=$?
run_arm cpu;  RC_CPU=$?

echo
echo "============================================================"
echo "=== SUMMARY  (exit: cuda=$RC_CUDA cpu=$RC_CPU)"
echo "============================================================"
# Re-print every timing line with its arm, so the verdict is readable without
# opening TensorBoard. The per-arm logs keep the full context.
for dev in cuda cpu; do
  echo "--- $dev ---"
  grep -h "^\[TIMING\]" "$ARMLOG_DIR/lstm-device-${SLURM_JOB_ID}-${dev}.log" \
    || echo "(no timing lines — arm probably died before its first iteration)"
done
echo "=== $(date -Is) done ==="

[ "$RC_CUDA" -eq 0 ] && [ "$RC_CPU" -eq 0 ]
