#!/bin/bash
# Is the RecurrentPPO optimization phase latency-bound, or is the A100 being
# wasted by something fixable?
#
# Follow-up to lstm_device_probe.sbatch, which measured CUDA only 1.32x faster
# than CPU at optimization (6.58 s vs 8.68 s). Working back from that: ~400k
# dependent LSTM cell-steps per optimization phase => 16.5 us per step on an
# A100, against ~0.03 us of actual arithmetic. That says latency-bound — but it
# is an estimate from a FLOP count, not a measurement.
#
# This measures it directly, with no Unity and no scheduler in the way:
# nn.LSTM fwd+bwd at the real shapes, sweeping ONLY the batch dimension.
#
#   flat from batch 1 -> 16 on GPU  => latency-bound, diagnosis confirmed, and
#                                      the batching fix in lstm_fastpath.py is
#                                      worth exactly the ratio measured here
#   scales with batch on GPU        => diagnosis wrong, look elsewhere
#
# Batch is the only axis worth sweeping: BPTT over T timesteps is a chain of T
# *dependent* ops, so time is linear in seq_len on any device by construction.
#
#   sbatch -p amdgpufast --gres=gpu:1 --cpus-per-task=4 --mem=32G \
#          --time=00:15:00 rci_port_probes/lstm_batch_scaling.sbatch
#
# NOTE: do NOT add `#!/bin/bash -l` — Lmod refuses to initialise inside a SLURM
# job (see rci_env.sh).
#SBATCH --job-name=lstm-batch
#SBATCH --partition=amdgpufast
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/mnt/personal/%u/logs/lstm-batch-%j.out
#SBATCH --error=/mnt/personal/%u/logs/lstm-batch-%j.out
set -uo pipefail

# shellcheck source=rci_env.sh
. "${RCI_ENV:-$HOME/rci_env.sh}"
ratsim_activate

echo "=== $(date -Is) job $SLURM_JOB_ID on $(hostname) ==="
echo "cpus-per-task: ${SLURM_CPUS_PER_TASK:-?}   OMP_NUM_THREADS: ${OMP_NUM_THREADS:-unset}"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || echo "no GPU"
echo

python - <<'PY'
import time
import torch as th
import torch.nn as nn

# The real shapes: deep_memory preset (lstm_hidden_size 256, n_lstm_layers 1)
# on a task with episode_max_steps 5000, so n_steps=5000 => a 5000-step unroll.
# INPUT is the features-extractor output = flattened obs; 100 is the right order
# for lidar+goal+compass+gps and barely matters (the 256x1024 recurrent matmul
# dominates the per-step cost either way).
SEQ, HIDDEN, INPUT = 5000, 256, 100
BATCHES = [1, 2, 5, 16, 64]
REPS = 3

def bench(device, batch):
    dev = th.device(device)
    lstm = nn.LSTM(INPUT, HIDDEN, num_layers=1).to(dev)
    x = th.randn(SEQ, batch, INPUT, device=dev, requires_grad=True)
    h = th.zeros(1, batch, HIDDEN, device=dev)
    c = th.zeros(1, batch, HIDDEN, device=dev)

    def one():
        out, _ = lstm(x, (h, c))
        out.sum().backward()
        lstm.zero_grad(set_to_none=True)
        if x.grad is not None:
            x.grad = None

    one()                                    # warm-up: cuDNN plan + allocator
    if device == "cuda":
        th.cuda.synchronize()
    times = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        one()
        if device == "cuda":
            th.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return min(times)                        # min, not mean: less scheduler noise

print(f"nn.LSTM({INPUT}, {HIDDEN}) | seq_len={SEQ} | fwd+bwd | best of {REPS}")
print(f"torch {th.__version__}, cuda available={th.cuda.is_available()}")
print()
print(f"{'batch':>6} | {'cuda ms':>9} {'us/step':>8} | {'cpu ms':>9} {'us/step':>8} | "
      f"{'cpu/cuda':>8} | {'cuda vs b=1':>11}")
print("-" * 78)

base = {}
for b in BATCHES:
    row = {}
    for dev in ("cuda", "cpu"):
        if dev == "cuda" and not th.cuda.is_available():
            row[dev] = float("nan"); continue
        row[dev] = bench(dev, b)
    base.setdefault("cuda", row["cuda"])
    # us per cell-step: the unroll is SEQ dependent steps regardless of batch.
    cu_us, cp_us = row["cuda"] / SEQ * 1e6, row["cpu"] / SEQ * 1e6
    print(f"{b:>6} | {row['cuda']*1e3:>9.1f} {cu_us:>8.1f} | "
          f"{row['cpu']*1e3:>9.1f} {cp_us:>8.1f} | "
          f"{row['cpu']/row['cuda']:>8.2f}x | {row['cuda']/base['cuda']:>10.2f}x")

print()
print("READ IT LIKE THIS:")
print("  'cuda vs b=1' flat (~1.0x) across batches  -> latency-bound: the card is")
print("     idle waiting on a dependency chain, and batching 64x more work costs")
print("     nothing. The 1.32x device gap is then structural, not fixable.")
print("  'cuda vs b=1' growing ~linearly           -> compute-bound: the GPU was")
print("     doing real work and the earlier diagnosis is wrong.")
print("  RecurrentPPO today runs at batch=1 (lstm_fastpath loops per sequence);")
print("  batching the sequences would move it to batch=2, or 5 at batch_size=20000.")
PY

echo "=== $(date -Is) done ==="
