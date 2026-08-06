# RCI cluster port — probes and job scripts

Working scripts for running ratsim on CTU's RCI cluster. Findings and the roadmap
live in `../RCI_CLUSTER_PORT.md`; read that first.

| File | What it's for |
|---|---|
| `check_cluster_node.sh` | Two-tier node probe. Tier 1 (no args) reports capability: login-node detection, xvfb (loading the module if needed) **and whether it actually starts**, per-job scratch, GPU + driver with wheel advice, Python/venv, `ss`, existing listeners on 9000–9999, outbound internet. Tier 2 (pass the build path) launches Unity headless and is the only conclusive answer. |
| `rci_env.sh` | Sourced by every job script. Initialises Lmod (SLURM jobs do **not** get it by default), loads the Python/Xvfb/git modules, sets `RATSIM_*` paths and a per-job `TMPDIR`, and provides `ratsim_activate` / `ratsim_activate_dreamer`. |
| `install_job.sbatch` | Installs/updates the stack. Runs on a **CPU node** on purpose — see the glibc note inside. |
| `train_job.sbatch` | Runs one training. Everything after the script name is passed to `train.py`. Defaults to `cpufast`/no GPU; override with `sbatch -p gpufast --gres=gpu:1 --time=...`. Sets `RATSIM_RUNDIR="$TMPDIR"` and `RATSIM_XVFB=1`, derives an interim `base_port` from `$SLURM_JOB_ID`, and reaps leftover Unity instances on exit. |
| `scheduler_job.sbatch` | Runs the **scheduler** for one experiment in a single job (`sbatch scheduler_job.sbatch <exp>`), using `scheduler/machines/rci.yaml`. Sets `OMP_NUM_THREADS=4` to match that file's `needs.cpu_slot` — without it two concurrent runs oversubscribe the cgroup and `opt_seconds` goes from 1.5 to 188. Pick the partition for the whole experiment, not one run. |
| `gpu_check.sh` | Post-install verification: torch arch list + a **real matmul** (not just `is_available()`), SB3/gym/ratsim imports, JAX devices, dreamerv3 import. |
| `probe.sh` | Single-variant Unity headless launcher used for the original measurements. Verifies the *listener's* process group so leaked processes can't fake a pass. |
| `lidar_check.py` | Confirms lidar data is non-degenerate under headless mode, not merely present. |

## Usage

```bash
scp rci_port_probes/{check_cluster_node.sh,rci_env.sh,gpu_check.sh,install_job.sbatch,train_job.sbatch} rci:~/
ssh rci
sbatch ~/install_job.sbatch                                    # install
srun -p gpufast --gres=gpu:1 --time=00:08:00 ~/gpu_check.sh     # verify
sbatch -p gpufast --gres=gpu:1 --time=00:30:00 ~/train_job.sbatch \
    method=ppo world=maze_default total_steps=20000 n_stages=1  # smoke test
```

## Traps these scripts exist to avoid

- `module`/`ml` is a **shell function**. It is absent in SLURM jobs (OpenHPC's
  `/etc/profile.d/lmod.sh` returns early when `$SLURM_NODELIST` is set), and it must
  never be piped — a pipeline subshell discards the environment change.
- `torch.cuda.is_available()` returning `True` does **not** mean the GPU works. New
  wheels drop old archs; a V100 (sm_70) needs cu126.
- `nvidia-smi` on a GPU-less node prints its failure to **stdout**, so `2>/dev/null`
  won't hide it and the text can land in a variable you meant to be a version number.
- The venv is unusable without its `module load Python/...` line; the error mentions
  only `libpython3.12.so.1.0`.
- **`/proc/cpuinfo` shows the whole node, not your cgroup share.** PyTorch/OpenMP size
  their thread pools from it, so an allocation of 16 cores runs ~32 threads per process.
  `rci_env.sh` pins the `*_NUM_THREADS` vars to `$SLURM_CPUS_PER_TASK`. Only the optimize
  phase collapses (`rollout_fps` stays healthy), which makes it read as slow hardware.
- A batch job's `PATH` is `/usr/local/bin:/usr/bin` — **no `/sbin`**, so `ss` is invisible.
  An interactive `srun --pty bash -i` has it, which is how a probe can pass for a tool the
  real job cannot find. Interactive shells are more generous than batch ones in at least
  `PATH` and Lmod, so **verify capabilities from a batch job**, not an interactive one.
