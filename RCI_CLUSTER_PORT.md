# ORIGINAL INSTRUCTIONS FOR USING RCI CLUSTER
Please read also the instructions by the sysadmin in ~/Downloads/rci_how_to_start.html

# Porting ratsim to the RCI cluster (CTU) — analysis + roadmap

Handoff doc for running ratsim training on the **RCI cluster** (`login[1-4].rci.cvut.cz`),
a SLURM-scheduled HPC cluster at CTU. Written after reading the cluster's
`how_to_start` wiki page and auditing our stack against it.

**Where things stand (2026-08-07):** Phases 0–2 complete. Install, headless Unity, single training,
concurrent trainings and the scheduler all run on the cluster.

**Both design decisions were made on 2026-08-07.** The first is now implemented (2026-08-10); the
second is the next piece of work:
1. ✅ **`n_envs` moved out of the machine config onto the method / experiment def** — `MethodSpec.n_envs`,
   default `4`, same value on every machine; machine configs keep capacity only (§0.8). A machine
   config that still sets `n_envs:` is a hard error naming the fix. `machines/rci.yaml` was re-sized
   in the same change to the verified AMD shape (`cpu_slot: 112`, `needs: 16` → 7 concurrent runs),
   and `validate_against_machine` now warns below ~4 cpu_slots per env. `rci_env.sh` also raises
   `ulimit -u` to 32768 for every job (§0.95).
2. **Dreamer scales via N GPUs in one job, one scheduler** — needs a `GpuAllocator` mirroring
   `PortAllocator` to export `CUDA_VISIBLE_DEVICES` per child (before phase 3). A whole GPU node
   (72 threads, 4× V100) is exactly 4 runs × 18 threads, which is also the fair-share ratio.
   Until it exists, `rci.yaml` declares `gpu: 1` precisely to stop several dreamers landing on
   device 0 — do not raise it without the allocator.

Account resource ceilings and node topology are in **§0.55** — planning limits only; **do not
launch anything large without asking the user first.**

⚠️ **Read §0.56 before planning anything.** The wiki scrape overturned several assumptions:
a SLURM "CPU" is a **thread** not a core; max allocatable per node is 46/70/126/252, not the node's
full count; the admins explicitly recommend the **AMD nodes**, which we had never used; the
Intel-built venvs **already run there unmodified** (verified on an A100); and the claim that dreamer
is simulator-bound is **false** — it is training-bound, which is the actual argument for A100.

**Untested gaps:** whether the ~4-cores-per-env threshold (§0.9) holds for dreamer as it does for
PPO, `n_envs=4` beyond 16 cores, and any laptop dreamer baseline.

**Epistemic status:** every claim about *our code* below was verified by reading the
source (file:line given). **§1 was verified empirically on the laptop (2026-08-06), and
it overturned this doc's original conclusion.** **Phases 0, 1 and 2 are now COMPLETE — the probe has
been run on a real RCI compute node and a 20k-step PPO training ran to completion there
(2026-08-06), with checkpoints and episode logs readable from `login1`.**
Measured cluster facts are in §0.5; they supersede the wiki wherever they disagree. The
wiki's example outputs are visibly stale (its `nvidia-smi` sample shows driver 418 / CUDA
10.1, i.e. ~2019, while the sidebar advertises an "H200 nodes upgrade 2025"), so treat its
specifics as illustrative. Remaining **[VERIFY]** items are called out in §9.

---

## 0. Cluster model in one paragraph

A **node** is a physical server, not a process. RCI has `n01`–`n33` (Intel; `n21`–`n32`
have GPUs), `a01`–`a16` (AMD CPU), `g01`–`g12` (AMD GPU), plus `login1`–`login4`.
A **partition** is a queue — a named policy over a set of nodes, mostly differing by wall-clock
limit: `gpufast` 4h, `gpu` 1 day, `gpulong` 3 days, `gpuextralong` 21 days. The same physical
nodes appear in several partitions. **There is no direct SSH to compute nodes**; you get there
only through SLURM (`srun`/`salloc`/`sbatch`). Critically, **SLURM allocates fractions of nodes** —
ask for 1 of 4 GPUs on `n21` and the other 3 go to other jobs. Your processes and strangers'
processes therefore share one kernel, one network stack, and (usually) one `/tmp`. That
sharing is the root of issues #1 and #2 below.

`/home` and `/mnt/personal/<user>` are network filesystems mounted on **every** node. The wiki
is explicit: *"Don't use home directory for storing data. Always use /mnt/personal which is
faster and bigger."* Node-local scratch (`/data/temporary/`, or `/mnt/job-<jobid>` for
multi-node) is fast but visible only on the allocated node and **deleted when the job ends**.

---

## 0.5 Measured cluster facts (Phase 0 complete, 2026-08-06)

Observed on a `gpufast` compute node via `rci_port_probes/check_cluster_node.sh`.
**These are measurements, not wiki claims.**

| Fact | Value | Consequence |
|---|---|---|
| **Headless launch** | ✅ **PASS — build boots under xvfb, TCP port opens, no fatal signals** | The blocking risk is gone. No container needed. |
| `Xvfb` availability | **Provided as an Lmod module**, not on `PATH` by default | Every job script needs `ml Xvfb`. |
| GPU | **Tesla V100-SXM2-32GB**, 1 visible with `--gres=gpu:1` | Volta, sm_70. See bf16 note below. |
| Driver | **575.51.03** | Supports CUDA up to **12.9**. ⚠️ **Not** "any wheel works" — see the CUDA 13 trap below. |
| Outbound internet on compute node | ✅ works | `pip install` need not be confined to `login1`. Issue #9 dissolved. |
| Python | system `python3` is **3.6.8** everywhere (has `venv`, so the old preflight *passed* it) | **Must use `ml Python/...`.** 3.6 is far below `requires-python >=3.9`. `install.sh` now enforces a version floor. |
| **login1 OS** | **CentOS 7, glibc 2.17**, git 1.8.3.1, Python 3.6.8 | ⚠️ **Do NOT pip install here** — see below. It is the *only* old login node. |
| **login2 / login3 / login4 OS** | **Rocky Linux 8.10 / 8.8 / 8.10, glibc 2.28** | Same glibc as the compute nodes, so the `Python/3.12.3` module and the venv work here. Use `login2` for monitoring (§ phase 3 #11). |
| **compute node OS** | **glibc 2.28**, git 2.31.1 | Modern manylinux wheels work here. |
| `$TMPDIR` | **unset** | Must scope pidfiles/logs by `$SLURM_JOB_ID` ourselves — issue #2 confirmed live. |
| Ports 9000–9999 on that node | **2 listeners already present** | **Issue #1 confirmed real, not theoretical.** Foreign processes occupy our default range. |
| `ss`, Lmod | present | `start_ratsim_headless.sh` precondition satisfied. |

**V100 is sm_70 (Volta)** — no native bf16 tensor-core path, and importantly **new CUDA wheels
have started dropping this arch entirely** (see the CUDA section: cu128 ships CC ≥9.0 only).
This is the single most consequential hardware fact here. `torch.cuda.is_bf16_supported()`
nonetheless reports `True`. **Measured outcome for DreamerV3: keep bfloat16.** Lacking bf16
*tensor cores* does not stop XLA emulating the dtype correctly, and forcing `float16` actually
**breaks** dreamer (§0.7). Do not infer "avoid bf16" from sm_70.
If RCI's advertised H200 nodes are reachable, most of this reverses — another reason to record
hardware per partition.

**Nodes are heterogeneous**: the above is one `gpufast` node. Re-run the probe on `gpu` /
`gpulong` before trusting driver and GPU model there.

### 0.55 Account resource limits — the ceiling we are allowed to ask for

From RCI's web interface (user-reported 2026-08-07). **These are per-user aggregate caps across all
of your simultaneously running jobs in that partition group — not per-job limits.**

| Partition group | Max jobs | Max CPUs | Max Mem | Max GPUs |
|---|---|---|---|---|
| `cpufast`, `gpufast`, `amdfast`, `amdgpufast` (4 h) | 500 | **700** | 6 | **8** |
| `cpu`, `gpu`, `amd`, `amdgpu` (1 day) | 400 | 200 | 3 | **8** |
| `cpulong`, `gpulong`, `amdlong`, `amdgpulong` (3 days) | 300 | 200 | 4 | **6** |
| `cpuextralong`, `gpuextralong`, … (21 days) | 500 | 200 | 3 | **8** |

#### Node topology, and the accelerators we deliberately do not use (measured 2026-08-07)

`sinfo -o '%20P %6D %24G %N'`:

| Partition family | Nodes | GRES |
|---|---|---|
| `cpu*` | `n[01-20,33]` | — |
| `gpu*` (`gpufast`/`gpu`/`gpulong`/`gpuextralong`) | `n[21-32]` | **`gpu:v100:4`** |
| `amd*` | `a[01-16]` | — |
| `amdgpu*` | `g[01-12]` | `gpu:a100:4` (`g[11-12]`: `a100:8`) |
| **`h200*`** | `h[01-03]` | **`gpu:nvidia_h200_nvl:8`** |
| `ipu` | `ipu01` | `ipu:1` |

**A GPU node is 72 CPUs, 384 GB and 4× V100 — so the proportionate ask is 18 cores per GPU.**
This is the number that should size any multi-GPU job: 1 GPU → ~18 cores, 4 GPUs → the whole node
at 72. Taking 32 cores with a single GPU would strand two GPUs nobody else can then schedule.

> 🚫 **Stay off the `h200*` partitions** (`h[01-03]`, 8× H200 141GB). The admin rule is to use them
> only at high utilisation, and one dreamer world model will not saturate one. **This is *not*
> because the sim is the bottleneck** — an earlier version of this doc said dreamer was
> "Unity-bound at ~67 env-steps/s" and that is **wrong**, see §0.56.
>
> ✅ **`amdgpu*` (A100) is a different matter and is now the recommended target** — see §0.56.

### 0.56 ⚠️ The AMD subcluster — the biggest thing this port got wrong

Read after the wiki scrape (`ratsim_experiments/docs/rci_wiki/`) and the 2022 tutorial. The admin's
own recommendation slide says, verbatim: **"Use much faster AMD nodes."** Phases 0–2 used only the
2019 Intel hardware.

#### First, three corrections to numbers used throughout this document

1. **A SLURM "CPU" is a *thread*, not a core.** Intel CPU nodes are 24c/48t, Intel GPU nodes 36c/72t,
   AMD CPU nodes 64c/128t. So `--cpus-per-task=16` is **8 physical cores**, and the
   "~4 cores per env" threshold of §0.9 is really **~4 threads = 2 physical cores per env**. The
   measurements are unaffected; the vocabulary was wrong.
2. **Max allocatable per node is less than the node has**: **46** on Intel CPU nodes, **70** on Intel
   GPU nodes, **126** on AMD CPU nodes, **124** on `g[01-10]`, **252** on `g[11-12]`, **382** on n33.
   A planned `--cpus-per-task=48` would never have scheduled. On Intel that caps a CPU node at
   **2** PPO runs at 16 threads, not the 3 claimed earlier.
3. `--cpus-per-task` **should be even** (cores are allocated whole), and `--nodes=1` should be set
   explicitly even for single-node jobs — both are in the tutorial's recommendations.

#### What was measured on AMD (2026-08-07, jobs 11317999 / 11318000)

**A1 on `a07` (`amdfast`) — everything works, and the module story is simpler than the wiki says.**
The wiki states Intel and AMD use different module trees. **For our stack they do not.**
`Python/3.12.3-GCCcore-13.3.0`, `Xvfb/21.1.14-GCCcore-13.3.0` and `git/2.45.1-GCCcore-13.3.0` all
load on AMD from the *identical* `/mnt/appl/software/...` paths, with the same
`MODULEPATH=/opt/ohpc/pub/modulefiles` and the same Lmod bootstrap. Rocky 8.9, **glibc 2.28 — same
as the Intel compute nodes.** Unity booted under xvfb in **1 s**. The missing-`ss`-without-`/sbin`
trap is identical. **`rci_env.sh` needs no AMD-specific changes.**

**A2 on `g05` (`amdgpufast`, A100-SXM4-40GB) — the Intel-built venvs run unmodified.**

| | result |
|---|---|
| SB3 venv on AMD | ✅ all imports; torch 2.13.0+cu126 sees the A100 at **cc 8.0**, `is_bf16_supported()` True |
| Dreamer venv on AMD | ✅ jax 0.4.33 → `[CudaDevice(id=0)]` |
| jax matmul 4096³ | **bf16 0.7 ms vs fp32 1.3 ms** — native bf16 on sm_80, unlike the V100 |
| PPO end-to-end | ✅ exit 0, 4 Unity envs |
| Dreamer end-to-end | ✅ exit 0, stage completed |

**So there is no second venv to build.** Every wheel is generic manylinux x86-64 and the interpreter
path is shared, so one venv serves both subclusters. This also retroactively justifies pip-over-
modules: module builds are architecture-specific and would *not* have been portable this way.

⚠️ **The AMD GPU nodes run an OLDER driver**: `550.54.14` (CUDA 12.4) against `575.51.03` (12.9) on
the Intel GPU nodes. JAX warns that its PTX compiler (12.9.86) is newer than the driver and
therefore **disables parallel compilation**, so JIT warm-up is slower. Not a correctness problem —
both smokes exited 0 — but it eats into short runs.

#### Why this changes the plan

| | Intel (phases 0–2) | AMD |
|---|---|---|
| CPU node | `n01-20`: 46 threads → **2 PPO runs** | `a01-16`: **126 threads, 1 TB → 7 PPO runs** |
| GPU node | `n21-32`: 4× V100 32GB, sm_70, **emulated** bf16 | `g01-10`: 4× A100 40GB · `g11-12`: **8× A100**, sm_80, **native** bf16 |

#### AMD vs Intel, measured properly (job 11318011, `a05`, 16 threads, same 20k run)

| | | Intel `n02` | AMD `a05` | |
|---|---|---|---|---|
| **`n_envs=1`** | overall fps | 360 | **487** | +35% |
| | `rollout_fps` (env stepping) | 482 | **629** | **+30%** |
| | `opt_seconds` (training) | 1.29 | **0.801** | **1.6× faster** |
| **`n_envs=4`** | overall fps | 688 | **723** | +5% |
| | `rollout_fps` | 999 | 967 | **−3%, tied** |
| | `opt_seconds` | 4.29 | **3.52** | **1.2× faster** |

**Training compute is faster on AMD in both cases** (Zen 3 IPC on PPO's small CPU matmuls).
**Env stepping is faster only at `n_envs=1`.** At `n_envs=4` the two architectures land within 3% of
each other at ~1000 `rollout_fps` — two very different CPUs hitting the same number is the tell:

> ⚠️ **At `n_envs=4` on 16 threads the rollout is bounded by our single-threaded Python side**
> (obs assembly, TCP connector, VecEnv stepping), not by Unity or the CPU. ~1000 env-steps/s is a
> ceiling **in our code**, and it is now the binding constraint on PPO throughput. Worth profiling
> before buying more hardware to push against it.

❌ **Discount the A2 smoke's "fps 576 on 8 AMD threads vs 251 on 8 Intel threads".** That was at
8 threads, where Intel sits *below* the cores-per-env threshold (§0.9) and AMD does not — it
measured the cliff, not raw speed. At 16 threads, above the threshold on both, the gap collapses
from 2.3× to 5%.

**So the case for AMD is capacity, not speed.** At our actual `n_envs=4`, a single run gains ~5%,
which is noise. The reason to move is **126 usable threads against 46 → 7 concurrent runs per job
instead of 2**, plus native bf16 and 8 GPUs/node on `g[11-12]` for dreamer.

#### The bottleneck claim this overturns

This doc previously said dreamer is *"Unity-bound at ~67 env-steps/s"*. **That is false**, and §0.9's
own data disproves it: PPO drives the *same* 4 Unity envs on the *same* 16 threads at **688**
env-steps/s, while dreamer manages ~53. The simulator can go 13× faster than dreamer uses, and
`n_envs=4` buying dreamer only 6% is the same fact. **Dreamer is bounded by the training side.**

Consequently a faster GPU plausibly *does* speed dreamer up, which is the real argument for A100.
Whether the binding constraint is GPU compute or JAX/Python overhead is **[VERIFY]** — one
single-run A100-vs-V100 comparison settles it and should come before any multi-GPU work.

Memory came through as a bare number; 6/3/4/3 next to 700/200 CPUs reads as **TB**, but it is not
worth relying on — every job here requests tens of GB, three orders of magnitude below the cap.

> ⚠️ **Treat this table as a ceiling for *planning*, never as a target.** The user's standing rule:
> **do not launch anything large without asking first.** The cluster is shared and monitored, and
> these numbers are what the account *may* consume, not what it is polite to consume. Probe and
> smoke-test jobs stay small and on the `*fast` partitions.

What actually binds, per method:

- **Dreamer and recurrent_ppo need 1 GPU each**, so the GPU column is the real limit on concurrency:
  **at most 8** concurrent GPU runs (fast / 1-day) or **6** (3-day). CPUs and memory are nowhere
  near binding for these.
- **Plain PPO is CPU-only here** (`method.device: cpu`, §0.6), so it is bounded by the CPU column:
  200 CPUs ÷ 4 `cpu_slot` per run ≈ **50 concurrent PPO runs** outside the fast group. That is far
  more than any experiment needs; the practical limit becomes disk and the scheduler process.
- The fast group's 700 CPUs is the generous one, but **4 h wall-clock** makes it a smoke-test
  partition only — one 1M-step dreamer run alone is ~4.1 h (§0.7).

⚠️ **The GPU cap is not the same as GPUs-per-job.** A single SLURM job's `--gres=gpu:N` must be
satisfied *on one node*, so the per-job ceiling is the node's GPU count, which we have **not
measured**. One command settles it before committing to the multi-GPU design:
`sinfo -p gpu,gpulong -o '%P %G %N'`.

### Three things from the sysadmin's `how_to_start` worth obeying

Read it directly at `~/Downloads/rci_how_to_start.html` — these are easy to miss:

- **"Don't use salloc/ssh method for GPU nodes (gpufast partition), please."** An explicit request.
  Use `srun`/`sbatch`; `ssh`-ing into an allocated GPU node is out. (Everything we do already
  goes through `srun`/`sbatch`.)
- **Node-local scratch is `/data/temporary/` for single-node jobs**, and `/mnt/job-<jobid>` only
  for *multi-node* ones — described as "very fast", visible only on the allocated node, deleted
  when the job ends. We currently scope pidfiles by `$SLURM_JOB_ID` under `$TMPDIR`/`/tmp`, which
  is fine for a few tiny files; `/data/temporary/` is the place to look if staging the venv to
  local disk ever becomes worthwhile (see the small-files caveat in §3). **[VERIFY]** — not yet
  confirmed present on a node.
- **Interactive jobs are only allowed in `cpufast`/`gpufast`/`amdfast`/`amdgpufast`**, and SMB
  shares exist for bulk copies (`\\login3.rci.cvut.cz\personal\%username`,
  `\\147.32.87.117\home\%username`) if `scp` is ever inconvenient.

### ⚠️ Install on a COMPUTE node, not `login1` — this reverses the obvious advice

`login1` is **CentOS 7 with glibc 2.17**. Compute nodes have **glibc 2.28**. `pip` resolves
wheels against *the machine it runs on*, and current torch wheels are `manylinux_2_28`. Install
on `login1` and pip either fails or silently pulls an ancient torch that then can't be used.

Verified on a compute node with `ml Python/3.12.3-GCCcore-13.3.0`: `torch` resolves to
**2.13.0** and `jax[cuda12]` to **0.11.0**, with pip 26.2.1. So the install belongs in a small
`cpufast` batch job (compute nodes have internet, §0.5). Pass `--gpu` explicitly, since a CPU
node has no `nvidia-smi` and `install.sh` would auto-select CPU JAX/TF wheels.

### ⚠️ The CUDA 13 trap — stock `pip install torch` is NOT safe here

**CONFIRMED on a V100 node**, not predicted: a stock install produced `torch 2.13.0+cu130` and

```
UserWarning: CUDA initialization: The NVIDIA driver on your system is too old (found version 12090)
torch: 2.13.0+cu130 | built for CUDA: 13.0
cuda available: False
```

`12090` = CUDA 12.9, the driver's ceiling. CUDA 13 is a *major* bump, so minor-version
compatibility does not rescue it. Note how quiet this failure is: pip succeeded, every import
succeeded, and training would simply have run on CPU forever.

The trap is that `torch` is a **transitive dependency of `stable_baselines3[extra]`**, so it
comes from the default index unless pinned first.

### ⚠️ …and the second half: new wheels DROP old GPU architectures

Fixing the driver problem by jumping to cu128 produced a *different* failure, one step later:

```
torch: 2.11.0+cu128 | built for CUDA: 12.8
cuda available: True                     <-- looks fine!
torch.AcceleratorError: CUDA error: no kernel image is available for execution on the device
```

`torch.cuda.is_available()` returns **True** and the first real kernel launch dies. The cu128
wheels ship only **CC ≥ 9.0**; the V100 is **sm_70**.

So the wheel is squeezed from both ends:

| Constraint | Direction | On RCI |
|---|---|---|
| CUDA runtime vs **driver** | too new a CUDA fails | cu130 needs driver ≥580; RCI has 575 |
| Bundled kernels vs **GPU arch** | too new a wheel fails | cu128 dropped sm_70; RCI has V100 |

**Measured working combination:**

```
torch 2.13.0+cu126
arch_list: ['sm_50','sm_60','sm_70','sm_75','sm_80','sm_86','sm_90']
MATMUL OK: True
```

`cu126` is the sweet spot: it needs only driver ≥525 and still ships sm_50–sm_90, so it covers
everything from Pascal to Hopper. Only Blackwell (sm_100+) needs cu128/cu130.

**Fixed in `install.sh`**: GPU mode now reads *both* the driver major **and** the GPU compute
capability, and installs torch *before* SB3 so the transitive dep can't pick the default index:

- CC ≥10 and driver ≥580 → default index
- CC ≥9 and driver ≥550 → cu128
- driver ≥525 → **cu126** (the default landing spot)
- driver <525 → cu118
- no working `nvidia-smi` (installing from a CPU node) → warn, use **cu126**

`RATSIM_TORCH_INDEX=<url|default>` overrides all of it. **CPU mode is untouched, so laptop
behaviour is unchanged.**

**Verify with a real kernel launch, never just `is_available()`:**

```bash
python -c "import torch; print(torch.cuda.get_arch_list()); \
           a=torch.randn(256,256,device='cuda'); print(bool(torch.isfinite((a@a).sum())))"
```

Note `torch.cuda.is_bf16_supported()` returns **True** on this V100 even though sm_70 has no
native bf16 tensor-core path — treat it as "will run", not "will run fast". Measured for
DreamerV3, bf16 is not merely fine but **required**: `float16` fails outright (§0.7).

**Always verify on a real GPU node rather than trusting the version table:**

```bash
srun -p gpufast --gres=gpu:1 --time=00:05:00 \
  python -c "import torch; print(torch.__version__, torch.cuda.is_available()); \
             print(torch.zeros(8,device='cuda').sum())"
```

This is the same class of bug as `CLAUDE.md`'s CUDA section, with a new twist: the danger is no
longer an *old* driver but a *new default wheel*. The rule that survives: **pin the CUDA variant
explicitly; never rely on the default index.**

### What the stack looks like after install (measured on a V100 node)

| Component | Version | GPU status |
|---|---|---|
| JAX / jaxlib | 0.4.33 (as pinned) | ✅ `jax.devices() → [CudaDevice(id=0)]` |
| dreamerv3 | imports OK | ✅ |
| torch | **2.13.0+cu126** | ✅ `cuda available: True`, real matmul verified on V100 |
| stable-baselines3 / sb3-contrib | 2.9.0 / 2.9.0 | n/a |
| gymnasium | 1.3.0 | n/a |
| ratsim | editable install, imports OK | n/a |

**Latent issue, not cluster-specific — `install.sh` dependency drift.** The dreamer venv pins
jax **0.4.33**, but unpinned `chex`/`optax` now resolve to versions that demand jax ≥0.7:

```
chex 0.1.92 requires jax>=0.7.0, but you have jax 0.4.33 which is incompatible.
optax 0.2.8 requires jax>=0.5.3, but you have jax 0.4.33 which is incompatible.
```

pip reports these as conflicts yet installs anyway; `jax`, `chex`, `optax`, `tensorflow` and
`dreamerv3` all import successfully and JAX sees the GPU, so it is **not** currently blocking.
But it is unpinned drift that will eventually break at runtime rather than import time, and
**a fresh install on the home PC would hit the identical conflict** — this is not something the
cluster introduced. Worth pinning `chex`/`optax` alongside jax in
`ratsim_experiments/methods/dreamerv3/requirements.txt`. There is also a benign-looking
`protobuf 5.29.6` vs `google-api-core` conflict from TensorFlow 2.19.

This also kills the portable-deb xvfb fallback for good: it needed glibc ≥2.38, and RCI has
2.17/2.28. The `Xvfb` module is the only sane route, and it exists.

### Exact module names (measured)

```bash
ml Python/3.12.3-GCCcore-13.3.0    # matches the laptop's 3.12.3
ml Xvfb/21.1.14-GCCcore-13.3.0     # DOES provide xvfb-run, not just Xvfb
ml git/2.45.1-GCCcore-13.3.0       # system git is 1.8.3.1 on login1
```

All three share the `GCCcore-13.3.0` toolchain — keep them consistent. Loading `Xvfb` also
pulls in `X11`, `Mesa`, `libglvnd`, and `libdrm`, which is exactly the set Unity `dlopen`s.
`CUDA/12.x` modules exist up to 13.0.2 but aren't needed: pip torch/JAX wheels bundle their own
CUDA runtime.

### Five traps that cost time here — all measured, all with confusing symptoms

1. **`ml` does not exist inside a SLURM job, and `#!/bin/bash -l` does not fix it.** `module` is
   a shell function, and `/etc/profile.d/lmod.sh` refuses to define it under a resource manager:
   ```sh
   # NOOP if running under known resource manager
   if [ ! -z "$SLURM_NODELIST" ];then return; fi
   ```
   So every `ml` line in a job script silently does nothing. It *appears* to work when you test
   via `srun --pty bash -i` or `bash -lc "srun ..."`, because there an interactive login shell
   already initialised Lmod and SLURM propagated the exported function. **Fix: source
   `rci_port_probes/rci_env.sh`**, which sets `MODULEPATH` and sources Lmod's own
   `$LMOD_PKG/init/bash` directly — doing what that profile script would have done past its guard.
2. **Never pipe `ml`.** `ml Python/... | tail -2` runs it in a subshell, so the environment
   change is discarded — it *looks* like it worked and the module isn't loaded. Traps 1 and 2
   each produced a confusingly "successful" probe run that was still on system Python 3.6.8.
3. **The venv cannot be used without its `ml Python/...` line**, and the error names neither
   modules nor venvs:
   ```
   python: error while loading shared libraries: libpython3.12.so.1.0:
           cannot open shared object file: No such file or directory
   ```
   The venv symlinks the module interpreter and needs the module's library path. This is the
   concrete failure behind §5's warning — **put the identical `ml` line in every job script.**
4. **The module Python does not run on `login1` at all:**
   ```
   python: /lib64/libc.so.6: version `GLIBC_2.27' not found
   ```
   It is built for the compute nodes' glibc 2.28; login1 has 2.17. So you cannot even sanity-
   check the venv from the login node — every venv operation belongs in a job. (`scheduler_status`
   is unaffected: it only reads files.)
5. **A batch job's `PATH` is `/usr/local/bin:/usr/bin` — no `/sbin`.** So `ss` (at `/sbin/ss`) is
   invisible, and `start_ratsim_headless.sh` aborted with `missing 'ss' (install iproute2)`.
   This is trap 1's shape again: `srun --pty bash -i` *does* have `/sbin` on `PATH`, so
   `check_cluster_node.sh` had cheerfully reported "ss present" for a binary the real job could
   not find. Fixed on both sides — `rci_env.sh` appends `/sbin:/usr/sbin`, the launcher also looks
   up `ss` by absolute path and falls back to a bash `/dev/tcp` probe, and the node probe now
   distinguishes "on PATH" from "exists but not on PATH".
   **General lesson: verify capabilities from a batch job, not an interactive one.** Interactive
   and batch shells differ in at least `PATH` and Lmod, and interactive is the more generous of
   the two — so every interactive probe is optimistic by construction.

### A non-Lmod trap: `nvidia-smi` writes failures to *stdout*

On a GPU-less node `nvidia-smi` is still installed at `/usr/bin/nvidia-smi`; it fails at runtime
and prints `NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.` **to
stdout**, so `2>/dev/null` does not suppress it. A naive
`DRV=$(nvidia-smi --query-gpu=driver_version ... 2>/dev/null)` captures that sentence as the
"version", and any later `(( DRV >= 580 ))` aborts the script under `set -e` with no message.
Guard with **both** an exit-status check and a numeric test:

```bash
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  DRV_MAJ="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)"
fi
[[ "$DRV_MAJ" =~ ^[0-9]+$ ]] || DRV_MAJ=""
```

### 0.6 How fast is the cluster, actually? (measured 2026-08-06)

**A single run is ~3× slower on the cluster than on the laptop.** The cluster's value here is
parallelism and wall-clock limits, not per-run speed.

Identical config both sides — `method=ppo world=maze_default total_steps=20000 n_stages=1`:

| | laptop (Ryzen 7 PRO 7840U, Zen 4) | RCI `n23`/`n27` (Xeon Gold 6150 @ 2.70 GHz, 2017 Skylake-SP) |
|---|---|---|
| overall `fps` | **816** | **232** (4 CPUs) → **280** (16 CPUs) |
| `rollout_fps` | ~1000 | ~290 (4) → ~350 (16) |
| 20k steps, wall | **25 s** | 88 s (4) → 73 s (16) |

Raw sim stepping, no learning (`python -m ratsim.fps_test --world maze_default --max-steps 3000`):

| CPUs allocated | mean FPS |
|---|---|
| 1 | 386 |
| 4 | 1203 |
| 8 | 1672 |
| 16 | 1877 |
| laptop (16 threads) | **2970** |

> ⚠️ **The cluster training numbers in this table were measured with OpenMP oversubscribed** (see
> the thread trap below) and are a *floor*, not a fair comparison. **Corrected figure: a single
> cluster run with 4 threads reaches fps 442**, against **814** on the laptop — so the laptop is
> ~1.8× faster per run, not ~3×. Raw `fps_test` figures are unaffected; Unity threads itself.
>
> The laptop's 814 also depends on this session's xvfb switch: the same run on the old
> `DISPLAY=:99` rendering path gives **355** (opt_seconds 0.72 vs 0.52), i.e. **xvfb is worth
> ~2.3× on training**, not just on raw stepping. Both sides of the comparison use xvfb.

Two things to take from this:

- **Never let a job default to 1 CPU.** `sbatch --wrap` without `--cpus-per-task` gives you one,
  and that alone costs **3×** on raw stepping (386 vs 1203). `train_job.sbatch` asks for 4.
- **Concurrent runs do not interfere.** One run alone gets **442** fps; each of two concurrent
  runs gets **434–452**. So the `cpu_slot: 16` / `needs.cpu_slot: 4` packing in `machines/rci.yaml`
  should scale to ~4×440 ≈ 1760 aggregate, roughly **2× the laptop's total** — the actual reason to
  use the cluster. (Verified at 2 concurrent; 4 is extrapolation.)
- **Raw stepping scales with CPUs but training barely does.** 4→16 CPUs is +56% on `fps_test` and
  only +20% on training throughput, so above ~4–8 cores the bottleneck is the single-threaded
  Python side (env wrapper, obs assembly, PPO forward), not Unity. Asking for 16 mostly buys
  queue wait. The V100 is irrelevant for PPO `n_envs=1` — it is not compute-bound.

The per-core gap is expected: the Xeon Gold 6150 is a 2017 server part optimised for core count,
and both Unity's single-env step and the Python loop are latency-bound single-thread work where a
modern laptop core wins. Don't read it as a misconfiguration.

### 🔥 The thread-oversubscription trap — worth more than every other perf fix here

**`/proc/cpuinfo` reports the whole machine (32 CPUs on `n23`) while SLURM cgroup-limits you to
`--cpus-per-task`.** PyTorch and OpenMP size their thread pools from the machine count, so a job
allocated 16 cores runs ~32 OpenMP threads *per process*.

Measured on two concurrent PPO runs in one 16-core allocation:

| | `opt_seconds` per iteration | `rollout_fps` | overall `fps` |
|---|---|---|---|
| 2 concurrent runs, default threads (~32 each) | **188** | ~300 | **20** |
| 2 concurrent runs, `OMP_NUM_THREADS=4` | **1.46** | ~355 | **434–452** |
| **1 run, `OMP_NUM_THREADS=16`** | **53.0** | ~531 | **39** |
| **1 run, `OMP_NUM_THREADS=4`** | **0.98** | ~562 | **442** |
| laptop, 1 run | ~0.5 | ~1000 | 814 |

**The last two rows are the important ones, and they are the same node and the same run — only
the thread count differs. 4 threads beats 16 by 11× on throughput and 54× on the optimize
phase.** So this is not merely "don't oversubscribe the cgroup": *more threads is worse, full
stop*, because PPO's optimize phase is many tiny matmuls where thread synchronisation dominates
the arithmetic. `rci_env.sh` therefore defaults to **4**, not `$SLURM_CPUS_PER_TASK` — an earlier
version of this fix scaled with the allocation and would have left a 16-core single-run job at
fps 39. Raise it only for a method measured to benefit.

Note `rollout_fps` was ~531–562 in *both* thread settings: Unity does its own threading and is
indifferent, so this only ever shows up in the optimize phase.

**~130× on `opt_seconds`, ~22× on throughput.** Note how it hides: `rollout_fps` stays *perfectly
healthy* because Unity does its own threading, and only the optimize phase collapses — so the
symptom reads as "the cluster's CPU is slow" rather than "we misconfigured threads". It also means
every earlier single-run timing in this doc was quietly paying some of this cost.

Fixed in `rci_env.sh`, which defaults `OMP_NUM_THREADS`/`MKL`/`OPENBLAS`/`NUMEXPR` to
`$SLURM_CPUS_PER_TASK`. **A job running several trainings at once must lower it further to its
per-run share** — `scheduler_job.sbatch` sets 4 to match `needs.cpu_slot` in `machines/rci.yaml`.
`needs.cpu_slot` is only scheduler bookkeeping and does *not* constrain a child's thread count, so
these two must be kept in sync by hand.

**Watch out:** these are `gpufast` nodes (`n23`, `n27`, `n28`, all the same Xeon). Nodes are
heterogeneous — re-measure on `gpu`/`gpulong` before planning long runs there.

### 0.7 DreamerV3 on the V100 — measured (2026-08-06)

`gpufast` node `n21`, `world=maze_default`, 10 000 steps, 4 CPUs, one variable changed:

| `method.jax.compute_dtype` | outcome | `fps/train` | `fps/policy` |
|---|---|---|---|
| **`bfloat16`** (dreamerv3 default) | ✅ exit 0 | **2039.8** | **67.3** |
| `float16` | ❌ **exit 1** after 106 s | — | — |

`float16` dies with:

```
TypeError: Argument types differ from the types for which this computation was compiled.
```

**So "Volta has no native bf16, therefore use float16" is wrong, and this doc asserted it for a
while.** Lacking bf16 *tensor cores* doesn't stop XLA emulating the dtype correctly, and
dreamerv3's graph evidently pins bfloat16 somewhere a `float16` compute_dtype then contradicts.
`machines/rci.yaml` originally carried that override and would have broken **every** dreamer run
on the cluster. It now sets only `jax.platform: cuda`.

#### What `train_ratio` actually means, and why the two rates aren't independent

`train_ratio` is **replayed timesteps per environment step**. From
`dreamerv3/embodied/run/train.py`:

```python
batch_steps  = batch_size * batch_length          # 8 * 48 = 384
should_train = Ratio(train_ratio / batch_steps)   # 32/384 = 1/12
```

So with our settings (`train_dreamerv3.py:207` sets `run.train_ratio: 32.0`) a gradient step fires
**every 12 env steps**, each replaying 384 timesteps. `train_fps.step(batch_steps)` counts
*replayed* timesteps, so the two logged rates are locked together:

> **`fps/train` ≈ `train_ratio` × `fps/policy`** — measured 67.3 × 32 ≈ 2154 against a logged
> 2039.8. They are one number reported two ways, not two independent measurements.

**Planning numbers that follow.** `fps/policy` 67 against PPO's 442 is expected rather than
alarming — dreamer does ~32× the compute per env step. But in wall-clock it means:

| | |
|---|---|
| dreamer vs PPO, per env step | **~6.6× slower** |
| 1M dreamer steps | **~4.1 h** |

**So `gpufast` (4 h) cannot finish a single 1M-step dreamer run.** Use `gpu` (1 day), `gpulong`
(3 days) or `gpuextralong` (21 days), and remember `needs.gpu: 1` caps dreamer at one concurrent
run regardless of spare cores — see the scaling options in phase 3.

**Not measured:** any laptop dreamer baseline for comparison, and whether `OMP_NUM_THREADS=4`
(inherited from the PPO finding) is right for dreamer — its compute is JAX-on-GPU, so the
tiny-matmul thread argument that produced that number may not apply at all.

### 0.8 ⚠️ Cross-machine comparability — read this before putting curves in a paper

**Nothing in the cluster port changes what the agent learns, with one exception, and the exception
is easy to miss.**

Safe (throughput only, no effect on learning):
- **xvfb vs `DISPLAY=:99`** — only the render path. Lidar is `Physics.Raycast`
  (`SemanticLidarSensor.cs`) and was verified structurally identical under both (§1).
- **`OMP_NUM_THREADS`** — reorders float reductions; no systematic bias.
- **Ports, pidfiles, rundirs** — inert.
- **`method.device: cpu` vs `cuda`, and compute dtype** — can shift low-order bits, but no
  systematic direction.

**NOT safe: `n_envs` differs between machine configs.**

| config | ppo | dreamer |
|---|---|---|
| `default.yaml` | **4** | 4 |
| `gpu_example.yaml` | 8 | 1 |
| `rci.yaml` (new) | **1** | 1 |

For SB3 PPO the rollout buffer is `n_steps × n_envs`, so `n_envs` changes how much data each
update sees and therefore the number of updates for a fixed `total_steps`, and the gradient noise.
**A laptop curve at `n_envs=4` is not comparable to a cluster curve at `n_envs=1`.**

This is by design, and the design is in tension with what you want. `scheduler/config.py` puts
`n_envs` in the machine config deliberately — *"a 'what does this box have capacity for' question,
not 'what's the experiment about' question"*. That is right for throughput and wrong for
cross-machine curves.

**Note this predates the cluster port.** `n_envs` has been a `MethodProfile` field since scheduler
V1 (`1076e14`); `default.yaml` and `gpu_example.yaml` already disagreed with each other. Adding
`rci.yaml` as the third machine is what made the consequence visible.

There is direct evidence the placement drifts. `gpu_example.yaml`'s own comment block says
*"recurrent_ppo stays at 1"* and *"PPO and dreamer both vectorize to 4 envs"*, while the values
beneath it are **8, 8 and 1**. Nothing ties a machine file to the experiment it will run, so its
prose rots independently of its values — and a silently wrong `n_envs` produces a plausible curve,
not an error.

#### ✅ DECIDED (user, 2026-08-07) — `n_envs` moves to the method / experiment def

Same value on every machine. The split:

- **The experiment def owns `n_envs`**, per method — versioned with the experiment, identical
  everywhere it runs.
- **The machine config owns capacity only** (`needs.cpu_slot`, `resources`). A box that cannot
  afford the def's `n_envs` runs **fewer concurrent runs**, not different envs. Deliberately *no*
  per-machine override: an override reintroduces exactly the hazard being removed, and a paper run
  must not be able to silently pick up a different rollout size.
- `n_envs` stays in `RESERVED_ARGS` — the scheduler still owns the CLI flag, it just sources the
  value from the def instead of the profile.
- Keep the existing `n_envs > 10` rejection (`PortAllocator` window is 10), and **add** a validation
  warning when a run's granted `cpu_slot` is below its `n_envs`, since each Unity env is roughly a
  core. On `rci.yaml` (`cpu_slot: 4`) any `n_envs > 4` should warn.

**Implemented 2026-08-10.** As built, the warning threshold is `4 × n_envs`, not `1 × n_envs` —
§0.9 had already measured that one env needs ~4 *threads*, not one, so the looser rule written above
would have passed the exact 4-thread configuration that ran at 17 fps. `cpu_slot` is now documented
as counting threads in all three machine configs, which is what makes the check comparable across
them. Touched: `experiment_defs.py` (`DEFAULT_N_ENVS`, `MethodSpec.n_envs`, snapshot),
`scheduler/config.py` (profile field removed + migration error + warning),
`scheduler/scheduler.py` (dispatch and `--use-port-9000` read the def), all three
`machines/*.yaml`, `scheduler/README.md`.

#### ✅ The value: **`n_envs = 4`**, read off the existing runs (2026-08-07)

Not a guess — the scheduler passes `n_envs=<n>` on the dispatched command line
(`scheduler/scheduler.py:559`), so every run's `scheduler_logs/stage_*.log` records what it used.
Scanning `results/experiments/*/runs/*/scheduler_logs/`:

**The whole current wells/ortho line — the paper work — is consistently `n_envs=4`, for both
methods:**

```
ortho_wells_adaptive_nohomeprime                  dreamer 4 · ppo 4
ortho_wells_adaptive_nohomeprime_bl256            dreamer 4
ortho_wells_adaptive_nohomeprime_dreamer_ladder   dreamer 4
ortho_wells_adaptive_no_explore_reward            dreamer 4 · ppo 4
wellmaze_cue_5x5 / wellmaze_cue_test_smol_model   dreamer 4
```

So pin **4 for every method, including dreamer.** This overrides the "dreamer should sit at 1 for
leak management" reasoning inherited from `gpu_example.yaml`: the actual dreamer runs all used 4,
and matching them matters more than slowing the leak, which the 30 GB RAM-kill already handles.

**The older experiments are inconsistent, and one of them is an ablation.** This is the failure
mode the move fixes, and it has already happened:

```
gps_ablation_5house:
  no_gps__dreamer__seed1    stage_0   n_envs=4
  no_gps__dreamer__seed2    stage_0   n_envs=4
  with_gps__dreamer__seed0  stage_0   n_envs=4
  with_gps__dreamer__seed0  stage_10  n_envs=8   <-- changed MID-RUN
```

One arm switched to 8 partway through while the other stayed at 4, so those two arms are not
matched. `memory_orthomaze` ppo seed0 shows both 8 and 4, and `wellmaze_test` / `wellmaze_cue_test`
show 1 and 4 — re-dispatches picking up a machine config that had changed underneath them. A wrong
`n_envs` produces a plausible curve, never an error.

Secondary, same family: `step_multiplier`, `total_steps`, `n_stages` and `metaseed` all come from
the experiment def rather than the machine config, so those are already machine-independent.

### 0.9 What `n_envs=4` actually costs — measured 2026-08-07

`n_envs=4` had **never been run inside a SLURM job** before this. It works: four Unity instances,
four distinct ports from the job's own window (`[9241, 9242, 9243, 9244]`), clean exit. But the
throughput story reverses depending on how many cores the run has, and one data point is not enough
to see it.

Identical run throughout — `method=ppo world=maze_default total_steps=20000 n_stages=1`, `OMP=4`:

| | `n_envs=1` | `n_envs=4` | | wall at `n_envs=4` |
|---|---|---|---|---|
| cluster, **4 cores** (n13) | 135 | **17** | 8× **loss** | **2037 s** |
| cluster, **8 cores** (n02) | 369 | 251 | 1.5× loss | 143 s |
| cluster, **16 cores** (n02) | 360 | **688** | 1.9× **gain** | 76 s |
| laptop, 16 threads | 685 | 581 | 1.2× loss | 61 s |

**There is a threshold at roughly 4 cores per env.** Below it the vectorized version does not
degrade gracefully, it collapses: at 4 cores a 20k-step run takes **34 minutes** instead of 76
seconds, because four Unity processes plus the trainer are fighting over four cores
(`opt_seconds` 67.1 against 4.29 at 16 cores). Above the threshold, vectorization pays properly.
Anyone who measures this at one core count will draw the opposite conclusion to anyone who measures
it at another — the 8-core point alone says "n_envs=4 is slower", and it is wrong.

⚠️ **Node variance is large at low core counts, so don't read these to two significant figures.**
The same 4-core `n_envs=1` run gave **135 fps on n13** and **175 fps on n06**. `cpufast` nodes are
shared and the cgroup caps CPUs but not memory bandwidth. The 16-core points are steadier; the
4-core row is only reliable as "catastrophic".

`opt_seconds` rises ~4–5× everywhere (0.99 → 4.79 at 8 cores, 0.50 → 2.62 on the laptop). That part
is inherent — the rollout buffer is `n_steps × n_envs`, so each update does 4× the work and there
are 4× fewer of them. **That is the part that changes the learning curve**, and the whole reason
the value has to be pinned rather than chosen per box.

#### Dreamer does NOT behave like PPO here

Same question, 10k steps, **16 cores + 1 V100** (so above the threshold), `n21`/`n23`:

| dreamer | wall | peak process-tree RSS |
|---|---|---|
| `n_envs=1` | 198 s | 2742 MB |
| `n_envs=4` | **187 s** | 2483 MB |

**~6% on wall clock, against PPO's 1.9× at the same core count.** This is `train_ratio` (§0.7) seen
from the other side: dreamer replays 32 timesteps per env step, so the world-model update dominates
and collecting experience faster barely moves the total. It is the same fact as `fps/train 2039` vs
`fps/policy 67`.

> ⚠️ **This is a throughput result and nothing else. `n_envs` still changes what dreamer learns.**
> With 4 envs the replay buffer is filled from 4 independent episodes concurrently, so it holds more
> decorrelated trajectories and sampled batches are less correlated — a real difference in what the
> world model sees, invisible in any wall-clock number. Do not read "6%" as "n_envs doesn't matter
> for dreamer"; it makes pinning at 4 **more** important for dreamer, not less, because dreamer is
> the case where you pay the cost and get *only* the learning effect.

Sizing consequence, and it is the opposite of the tempting one: because dreamer gets no throughput
back, it must **still be given 16–18 cores** so its 4 envs are not starved. At 8 cores `n_envs=4`
measured *slower* than `n_envs=1` (228 s vs 203 s) — under-provisioning costs speed while the
learning difference persists. Four dreamers × 18 cores on one GPU node is the right shape.

Also measured: **peak RSS is only ~2.7 GB** for a 10k-step run, so the leak does not show at this
length and `--mem=48G` was wild overkill. The 30 GB `max_ram_gb` threshold is for long runs, not a
sizing guide for short ones.

> **Not captured: `fps/train` / `fps/policy` for this comparison.** dreamerv3 prints only a
> `Metrics filtered by: 'score|length|fps|ratio|…'` header to stdout and writes the values to
> `<run>/dreamer_logdir/metrics.jsonl` — which the probe script deleted along with the run dir.
> Read that jsonl before cleanup next time. Wall time answers the question here (~53 env-steps/s
> including startup, consistent with the 67 `fps/policy` measured in §0.7), so this did not justify
> another GPU job.

#### Concurrency: packing runs on top of vectorization does not work

Same 4-run PPO def, one 16-core allocation, scheduler dispatching:

| packing | cores per run | wall for all 4 runs |
|---|---|---|
| **2-wide** | 8 | **339 s** |
| **4-wide** | 4 | **1265 s** (3.7× worse) |
| (1 run at a time, 16 cores, from the table above) | 16 | ~304 s |

4-wide puts 16 Unity instances on 16 cores — precisely the starved regime. And 2-wide is no better
than running them sequentially with all 16 cores. **At `n_envs=4`, vectorization has already
consumed the parallelism; concurrent runs on the same cores buy nothing.** Concurrency only pays
once the *allocation* grows, which is the next point.

#### One scheduler job = one node, and that sets the concurrency ceiling

The scheduler dispatches runs with plain `subprocess.Popen` (`scheduler/scheduler.py:609`) — no
`srun` — so every child lands on whichever node the scheduler itself got. A single job therefore
cannot exceed one node's cores, no matter what `--cpus-per-task` asks for:

| | cores | concurrent runs at ~16–18 cores each |
|---|---|---|
| CPU node (`n[01-20,33]`) | 48 | **3 PPO runs** → ~2060 fps aggregate |
| GPU node (`n[21-32]`) | 72 + 4× V100 | **4 dreamer runs** |

PPO is the comfortable case: it needs no GPU, so it belongs on a plain `cpu` node and strands
nothing. Three concurrent runs is ~3.5× the laptop's single-run throughput.

Going wider means several jobs, each with its own scheduler — the Option-2 trade-off (more
parallelism, bookkeeping split across schedulers). The `cpu` partition's 200-CPU aggregate cap
allows roughly 4 such jobs, i.e. ~12 concurrent PPO runs.

#### ✅ 7-wide PPO on one AMD node — verified (job 11318081, `a10`, 112 threads)

7 seeds × 100k steps, `n_envs=4`, one scheduler, one job. **7/7 completed, no retries, no errors.**

| seed | fps | rollout_fps | opt_seconds |
|---|---|---|---|
| 0 | 593 | 962 | 3.98 |
| 1 | 625 | 898 | 4.40 |
| 2 | 604 | 1030 | 3.77 |
| 3 | 607 | 1170 | 3.38 |
| 4 | 603 | 1230 | 3.35 |
| 5 | 607 | 931 | 4.08 |
| 6 | 614 | 1040 | 3.99 |
| **aggregate** | **4253** | | |

Each run holds **~608 fps against 723 solo (84%)**, so packing costs 16%. Against the laptop's
single-run **581**, that is **7.3×** by the fps metric, or **4.8×** by wall clock (700k steps in
251 s, startup included).

Two things this settles:
- **`opt_seconds` is unchanged under packing** (3.35–4.40 vs 3.52 solo), so the training phase does
  not degrade; the 16% is rollout and startup contention.
- **Each run still reaches ~1000 `rollout_fps` even 7-wide.** Seven processes each hitting the same
  ceiling independently confirms it is *per-process* Python overhead, not a shared node resource.

### 0.95 🔥 The thread-count trap, one layer below OMP — Unity spawns 266 threads per instance

**B2 failed twice before it worked, and neither failure was CPU, memory, or a slow boot.**

```
start_ratsim_headless.sh: fork: Resource temporarily unavailable
```

`EAGAIN` on `fork` — a **task-count limit**. Measured directly (job 11318080, `a01`):

| | |
|---|---|
| `ulimit -u` soft | **4096** ← the wall |
| `ulimit -u` hard | 4127387 |
| **threads per Unity instance** | **266**, exactly, every time |
| max instances under the soft limit | **15** (14 = 3728 tasks; 16 would exceed 4096) |

7 runs × 4 envs = 28 instances = **7448 threads** against a 4096 limit. It could never have worked,
and the mode of failure was misleading: the first attempt surfaced as
`subprocess.TimeoutExpired`, which sent this investigation after a boot-timeout red herring.
Boots were never slow — the fixed run logs `TCP server up on port 9440 after 1s` under full load.

**Fix: raise the soft limit in the job.** `ulimit -u 32768`. The hard limit is 4.1M, so this needs
no privileges — `RLIMIT_NPROC` is a fork-bomb guard, not a policy quota, and SLURM cgroups still
bound CPU and memory. ⚠️ **It is per-user per-node, not per-job**, so two of your jobs on one node
share the budget and each must raise it. Belongs in `rci_env.sh`, not in individual scripts.

**Done (2026-08-10)** — `rci_env.sh` raises it whenever `$SLURM_JOB_ID` is set, and warns rather
than failing if the raise is ever refused. Guarded on `$SLURM_JOB_ID` so sourcing the file on the
login node changes nothing there.

> 🔎 **[VERIFY] 266 threads per instance is Unity sizing its job-worker pool from the node's 128
> CPUs, not from our cgroup allocation — the exact shape of the `OMP_NUM_THREADS` trap (§0.6), one
> layer down.** At `n_envs=4` on 16 allocated threads that is **1064 Unity threads on 16 CPUs**.
> This is a strong candidate for *causing* the cores-per-env cliff in §0.9. Unity accepts
> `-job-worker-count N`; capping it could cut threads by an order of magnitude and reduce
> contention. **If that pans out, several numbers in this document improve** — the packing limit,
> the cliff, and possibly the per-run fps. Worth testing before any further capacity work.

Also fixed while chasing this: the Unity boot wait was hardcoded to 30 s in **two** places that must
agree — the shell loop in `start_ratsim_headless.sh` and `_spawn_via_script`'s `timeout_s`. If the
Python one is shorter it kills the script mid-wait and a timeout reads as a crash. Both now read
**`RATSIM_BOOT_TIMEOUT`**, defaulting to 30 so laptop and single-run behaviour are unchanged.

#### The throughput price of comparable curves

The old sizing (`n_envs=1`, 4-wide, `cpu_slot: 16` / `needs.cpu_slot: 4`) reached ~1700 fps
aggregate on 16 cores. `n_envs=4` on the same 16 cores gives **688**. So pinning at 4 costs roughly
**2.5× aggregate throughput** unless the allocation grows with it. That is the real price of
cross-machine comparability, and it is worth paying knowingly rather than discovering later.

The fix is cores, not concurrency. With `n_envs=4`:
- **one PPO run wants ~16 cores**, so 4 concurrent runs want ~64. `cpufast` allows 700 CPUs, so
  that is an easy ask for CPU-only PPO.
- **a whole GPU node is 72 cores + 4 V100** = 4 dreamer runs at 18 cores each. That is exactly the
  fair-share ratio *and* exactly the Option-1 multi-GPU job shape. The pieces line up.

⚠️ `machines/rci.yaml` is still sized for the old world (`cpu_slot: 16`, `needs.cpu_slot: 4`,
`n_envs: 1`). Leaving `needs.cpu_slot: 4` while moving to `n_envs=4` produces the 4-wide case
above — the worst configuration measured. **Re-size it in the same change that pins `n_envs`.**

#### A harness trap worth recording: shared `run_folder` across jobs

Two copies of the sweep submitted together (`11317603`, `11317604`) both used
`run_folder=_tmp_nenvs_<n>`. `results/` is on shared storage, so the faster job's cleanup
`rm -rf`'d the slower job's tensorboard directory mid-run and killed both its legs with
`FileNotFoundError` on `events.out.tfevents...`. Different nodes, so no CPU interference and the
throughput numbers survived, but the runs were scrapped and re-run.

**Same shape as the port-collision bug of phase 2, one layer up:** anything derived from a name
rather than from `$SLURM_JOB_ID` collides as soon as two jobs run at once. `run_folder` now
includes `$SLURM_JOB_ID` in `nenvs_perf.sbatch`. Real experiments are not exposed — the scheduler
derives per-run folders from the experiment and run id — but any ad-hoc probe script is.

---

## 1. The headless-rendering question — MEASURED (Phase 0a done, 2026-08-06)

> **This section was rewritten after actually running the test. The original conclusion
> — "`-nographics` alone removes the X dependency" — is FALSE for our build.**
> Tested against `ForagerSimBuildV1` (Unity 6000.1.8f1) on Ubuntu 24.04.

**Our current headless approach cannot work on RCI.** `ratsim/scripts/setup_headless_display.sh`
needs `apt-get`, writes `/etc/X11/xorg-ratsim.conf`, edits `/etc/X11/Xwrapper.config`, and
installs a systemd unit running `Xorg :99`. All of that requires root and a persistent machine.
Compute nodes give you neither. **This script is dead on the cluster — do not try to adapt it.**

### Measured results

Each variant launched with the port verified free beforehand, and "success" defined as *the
process listening on the port is a member of the process group we launched* (a naive
`ss | grep :9000` check produces false positives — see the methodology warning below).

| Variant | Result | Boot time |
|---|---|---|
| `DISPLAY=:99 <bin>` (GUI baseline) | SUCCESS | 3–4s |
| `<bin> -batchmode` (X present) | SUCCESS | 1s |
| `<bin> -batchmode -nographics`, **no X at all** | **SEGFAULT, 3/3 deterministic** | dies in 2s |
| `DISPLAY=:77 <bin> -batchmode -nographics` (var set, **no server** behind it) | **SEGFAULT** | dies in 2s |
| `DISPLAY=:99 <bin> -batchmode -nographics` (real server on `:99`) | SUCCESS | 1s |
| `xvfb-run -a <bin> -batchmode -nographics` | **SUCCESS, 3/3** | 1s |
| `xvfb-run -a <bin>` (no nographics) | SUCCESS | 3s |

So: **`-nographics` still requires a real, running X server.** Unity gets impressively far
without one — the log shows `Selected window backend: (null)`, `Forcing GfxDevice: Null`,
`NullGfxDevice`, and PhysX initialising cleanly (`Threading Mode: Multi-Threaded`) — then
segfaults in `PlayerMain` right after shader-fallback warnings. This is a Unity-side
limitation, not something in our scene code.

**Setting `DISPLAY` is not sufficient on its own** — pointing it at a display number with no
server behind it segfaults exactly as an unset `DISPLAY` does. `DISPLAY` is an address, not a
switch. This rules out any "just export DISPLAY in the sbatch script" shortcut: something must
actually be serving that display, and on a compute node the only thing that can (without root)
is an `Xvfb` you start yourself. `xvfb-run` does precisely that — it launches a real X server
that renders to memory, picks a free display number, exports `DISPLAY` for its child, and tears
the server down afterwards.

### The recommendation: `xvfb-run -a <bin> -batchmode -nographics`

This is the cluster path, and it is *still* a good outcome — `xvfb-run` is entirely
userspace. **No root, no systemd, no persistent service, nothing to ask the admin for.**
It supersedes `setup_headless_display.sh` on the laptop too.

**It is also substantially faster.** Same world (`maze_memorymaze_11x11_wells`), same agent
(`sphereagent_2d_lidar_wells`), 3000 steps × 3 reps via `ratsim.fps_test`:

| Mode | mean FPS (3 reps) |
|---|---|
| GUI baseline on `:99` | 1139 / 1275 / 1283 → **~1230** |
| `xvfb-run` + `-nographics` | 2709 / 2744 / 2765 → **~2740** |

**~2.2× throughput, and lower variance.** Worth adopting for local headless runs regardless of
the cluster.

**Functional correctness verified**, not just "it boots": a probe
(`rci_port_probes/lidar_check.py`) confirmed 19-beam scans with varied geometry under
`xvfb-run -nographics` — ranges 0–18.65 m, mean 9.5, only 5.3% of returns at max range,
0% at exactly zero. Structurally identical to the GUI baseline (same beam count, same
`maxRange`, same at-max fraction). Lidar is `Physics.Raycast`-based
(`Assets/Sensors/SemanticLidarSensor.cs`) and genuinely does not care about the render path.

### On RCI specifically (measured 2026-08-06)

**`Xvfb` is available as an Lmod module** — it is *not* on `PATH` by default, which is why a
first probe run on the login node reported it missing. `ml Xvfb` provides it, and the build
then boots headless on a compute node. `check_cluster_node.sh` now attempts this module load
itself before declaring failure.

**Do not assume `xvfb-run` exists.** It is a convenience shell script shipped by the Debian
package; module-built installs often provide only the `Xvfb` binary. Starting the server by
hand is equivalent and measured-working:

```bash
Xvfb :99 -screen 0 1024x768x24 -nolisten tcp &
DISPLAY=:99 "$BUILD" -batchmode -nographics -port "$PORT" -logFile "$LOG"
```

The probe handles both paths. Any launcher we write must too.

Fallbacks, now unnecessary but recorded in case a partition differs:

- **Singularity/Apptainer** container carrying xvfb + the build. RCI supports it with `--nv`;
  images under `/mnt/appl/singularity_images/`. Immune to glibc mismatch. Build the `.sif`
  where you have root and copy it over.
- **Portable no-root xvfb prefix.** Verified working on the laptop: `apt-get download` (needs
  no root) of `xvfb` + 20 dependencies = 3.5 MB compressed / 13 MB extracted; `dpkg -x` into a
  prefix, prepend to `PATH` and `LD_LIBRARY_PATH`, pass `-xkbdir <prefix>/usr/share/X11/xkb`.
  Unity booted against it in 1s. **Caveat:** Ubuntu 24.04 packages need **glibc ≥ 2.38**, while
  the Unity build itself only needs 2.14 — so the prefix is the fussier half and must be built
  from packages matching or older than the target node's distro. Not needed on RCI; don't
  bother unless a node lacks the module.

### Deleting the X dependency properly (optional, later)

Unity **6000.1.8f1** supports the **Dedicated Server** build target, which strips graphics at
build time — no window backend, no X server, no `Xvfb`, no segfault. That is the clean fix and
would remove this whole section's complexity from the cluster path.

Cost: no graphics device means no GPU context, so `RenderTexture`/`Camera.Render`/`ReadPixels`
cannot execute — `Assets/Sensors/RGBDSensor.cs` (the only camera-based sensor) stops producing
images. **But `-nographics` already has that consequence** (`Forcing GfxDevice: Null`), so if
we commit to `-nographics` on the cluster, a Dedicated Server build costs nothing extra.
Lidar is `Physics.Raycast`-based and unaffected either way.

If camera sensors *are* wanted on the cluster, drop `-nographics` and keep `Xvfb` — measured
working (3s boot), but rendering falls back to software (llvmpipe). The `VirtualGL` module,
also present on RCI, is the standard way to get GPU-accelerated headless GL if that becomes
a bottleneck. **[VERIFY]** — untested by us.

### Why the `ldd` check in the old Phase 0c was useless

The original plan proposed `ldd <bin> | grep "not found"` to detect missing X libraries.
**That check cannot detect this problem.** Measured on the real binary:

- `UnityPlayer.so` has only **9 DT_NEEDED entries**, all glibc-level (`libdl`, `libanl`, `libm`,
  `librt`, `libgcc_s`, `libpthread`, `libc`, vdso, loader). **Zero X11/GL entries.**
- `libX11.so.6`, `libXrandr.so.2`, `libXcursor.so.1`, `libXinerama.so.1`, `libXi.so.6`,
  `libX11-xcb.so`, `libGL.so.1`, `libEGL.so.1`, `libwayland-client.so.0` appear **only as
  strings** — they are `dlopen`ed at runtime.

`ldd` will therefore come back clean on a node that cannot run the binary. The meaningful test
is simply **launching it** (Phase 0c below). The upside of dlopen: missing libs surface as
soft failures Unity may tolerate, rather than a hard exec-time link error.

### Methodology warning for whoever re-runs this

The first pass of this test produced **entirely wrong results** and briefly "confirmed" the
false conclusion. Three traps, all worth avoiding:

1. **Leaked processes fake success.** `xvfb-run` is a shell script; killing it does *not* kill
   its Unity child. A leaked instance kept holding port 9000, so every later variant's
   `ss | grep :9000` reported SUCCESS regardless of whether that variant had crashed. The fix
   is `setsid` + kill the whole process group, and to verify the *listener's* pgid matches the
   launched pgid. `rci_port_probes/probe.sh` does this.
2. **Check the log, not just the port.** Runs reported "SUCCESS" while their logs contained
   `Caught fatal signal - signo:11`. Grep the log for that string as an independent signal.
3. Minor: `pkill -f ForagerSim...` matches the *invoking shell's own command line* and kills
   your own session. Use `pkill -x` on the comm name (`ForagerSimBuild`, 15-char capped).

---

## 2. Blocking issues, ranked

| # | Issue | Evidence | Severity |
|---|---|---|---|
| 1 | ~~**Hardcoded TCP ports**~~ — ✅ **FIXED in phase 2 (§2.5).** Job-derived port window, bind test, ownership check, retry. Verified: two trainings on one node took 9130/9131 and both completed | was `unity_launcher.py` `FRESH_PORT_BASE`, `scheduler/ports.py` `PortAllocator(start=9100)` | None |
| 2 | ~~**Pidfiles / logs in shared `/tmp`**~~ — ✅ **FIXED in phase 2 (§2.5).** Both the script and `unity_launcher.py` derive the dir from `$SLURM_JOB_ID` with identical precedence, and the pidfile is now an atomic reservation rather than a plain write | `unity_launcher.py` `_rundir()`; `start_ratsim_headless.sh` `RUNDIR=` / `claim_pidfile()` | None |
| 3 | **`install.sh` hardcodes `$HOME` paths** | `install.sh:20` `GIT_DIR="${HOME}/git"`, `:22` `VENV_DIR="${HOME}/ratvenv"` | Medium |
| 4 | **`install.sh` preflight demands apt** | `install.sh:53-63` errors out telling you to `sudo apt install python3-venv` | Medium (trivial fix) |
| 5 | ~~**CUDA / driver version matching**~~ — **RESOLVED for `gpufast`, but *not* with default wheels: driver 575 + V100 (sm_70) needs `cu126` explicitly.** Default PyPI torch (`cu130`) wants driver ≥580; `cu128` installs and reports `cuda available: True` but ships CC≥9.0 kernels only and dies at the first matmul. `install.sh` now picks the index from driver **and** compute capability. `jax[cuda12]` is fine as-is. Re-check on other partitions. The "V100 lacks bf16" sub-risk turned out not to apply to DreamerV3 (§0.7) | measured driver 575.51.03; `torch 2.13.0+cu126` arch_list `sm_50..sm_90`, matmul OK | Low |
| 6 | **`setup_headless_display.sh` unusable** | Needs root + systemd; see §1 | Medium (superseded by `xvfb-run`) |
| 6b | **`start_ratsim_headless.sh` hardwires `DISPLAY=:99`** | `:24` `DISPLAY_NUM=${DISPLAY_NUM:-:99}`, `:58` requires `/tmp/.X11-unix/X99` to pre-exist | Medium — must learn to spawn its own xvfb |
| 6c | **Nothing loads the `Xvfb` module** | Measured: `Xvfb` is not on `PATH` on RCI without `ml Xvfb` | Medium — every job script needs it |
| 7 | **Checkpoints only at stage boundaries** | `ratsim_experiments/train.py:612` saves `stage_{i}.zip` after each stage; no periodic `CheckpointCallback` | Low–Medium |
| 8 | **Results dir fixed inside the repo** | `scheduler/scheduler.py:57` `RESULTS_DIR = EXPERIMENTS_DIR / "results"` | Low |
| 9 | ~~**No internet on compute nodes**~~ — **RESOLVED: outbound internet works from the compute node.** Installs are not confined to `login1` | measured 2026-08-06 | None |
| 10 | **TensorBoard binds localhost** | `ratsim_experiments/tensorboard.sh` = `tensorboard --logdir results` | Low |

### #1 — Port collisions, in detail

Unity binds `127.0.0.1:<port>`. Ports are a machine-wide resource; two jobs on the same node
share the network namespace. Two failure modes with very different severity:

- **A stranger's job holds 9100.** You get a clean failure — `start_ratsim_headless.sh` runs
  `ss -tln | grep :$PORT`, sees it occupied, refuses; `unity_launcher.allocate_unity_instances`
  raises `RuntimeError`. Loud and safe, but confusing (the error suggests *you* left something
  running).
- **Two of your own jobs land on one node.** This is the dangerous one, and it compounds with #2.

**Fix:** derive the base port from the allocation instead of hardcoding. Cleanest is SLURM's
`--resv-ports` (exports `SLURM_STEP_RESV_PORTS`). Simpler and adequate: seed an offset from
`SLURM_JOB_ID` and probe upward for a free port, threading the result through
`allocate_unity_instances(base_port=...)` — which already accepts an explicit `base_port`, so
the plumbing exists.

### #2 — Pidfiles, in detail

A pidfile is a text file holding one number: a process ID. `start_ratsim_headless.sh` launches
Unity, gets PID e.g. `48213`, writes it to `/tmp/ratsim_9100.pid`; `stop_ratsim_headless.sh --port
9100` and `unity_launcher._kill_owned` read it back to find and kill that process from a different
shell.

On a shared node `/tmp/ratsim_9100.pid` is a collision point:

- **Different user** → file exists and is owned by them; your write fails (permission denied).
- **Same user, two jobs** → job B overwrites job A's pidfile. `_kill_owned` then targets the
  wrong PID, and because the UID matches the kill **succeeds**. Job A's Unity dies mid-training
  with no obvious cause. This is a silent-data-loss bug, not just an annoyance.

**Fix:** honour `$TMPDIR` (SLURM typically sets a per-job value) or interpolate `$SLURM_JOB_ID`
into the path. Same treatment for the default Unity log path.

### #5 — CUDA, in detail

Per `CLAUDE.md`: driver 550 → CUDA 12.4, 545 → 12.3, 525 → 12.0. Default PyPI torch currently
targets CUDA 12.6/12.8 (needs driver ≥560); for driver 550 use
`--index-url https://download.pytorch.org/whl/cu124`. JAX `0.4.33` with `jax[cuda12]` bundles
CUDA 12.3 → needs driver ≥545. Given RCI nodes are heterogeneous (Tesla V100 era *and* H200 era),
**check the driver per partition** and be prepared to keep separate venvs per node generation, or
to pin jobs to one partition.

---

## 3. What already works unchanged (don't break these)

- **`scheduler_status.py` works from the login node, live, while training runs elsewhere.**
  `cmd_status` (`scheduler/scheduler.py:944`) reads *only* the filesystem — `state.json`, run dirs,
  `.done` markers, `train_episodes.jsonl`. It never attaches to a process or opens a socket. Since
  `/mnt/personal` is mounted on all nodes, `python scheduler_status.py <exp> --watch 5` on `login1`
  monitors a job on `n27` with negligible load. **Condition:** results must live on shared storage,
  not node-local `/data/temporary/`.
- **Resume-by-default.** `scheduler_run.py run <exp>` resumes; `train.py` auto-resumes from
  `stage_{N-1}.zip` when `start_stage > 0` (`train.py:516-527`). Maps well onto requeued SLURM jobs
  *at stage granularity* — see #7.
- **Machine profiles.** `scheduler/machines/{default,gpu_example}.yaml` already parameterise
  `cpu_slot`, `n_envs`, and per-method device args, selected via `--machine` or
  `$RATSIM_SCHEDULER_MACHINE`. An `rci.yaml` drops straight in — no new mechanism needed.
- **Multi-instance port windows.** `PortAllocator` already hands out non-overlapping windows;
  it just needs a cluster-aware `start` rather than a new design.

---

## 4. Storage layout

| What | Where | Why |
|---|---|---|
| git repos, venvs | `/mnt/personal/$USER/git`, `/mnt/personal/$USER/ratvenv` | Wiki: don't use `$HOME` for data |
| Unity build | `/mnt/personal/$USER/` | ~GB; needed on every node |
| results / checkpoints | under `/mnt/personal/$USER/` | **must** be shared-FS so `scheduler_status` works from login node |
| Unity logs, pidfiles | `$TMPDIR` / node-local scratch | chatty, per-job, disposable |

Caveat worth measuring **[VERIFY]**: venvs are thousands of small files, and network filesystems
can be slow at that. If import time is painful, consider staging the venv to node-local scratch at
job start. Don't pre-optimise this — measure first.

Getting data there: `scp` from the laptop, or the SMB shares the wiki lists
(`\\login3.rci.cvut.cz\personal\%username`). The Unity build must be produced on a machine with
the Editor + licence (i.e. the laptop) and copied over — there is no Editor on the cluster.

---

## 5. Environment setup without root

`apt` is for system packages and needs root; you don't need it. Two userspace mechanisms:

**Modules (Lmod)** — the cluster's user-facing package manager:

```bash
module avail Python
ml Python/3.11.3-GCCcore-12.3.0     # exact name TBD [VERIFY]
python3 -m venv /mnt/personal/$USER/ratvenv/venv
source /mnt/personal/$USER/ratvenv/venv/bin/activate
pip install -r requirements.txt      # pip inside a venv never needs root
```

This also dissolves issue #4: `install.sh`'s preflight (`python3 -c "import venv, ensurepip"`)
fails on bare system Python but **passes** on a module-provided Python. The preflight message
should be reworded to mention `ml Python/...` as the cluster path rather than only `sudo apt`.

**Gotcha that will bite:** a venv symlinks the interpreter that created it. If you `ml Python/3.11.3`
to build the venv but omit that `ml` line from the sbatch script, the venv breaks confusingly.
**Put the identical `ml` line at the top of every job script.**

**Containers** — for anything modules can't cover, especially the X11 `.so` question in §1.
Singularity is supported; `--nv` for GPU.

---

## 6. Interactive vs batch

`srun -p gpufast --gres=gpu:1 --pty bash -i` gives a shell **on a compute node, in your terminal** —
stdout streams live, Ctrl-C works, scheduler output appears exactly as it does locally. Two limits:
interactive jobs are only permitted in `cpufast`/`gpufast`/`amdfast`/`amdgpufast` (4h cap), and if
SSH drops the job dies — so run `tmux` on the login node first, then `srun` from inside it.

For real runs use `sbatch`: submit, log out, stdout lands in a file, monitor with
`scheduler_status.py` from the login node.

Anatomy of the hardware-probe command, since it recurs below:

| Token | Meaning |
|---|---|
| `srun` | run a command via SLURM rather than locally |
| `-p gpufast` | in the `gpufast` partition |
| `--gres=gpu:1` | request 1 GPU. **Mandatory** on GPU partitions — the wiki shows `srun -p gpufast --pty bash -i` being rejected with *"gpufast partition allows only GPU jobs"* |
| `--pty` | attach a terminal so output returns to you |
| `nvidia-smi` | the command; prints GPU model + driver version, exits |

**On monitored usage:** a one-second `nvidia-smi` allocation is the canonical "what hardware did I
get" check and is invisible noise. What admins actually object to is **running compute on the login
node** (`login1` is shared by everyone — never run `train.py` there), holding a large `salloc`
idle for hours, or reserving 4 GPUs and using 1. Nothing in this roadmap is unusual cluster usage.

---

## 7. Roadmap

### Phase 0 — De-risk ✅ COMPLETE (2026-08-06)

**0a. ✅ Laptop.** Result in §1: bare `-nographics` segfaults deterministically; xvfb +
`-nographics` works, is functionally correct (lidar verified non-degenerate), and is ~2.2×
faster than GUI mode.

**0b/0c. ✅ Cluster.** `rci_port_probes/check_cluster_node.sh` run on a `gpufast` compute node:
tier 1 clean (no FAILs), tier 2 **`THIS NODE WORKS`** — the real build boots headless, opens its
TCP port, and logs no fatal signal. All measured values are recorded in **§0.5**.

Still outstanding from Phase 0, and cheap: **re-run the probe on `gpu` / `gpulong`** to confirm
driver and GPU model there (nodes are heterogeneous), and note whether `/mnt/job-$SLURM_JOB_ID`
exists — the probe now checks for it, which the original run predates.

**Do not use `ldd`** to re-check any of this — X libs are dlopen'd, so `ldd` is clean even where
the binary can't run (§1). The probe launches the binary for exactly this reason.

### Phase 0.9 — Agent access & operating rules

The remaining phases are being executed by an agent over SSH. Setup:

**Credential handling — the passphrase must never be pasted into the conversation.** The user
loads the key into `ssh-agent` themselves; the agent then uses the already-authenticated
connection and never possesses the secret:

```bash
# user runs this (in Claude Code, prefix with `!` so it runs in-session):
ssh-add ~/.ssh/<keyfile>          # prompts for passphrase, entered by the user only
ssh-add -l                        # confirm the key is loaded
```

Add a `ControlMaster` entry so the agent's many short `ssh` calls reuse one authenticated
connection instead of re-authenticating each time:

```
# ~/.ssh/config
Host rci
    HostName login1.rci.cvut.cz
    User <username>
    ControlMaster auto
    ControlPath ~/.ssh/cm-%r@%h:%p
    ControlPersist 8h
```

Then `ssh rci '<cmd>'` works non-interactively for the whole session.

**Operating rules for the agent — the admin monitors cluster activity:**

1. **Never run compute on a login node.** `login1` is for git, pip, and `scp` only.
2. **Never leave an allocation idle.** Prefer `srun ... <cmd>` (exits when the command does)
   over `srun --pty bash -i`. After anything interactive, verify with `squeue -u $USER`.
3. **Always pass `--time`**, sized to the actual task, so a hung job self-terminates.
4. **Never `scancel` a job the agent did not create** — the user may have real training
   running. Match on the JobID the agent recorded when submitting.
5. **Request only what's used** — `--gres=gpu:1`, not 4.
6. **Write only under `/mnt/personal/$USER` and `$HOME/git`.** No `sudo`, no system changes.
7. **Do not destroy experiment data**, on the cluster or the laptop. No `--restart` on any
   scheduler invocation; no deleting `results/`. Standing instruction from the user.
8. **Long runs are the user's call.** Submit short smoke tests freely; ask before submitting
   anything multi-hour or multi-job.

**Exit criteria:** `ssh rci true` succeeds non-interactively; `squeue -u $USER` is empty.

### Phase 1 — First single-GPU training job — ✅ COMPLETE (2026-08-06)

**Exit criteria met.** Job `11314758`, `gpufast` node `n23`, `COMPLETED` in **1:47**, exit `0:0`,
MaxRSS 1.19 GB:

```
Tesla V100-SXM2-32GB, 575.51.03
rundir:     /tmp/ratsim-job-11314758          # per-job, not shared /tmp
base_port:  9800                              # derived from $SLURM_JOB_ID
launching ForagerSimBuildV1.x86_64 under xvfb-run -nographics port=9800
TCP server up on port 9800 after 1s (pid 94041)
Total steps this invocation: 20000
```

Artifacts written to `/mnt/personal` and read back **from `login1`**: `checkpoints/final.zip`,
`checkpoints/stage_0.zip` + `stage_0.done`, `DONE`, `run_config.json`, 10 episodes in
`train_episodes.jsonl`, and a TensorBoard event file. No pidfiles left behind, no leaked Unity or
Xvfb, `squeue -u $USER` empty afterwards. Risk #8 ("results dir fixed inside the repo") is a
non-issue as long as the repo itself lives on `/mnt/personal`, which it does.


1. ✅ **DONE.** `install.sh` now honours `RATSIM_GIT_DIR` / `RATSIM_VENV_DIR`, defaulting to the
   old `$HOME` paths so laptop behaviour is unchanged.
2. ✅ **DONE, and upgraded to a real bug fix.** The preflight only checked that `venv` imports —
   which CentOS 7's Python **3.6.8 passes**, then fails later in confusing ways. Added a
   **version floor of 3.9** (matching `ratsim_wildfire_gym_env`'s `requires-python`) with an
   `ml Python/...` hint, and added the same hint to the venv-missing branch.
   Also fixed the symlink step to compare *resolved* paths: it was rewriting the repo's
   committed relative links (`../ratsim`) to absolute ones on every run, dirtying the working
   tree and breaking `git pull` on the cluster clone.
3. ✅ **DONE — on a `cpufast` COMPUTE node, not `login1`** (see §0.5: login1's glibc 2.17 makes
   pip resolve the wrong wheels). Layout: `/mnt/personal/$USER/git/{meta_ratsim,ratsim,...}` —
   the meta-repo must be a **sibling** of the others because its symlinks are relative.
   Venvs at `/mnt/personal/$USER/ratvenv`. Job script at `~/install_job.sh`, logs to
   `/mnt/personal/$USER/logs/install-<jobid>.out`.
4. ✅ **DONE — but not as predicted.** Plain `pip install torch` does *not* work here: default
   wheels are `cu130` (driver ≥580) and `cu128` drops sm_70, so the V100 fails at the first
   matmul despite `torch.cuda.is_available()` returning True. `install.sh` now derives the index
   URL from driver version **and** compute capability, landing on `cu126`. `jax[cuda12]` needed
   no change. Verified on a V100 node: `torch 2.13.0+cu126`, arch_list `sm_50..sm_90`, real
   matmul finite, `jax 0.4.33` sees `CudaDevice(id=0)`. Still to do: set DreamerV3 precision to
   ~~`float16`/`float32` for the V100~~ — **measured wrong, keep the bfloat16 default (§0.7).**
5. ✅ **DONE — `start_ratsim_headless.sh` has two display modes.**
   - **`xvfb`** — the script provides the display itself and runs Unity `-batchmode -nographics`.
     Prefers `xvfb-run -a`; if only the `Xvfb` binary exists (the RCI module case) it starts its
     own server on a free display and exports `DISPLAY`.
   - **`gfx`** — the old `DISPLAY=:99` path, unchanged, X-socket precondition **kept** (§1 proves
     it is a real requirement, not vestigial — it just must not apply when xvfb owns the display).
     Use this for camera/RGBD agents: `-nographics` gives Unity a null graphics device.
   - Selection: `--xvfb` / `--gfx`, or `RATSIM_XVFB=1` / `0`. **Default is auto: xvfb when an
     xvfb binary is present, else gfx** — so the ~2.2× speedup is the default on Linux, as
     suggested, while a box with no xvfb keeps working exactly as before.

   How the three known traps are handled, all re-verified on the laptop:
   - *xvfb-run's Unity child survives the wrapper.* The pidfile now records the **Unity** pid,
     found by matching `-port <PORT>` in `/proc/<pid>/cmdline` among `pgrep -x <comm>` hits —
     matching on the port, not just the comm, also stops two concurrent launches of the same
     build from stealing each other's pid. Measured: pidfile got the Unity pid `1520124`, not the
     `xvfb-run` wrapper `1520101`. Killing Unity is then sufficient: `xvfb-run` tears down its own
     Xvfb when the command exits (verified — no survivors).
   - *Stray `Xvfb` when killed ungracefully.* The own-Xvfb path passes **`-terminate`**, so the
     server exits when its last client goes away. Verified: `kill -9 <unity>` alone left no Xvfb.
     Belt-and-braces, the launcher writes `.xpid`/`.pgid` sidecars next to the pidfile and
     `stop_ratsim_headless.sh` reaps both (`--all` too), guarding the group kill with a check that
     the group still holds something recognisably ours.
   - *Display-number selection.* Judged by the **lock file** `/tmp/.X<n>-lock` (and `/proc`, not
     `kill -0`, since another user may own it), not by `/tmp/.X11-unix/X<n>`: a SIGKILLed Xvfb
     leaves its socket behind and those accumulate — there were 14 stale ones on the laptop from
     Phase 0a. Candidates are port-seeded so concurrent launches don't race, a leftover socket we
     own is cleared first so its reappearance is real evidence our server bound, and failure just
     moves to the next candidate.

   `unity_launcher.py` needed a matching change: pidfile paths now come from `_rundir()`
   (`$RATSIM_RUNDIR`, default `/tmp`) so the two cannot disagree, and `_kill_owned` reaps the
   sidecars. Verified end-to-end through `allocate_unity_instances(fresh=True)`: correct pid in
   the pidfile, TCP connect OK, and `atexit` left no process, no listener and no stale file.

   Functional check on the laptop through the new launcher, not just "it boots":
   `python -m ratsim.fps_test --world maze_memorymaze_11x11_wells --agent
   sphereagent_2d_lidar_wells --max-steps 3000` → **2970 FPS**, consistent with the ~2740
   measured in §1 for `xvfb-run -nographics` and ~2.4× the ~1230 GUI baseline.
6. ✅ **DONE — `rci_port_probes/train_job.sbatch`.** Sources `rci_env.sh` (Lmod + `Python`/`Xvfb`
   modules + `RATSIM_*` paths + per-job `$TMPDIR`), activates the venv, sets
   `RATSIM_RUNDIR="$TMPDIR"` so pidfiles never land in shared `/tmp`, forces `RATSIM_XVFB=1` so
   the job fails loudly if xvfb ever goes missing rather than silently trying `:99`, derives an
   interim `base_port` from `$SLURM_JOB_ID` (train.py accepts `base_port=`), and passes `"$@"`
   through to `train.py`. Defaults to `cpufast` with no GPU (PPO `n_envs=1` is faster on CPU);
   override per submission, e.g.
   `sbatch -p gpufast --gres=gpu:1 --time=00:30:00 train_job.sbatch method=ppo world=maze_default
   total_steps=20000 n_stages=1`. It reaps leftover Unity instances on exit via
   `stop_ratsim_headless.sh --all`, and captures `train.py`'s exit code without `set -e` skipping
   that cleanup.

**Exit criteria:** ✅ met — see the top of this phase for the measured run.

**One thing this run did NOT prove:** the first attempt (job `11314754`) failed in 8 s on
`missing 'ss'` — trap 5. That is worth remembering as the *class* of remaining risk: Phase 1 was
exercised by exactly one PPO run, so anything a batch job needs that only an interactive shell
provides is still unproven. `train.py`'s **`n_envs>1`** path and the **dreamer** venv have not
been run in a job at all yet.

### Phase 2 — Make it safe for concurrent jobs (~1–2 days)

7. ✅ **DONE — port allocation (#1).** See §2.5 below for what it took.
8. ✅ **DONE — pidfiles + logs (#2)**, in the same change as #7 as planned.
9. ✅ **DONE — `scheduler/machines/rci.yaml`** plus `rci_port_probes/scheduler_job.sbatch` (one job
   hosts the scheduler; deliberately not a job array, since the scheduler already owns port
   windows, resume and RAM kills). Validated against **all 29 experiment defs** locally, then run
   end-to-end on `gpufast`: 2 PPO seeds × 2 stages, **4/4 stage `.done` markers, `done runs: 2/2`**,
   and `scheduler_status.py _rci_smoke` rendered the full dashboard **from login2** — progress bars,
   per-method FPS, reward/pickup tables.

   Sizing is `cpu_slot: 16` with `needs.cpu_slot: 4`, i.e. pack four runs rather than give one run
   16 cores: a single run stops gaining at ~4 cores (§0.6), and two concurrent runs measured
   **~440 fps each** against **286** for one. `n_envs` is deliberately **1** everywhere (unlike
   `default.yaml`'s 4) — concurrent runs buy the same throughput here without `SubprocVecEnv`, and
   `n_envs>1` still has never been run in a job.

   Two things this uncovered:
   - **`rci_env.sh` must export `PPO_PYTHON_PATH` / `DREAMER_PYTHON_PATH`.** The scheduler resolves
     each method's interpreter through an *env var name* (`config.py DEFAULT_PYTHON_ENV`), not a
     path — that indirection is what lets one scheduler drive both venvs.
   - **DreamerV3's `jax.compute_dtype: bfloat16` default is correct here** — see §0.7. The
     `float16` override this config originally carried was measured to break dreamer entirely.

   **Also found and fixed a pre-existing scheduler race.** `PortAllocator` releases a window and
   can immediately re-hand those ports to the next stage, while the previous stage's Unity is still
   shutting down. Stage 1 of a run failed twice, on 9630 then 9620 — precisely that job's stage-0
   ports (`9100 + (11315212 % 90) * 10 = 9620`). A later scan confirmed both were free and nothing
   foreign was listening, and that `ip_local_port_range` is the standard 32768–60999, so neither a
   stranger nor ephemeral-port overlap was involved. The old TCP-connect check papered over this
   (a dying Unity refuses connections slightly before it releases the bind); the stricter bind test
   exposed it. Fix: on an *explicitly requested* port, wait up to 20 s for it to free rather than
   failing — failing kills the run, and moving to another port would desync the scheduler's
   accounting.

   Also: run the scheduler as `python -u`, or its progress sits in a pipe buffer and `tail -f` on
   the SLURM log looks dead for minutes.
10. ~~Confirm results land on shared storage (#8)~~ — **non-issue.** The repo lives on
    `/mnt/personal`, so `results/` is already on shared storage; phase 1's checkpoints were read
    back from a login node.

**Exit criteria:** ✅ met — two trainings on one node (`n26`, job `11315163`) took ports **9130**
and **9131**, both completed 12 000 steps, exit 0, and left no pidfiles or stray processes.
**Phase 2 is complete**; the scheduler now runs end-to-end on the cluster (item 9).

### 2.5 What the concurrency fix actually needed (2026-08-06)

The first attempt — derive a per-job base port, bind-test it before use — **was not enough, and
the co-location test caught it.** Both trainings reported
`TCP server up on port 9990 (pid 2770568)`: the *same port and the same pid*. Three distinct bugs,
each invisible until the one before it was fixed:

1. **A bind test is only a hint.** Two processes can both pass it and both launch. The bind has to
   be released before Unity can take the port, so the window can never be fully closed this way.
2. **The loser adopted the winner's Unity.** The launcher resolved "our" pid by matching
   `-port <PORT>` in `/proc/<pid>/cmdline` — which finds *anyone's* Unity on that port. So the
   loser wrote the winner's pid into its own pidfile, every ownership check downstream passed, and
   its cleanup then killed the winner's simulator. Fixed by identifying our Unity by **ancestry**
   (walk `/proc/<pid>/stat` ppids up to `$WRAPPER_PID`). Process group looked equivalent and was
   not: the pgid must be *read*, and right after `&` the wrapper may not have called `setsid` yet,
   so an early read returns the shell's old group and matches a stranger.
3. **Both launches wrote the same pidfile path.** Same `$RUNDIR` + same port ⇒ same filename. The
   loser clobbered the winner's recorded pid and then deleted the file on its own failure, leaving
   the winner's Unity orphaned with no cleanup handle — a leak that survived process exit. Fixed
   with an atomic reservation (`set -o noclobber`) written **before** launching.

   Sharp edge inside the fix: the placeholder must be a *number*. Writing the word `"claiming"`
   made the staleness check (which extracts digits) read it as "no pid recorded", so the loser
   deleted the winner's fresh claim and proceeded anyway. `echo $$` is alive by definition.

**The resulting design is two independent layers**, because neither covers the other's case:

| Layer | Separates | Doesn't help when |
|---|---|---|
| Job-derived base port + bind test | different jobs, before anything launches | two runs share one `$SLURM_JOB_ID` (same job) |
| Atomic pidfile claim in `$RUNDIR` | runs inside one job | separate jobs — their `$RUNDIR`s differ |
| Ownership check + retry on the next port | everything else, including another *user's* process | — this is the backstop |

**Known remaining limit:** a collision with a foreign process is *detected and retried*, not
prevented — `start_ratsim_headless.sh` refuses to report success unless our own Unity holds the
port, and `_spawn_first_free()` then moves on. Preventing it outright would need a node-wide lock,
which is not worth it.

Also fixed here: the scheduler ignores `--use-port-9000` under SLURM, and `allocate_unity_instances`
never reuses :9000 there. On a shared node an open :9000 is far more likely to be a stranger's
process than your Editor, and attaching would silently train against someone else's simulator.

### ✅ DECIDED — scaling dreamer beyond one GPU: **Option 1** (user, 2026-08-07)

**One job holding N GPUs, one scheduler inside it.** Chosen to keep the scheduler orchestrating:
its BFS-based stage selection and run prioritisation are the point of having it, and Option 2 would
throw them away. **Next action: build the `GpuAllocator` described below.** Two things to settle
first — the per-node GPU count (`sinfo -p gpu,gpulong -o '%P %G %N'`; the account's 6–8 GPU cap in
§0.55 is an *aggregate* limit, not a per-job one), and the `--mem` figure once N is known.

The three options, kept for the reasoning:

Today `dreamer` declares `needs.gpu: 1` against `resources.gpu: 1`, so **exactly one dreamer run
at a time**, no matter how many cores are free. At ~4.1 h per 1M steps (§0.7) that is the binding
constraint on any dreamer experiment. Three ways up, not mutually exclusive:

**Option 1 — N GPUs, one job, one scheduler.** Request `--gres=gpu:4`, set `resources.gpu: 4` in
`machines/rci.yaml`, and the existing `ResourceManager` will dispatch 4 dreamers.
- *Needs a code change first.* The scheduler assigns a **port window** per run and nothing else,
  so all four children would initialise on GPU 0 and contend or OOM. It needs a `GpuAllocator`
  mirroring `PortAllocator`, exporting `CUDA_VISIBLE_DEVICES=<idx>` into each dispatched child's
  environment. Same shape and size as the port change already made, and method-agnostic (works for
  torch as well as JAX). `scheduler.py` already builds the child env and passes `base_port`, so
  this is one more field on the same path.
- Also budget `--mem` for ~4 × `max_ram_gb: 30` because of dreamer's leak.
- ✅ Keeps resume markers, RAM kills, port windows and a single `scheduler_status` view.
- ❌ Capped by one node's GPU count; a 4-GPU ask queues longer than a 1-GPU ask.

**Option 2 — one job per run** (`train_job.sbatch` × N, or a SLURM job array).
- ✅ **Zero code change**, and the only option that spreads across *nodes*, so it is the only way
  past one node's GPU count.
- ❌ Loses the scheduler's resume/`.done` bookkeeping and RAM kills; each run queues independently;
  no single status view (`state.json` is per-scheduler).
- This is the job-array-vs-scheduler question already raised as phase 3 #12 — deciding one decides
  the other.

**Option 3 — one run across several GPUs** (`method.jax.train_devices=[0,1,…]`, supported by
dreamerv3).
- ✅ Makes a *single* run faster rather than running more of them.
- ❌ **Changes effective batch semantics, so it is the wrong tool for paper curves** — see §0.8.
  Fine for a one-off "can we finish this faster" run, not for anything plotted against other runs.

**Recommendation:** Option 1 for a multi-GPU node, falling back to Option 2 only when you want more
GPUs than one node has. Option 3 only outside the paper's comparison set.

### Phase 3 — Comfort + scale (~1–2 days, partly optional)

11. **Monitoring — ✅ use `login2`, not `login1`.** Measured 2026-08-06: **`login1` is the only
    CentOS 7 login node.** `login2`/`login4` are Rocky Linux 8.10 and `login3` is 8.8, all with
    **glibc 2.28** — the same as the compute nodes. So on `login2` the `Python/3.12.3` module and
    the venv both work, and **both tools run directly**: verified `scheduler_status.py --help` and
    TensorBoard 2.21.0 serving a real run's event file (HTTP 200 in 2 s, scalars API returned the
    actual `custom/avg_*` tags).

    Reaching them, per the sysadmin's `how_to_start`: *"Access with SSH password is not enabled on
    the nodes login[2-4]... Login with a password is allowed only on login1."* So login2–4 are
    **key-only, not restricted** — the wiki's own examples use `user@login2:~$` as the ordinary
    working prompt. Two ways in:
    - **Hop from login1**, which works with no setup: Warewulf provisions a per-user
      `~/.ssh/cluster` key (registered in your own `authorized_keys`) on the shared NFS home for
      exactly this, and sets `StrictHostKeyChecking=no` for `Host *`.
    - **Direct from the laptop**, which needs your key *offered* — `ssh-add ~/.ssh/id_ed25519`
      first, or an `IdentityFile` line for `login[2-4]` in your local `~/.ssh/config`. Without an
      agent it fails with `Permission denied (publickey)`, which looks like a permissions problem
      and is really just an unoffered key.

    ```bash
    ssh -N -L 6006:localhost:6006 -J musilto8@login1.rci.cvut.cz musilto8@login2.rci.cvut.cz
    # on login2:  . ~/rci_env.sh && ratsim_activate && cd $RATSIM_GIT_DIR/ratsim_experiments && ./tensorboard.sh
    ```

    **Don't leave TensorBoard up for days** — a login node is shared, and a persistent web server
    is the sort of thing an admin notices. For always-on monitoring prefer the rsync route below.

    On `login1` specifically, neither tool runs at all: Python 3.6.8 there cannot even parse
    `from __future__ import annotations` (`SyntaxError: future feature annotations is not
    defined`), and the 3.12 module Python needs glibc 2.28 (§0.5 trap 4). Alternatives that work
    from anywhere:
    - **Live progress:** `ssh rci 'tail -f /mnt/personal/$USER/logs/train-<jobid>.out'`. Pure file
      reading, no Python, no allocation — and SB3's table already prints `fps` / `rollout_fps`.
    - **TensorBoard / plots / `scheduler_status`:** pull the results to the laptop and run them
      locally. Verified working, and the whole results tree was 508 KB:
      ```bash
      rsync -az rci:/mnt/personal/$USER/git/ratsim_experiments/results/ ~/rci_results/
      ```
      Add `--include` filters for just `tensorboard/` if checkpoints get large.
    - If you really want `scheduler_status.py` on the cluster, it has to be inside a job:
      `srun -p cpufast --time=00:05:00 bash -c '. ~/rci_env.sh && ratsim_activate && cd
      $RATSIM_GIT_DIR/ratsim_experiments && python scheduler_status.py'` — a whole allocation to
      read files, so prefer the rsync.
    - Note `scheduler_status.py` only sees **scheduler-managed** runs under
      `results/experiments/<def>/runs/`. A direct `train.py` invocation (what `train_job.sbatch`
      does) writes to `results/<run_name>/` and will not appear there.
    A two-hop SSH forward to a TensorBoard running *in a job* also works, but it costs an
    allocation and the admin monitors activity — rsync is the better default.
12. SLURM job arrays for the run matrix, if the scheduler's own parallelism isn't the better fit.
    Decide deliberately: `scheduler_run.py` already manages port windows and resume, so it may be
    simpler to run *one* long `sbatch` job hosting the scheduler than N array tasks.
13. **Mid-stage checkpointing (#7)** — add an SB3 `CheckpointCallback` and teach `train.py` to
    resume mid-stage. Less urgent than it first appears: `gpuextralong` allows 21 days, which
    covers current run lengths. Worth doing eventually regardless of the cluster, for crash
    resilience.

---

## 8. Effort estimate

| Milestone | Estimate |
|---|---|
| Phase 0a (local) | ✅ done |
| Phase 0b/0c (cluster probes) | ✅ done — headless confirmed working on `gpufast` |
| First training job on RCI | ~1 day |
| Safe concurrent multi-run | +1–2 days |
| Comfortable day-to-day setup | +1–2 days |

The headless-rendering risk that motivated this investigation is **fully resolved and confirmed
on the cluster itself**. It did *not* resolve the way this doc originally predicted —
`-nographics` alone is not enough — but the working answer is fully userspace (`ml Xvfb`, no
root, nothing to ask the admin for) and comes with a ~2.2× speed bonus. Two further risks
evaporated on measurement: CUDA wheel matching (driver 575 takes stock wheels) and
compute-node internet.

What's left is SLURM plumbing — ports and pidfiles (Phase 2) — and it is mechanical. The two
issues that *were* confirmed live are exactly those two: foreign listeners already sat in our
default port range, and `$TMPDIR` is unset so the naive `/tmp/ratsim_<port>.pid` path is what
would actually run. Neither is hard; both must land before concurrent runs.

---

## 9. Open questions for whoever picks this up

- ~~**[VERIFY]** Does `-batchmode -nographics` work?~~ **ANSWERED**: not on its own (deterministic
  segfault without X). `xvfb-run -a` + `-nographics` works, is correct, and is ~2.2× faster. See §1.
- ~~**[VERIFY]** Is `xvfb` present on the compute nodes?~~ **ANSWERED: yes, as an Lmod module
  (`ml Xvfb`). Not on `PATH` by default. No container needed.** `xvfb-run` may be absent — handle
  both (§1).
- ~~**[VERIFY]** Does the build boot under xvfb on an actual compute node?~~ **ANSWERED: yes —
  tier 2 probe returned `THIS NODE WORKS` on `gpufast`.**
- ~~**[VERIFY]** Does SLURM set a per-job `$TMPDIR`?~~ **ANSWERED: no, `$TMPDIR` is unset.** Scope
  by `$SLURM_JOB_ID`; check whether `/mnt/job-$SLURM_JOB_ID` exists (probe now reports this).
- ~~**[VERIFY]** Do compute nodes have outbound internet?~~ **ANSWERED: yes.**
- ~~**[VERIFY]** Driver/CUDA version~~ — **ANSWERED for `gpufast`: V100-SXM2-32GB, driver
  575.51.03.** Still **[VERIFY]** on `gpu` / `gpulong`, and whether the H200 nodes the wiki
  advertises are reachable (they'd restore bf16 and change the precision advice).
- **[VERIFY]** Exact Lmod module names for Python / Xvfb / CUDA (`module spider <name>`).
- **[VERIFY]** Does GPU-accelerated headless GL work via the `VirtualGL` module? Only matters if
  camera sensors are wanted on the cluster (§1).
- **[DECIDE]** One long `sbatch` hosting our scheduler, vs. SLURM job arrays with one run each.
  Affects Phase 2 and 3 design; our scheduler's built-in port management and resume argue for
  the former.
- **[DECIDE]** Venv on `/mnt/personal` vs. staged to node-local scratch — measure import time first.

---

## Appendix — SLURM cheat sheet

```bash
sinfo                                   # partitions, time limits, node states
squeue -u $USER                         # my jobs
srun -p gpufast --gres=gpu:1 --pty bash -i    # interactive shell on a GPU node (4h cap)
sbatch job.sh                           # submit batch job
scancel <jobid>                         # cancel
```

Sketch of a job script (fill in after Phase 0):

```bash
#!/bin/bash
#SBATCH --partition=gpulong
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --output=/mnt/personal/%u/slurm-%j.out

ml Python/3.11.3-GCCcore-12.3.0        # MUST match the module used to create the venv
ml Xvfb                                # REQUIRED — Xvfb is not on PATH by default on RCI
source /mnt/personal/$USER/ratvenv/venv/bin/activate
export TMPDIR=/mnt/job-$SLURM_JOB_ID   # $TMPDIR is unset on RCI; keeps pidfiles/logs per-job
export RATSIM_UNITY_BIN=/mnt/personal/$USER/ForagerSimBuildV1/ForagerSimBuildV1.x86_64
export RATSIM_XVFB=1                   # launch headless under xvfb + -batchmode -nographics
cd /mnt/personal/$USER/git/ratsim_experiments
python train.py def=<rundef> method=ppo
```

Verified-good launch lines. Prefer the first; fall back to the second when `xvfb-run` is absent
(likely on RCI — the module may ship only the `Xvfb` binary):

```bash
# with the wrapper (what Phase 0a measured on the laptop)
xvfb-run -a "$RATSIM_UNITY_BIN" -batchmode -nographics -port "$PORT" -logFile "$LOG"

# without it — equivalent, also measured working
Xvfb :99 -screen 0 1024x768x24 -nolisten tcp &
DISPLAY=:99 "$RATSIM_UNITY_BIN" -batchmode -nographics -port "$PORT" -logFile "$LOG"
```

Monitoring, from `login1`, while the above runs elsewhere:

```bash
cd /mnt/personal/$USER/git/ratsim_experiments
python scheduler_status.py <exp> --watch 5
```
