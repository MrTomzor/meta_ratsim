# ORIGINAL INSTRUCTIONS FOR USING RCI CLUSTER
Please read also the instructions by the sysadmin in ~/Downloads/rci_how_to_start.html

# Porting ratsim to the RCI cluster (CTU) — analysis + roadmap

Handoff doc for running ratsim training on the **RCI cluster** (`login[1-4].rci.cvut.cz`),
a SLURM-scheduled HPC cluster at CTU. Written after reading the cluster's
`how_to_start` wiki page and auditing our stack against it.

**Epistemic status:** every claim about *our code* below was verified by reading the
source (file:line given). **§1 was verified empirically on the laptop (2026-08-06), and
it overturned this doc's original conclusion.** **Phase 0 is now COMPLETE — the probe has
been run on a real RCI compute node (2026-08-06) and the build boots headless there.**
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
| **login1 OS** | **CentOS 7, glibc 2.17**, git 1.8.3.1 | ⚠️ **Do NOT pip install here** — see below. |
| **compute node OS** | **glibc 2.28**, git 2.31.1 | Modern manylinux wheels work here. |
| `$TMPDIR` | **unset** | Must scope pidfiles/logs by `$SLURM_JOB_ID` ourselves — issue #2 confirmed live. |
| Ports 9000–9999 on that node | **2 listeners already present** | **Issue #1 confirmed real, not theoretical.** Foreign processes occupy our default range. |
| `ss`, Lmod | present | `start_ratsim_headless.sh` precondition satisfied. |

**V100 is sm_70 (Volta)** — no native bf16 tensor-core path, and importantly **new CUDA wheels
have started dropping this arch entirely** (see the CUDA section: cu128 ships CC ≥9.0 only).
This is the single most consequential hardware fact here. `torch.cuda.is_bf16_supported()`
nonetheless reports `True`, so measure rather than trust it before using bf16 in DreamerV3.
If RCI's advertised H200 nodes are reachable, most of this reverses — another reason to record
hardware per partition.

**Nodes are heterogeneous**: the above is one `gpufast` node. Re-run the probe on `gpu` /
`gpulong` before trusting driver and GPU model there.

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
native bf16 tensor-core path — so treat it as "will run", not "will run fast", and measure
before relying on bf16 for DreamerV3.

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

### Four traps that cost time here — all measured, all with confusing symptoms

1. **`ml` only exists in a login shell.** A `#!/bin/bash` script under `srun`/`sbatch` has no
   `module` function and every `ml` line silently does nothing. **Use `#!/bin/bash -l`.**
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
| 1 | **Hardcoded TCP ports** — ⚠️ **collision confirmed observed: 2 foreign listeners in 9000–9999 on the probed node** | `ratsim/ratsim/unity_launcher.py:40` `PERSISTENT_PORT = 9000`, `:46` `FRESH_PORT_BASE = 9100`; `ratsim_experiments/scheduler/ports.py:20` `PortAllocator(start=9100, window_size=10)` | High |
| 2 | **Pidfiles / logs in shared `/tmp`** — ⚠️ **`$TMPDIR` measured unset on RCI, so the naive path is what runs** | `unity_launcher.py:49` `PIDFILE_TEMPLATE = "/tmp/ratsim_{port}.pid"`; `start_ratsim_headless.sh:56` same path, default log `/tmp/ratsim_<port>.log` | High |
| 3 | **`install.sh` hardcodes `$HOME` paths** | `install.sh:20` `GIT_DIR="${HOME}/git"`, `:22` `VENV_DIR="${HOME}/ratvenv"` | Medium |
| 4 | **`install.sh` preflight demands apt** | `install.sh:53-63` errors out telling you to `sudo apt install python3-venv` | Medium (trivial fix) |
| 5 | ~~**CUDA / driver version matching**~~ — **RESOLVED for `gpufast`: driver 575 ⇒ default PyPI torch + `jax[cuda12]` both fine.** Re-check on other partitions. New sub-risk: V100 lacks bf16 (§0.5) | `CLAUDE.md` § "CUDA / driver compatibility"; driver measured 575.51.03 | Low |
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

### Phase 1 — First single-GPU training job (~1 day)

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
4. ~~Select torch/JAX wheels from the driver reading~~ — **settled: driver 575 ⇒ plain
   `pip install torch` and `jax[cuda12]`.** No index URLs needed. Verify with
   `python -c "import torch; print(torch.cuda.is_available())"` inside a GPU job, then set
   DreamerV3 precision to `float16`/`float32` for the V100 (§0.5).
5. Teach `start_ratsim_headless.sh` an xvfb mode (`RATSIM_XVFB=1` or `--xvfb`): wrap the launch
   in `xvfb-run -a` and add `-batchmode -nographics`. **Keep** the X-socket precondition check
   at `:58` for the existing `DISPLAY=:99` path — §1 confirms an X server really is required, so
   that check is correct, not vestigial; it just must not apply when xvfb provides the display.
   Two traps the script must handle, both hit during Phase 0a testing:
   - `xvfb-run` is a shell script — its Unity child **survives killing the wrapper**. The pidfile
     must record the *Unity* pid (`pgrep -x ForagerSimBuild`), or the launch must use `setsid`
     and the stop path must kill the process group. Otherwise `stop_ratsim_headless.sh` silently
     leaves an instance holding the port.
   - `xvfb-run -a` picks its own display number and leaves a stray `Xvfb` behind if killed
     ungracefully. The stop path should reap it.
   Given the measured ~2.2× speedup, consider making this the **default** on Linux rather than
   an opt-in.
   **Third trap, specific to RCI:** `xvfb-run` may not exist even when `Xvfb` does (§1). The
   script must fall back to starting `Xvfb` itself on a free display and exporting `DISPLAY`.
   `check_cluster_node.sh` already implements both branches — reuse that logic.
6. Write a minimal `sbatch` wrapper: `ml Python/...` **and `ml Xvfb`**, venv activate,
   `export TMPDIR=/mnt/job-$SLURM_JOB_ID` (or an `$SLURM_JOB_ID`-scoped dir — `$TMPDIR` is unset
   on RCI, §0.5), `export RATSIM_UNITY_BIN=...`, one `train.py` invocation. Run one short job
   end-to-end on `gpufast` with an explicit `--time`.

**Exit criteria:** one training run completes a stage on a compute node and its checkpoint +
`train_episodes.jsonl` are visible from `login1`.

### Phase 2 — Make it safe for concurrent jobs (~1–2 days)

7. **Port allocation (#1).** SLURM-aware base port in `unity_launcher.py` and
   `scheduler/ports.py` — `--resv-ports`/`SLURM_STEP_RESV_PORTS`, or `SLURM_JOB_ID`-seeded offset
   with free-port probing. Keep the laptop default at 9000/9100.
8. **Pidfiles + logs (#2).** `$TMPDIR`- or `$SLURM_JOB_ID`-scoped paths in `unity_launcher.py:49`
   and `start_ratsim_headless.sh:56`. **Do this in the same change as #7** — they share a root cause
   and fixing only one leaves the silent-kill bug live.
9. Add `scheduler/machines/rci.yaml` with RCI-appropriate `cpu_slot` / `n_envs` / device args.
10. Confirm results land on shared storage (#8) — add an env override for
    `scheduler.py:57 RESULTS_DIR` if the repo can't simply live on `/mnt/personal`.

**Exit criteria:** two jobs deliberately co-scheduled on one node both run to completion without
interfering.

### Phase 3 — Comfort + scale (~1–2 days, partly optional)

11. TensorBoard over the login node: two-hop SSH forward, or just rsync event files out.
    (`tensorboard.sh` needs no change if you forward properly.)
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
