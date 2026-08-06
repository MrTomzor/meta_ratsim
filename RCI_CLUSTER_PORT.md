# ORIGINAL INSTRUCTIONS FOR USING RCI CLUSTER
Please read also the instructions by the sysadmin in ~/Downloads/rci_how_to_start.html

# Porting ratsim to the RCI cluster (CTU) — analysis + roadmap

Handoff doc for running ratsim training on the **RCI cluster** (`login[1-4].rci.cvut.cz`),
a SLURM-scheduled HPC cluster at CTU. Written after reading the cluster's
`how_to_start` wiki page and auditing our stack against it.

**Epistemic status:** every claim about *our code* below was verified by reading the
source (file:line given). Every claim about *the cluster* comes from the wiki page only —
nothing has been run on RCI yet. Items marked **[VERIFY]** must be checked on the real
machine before relying on them. The wiki's example outputs are visibly stale (its
`nvidia-smi` sample shows driver 418 / CUDA 10.1, i.e. ~2019, while the wiki sidebar
advertises an "H200 nodes upgrade 2025"), so treat its specifics as illustrative, not current.

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

## 1. The headless-rendering question — resolved, and it's the easy part

The received wisdom (from a colleague who ported an **Unreal** sim) is that game-engine sims
must be made truly headless to run on a cluster, and that this is the hard part. The first
half is right; the second half does not apply to us.

**Our current headless approach cannot work on RCI.** `ratsim/scripts/setup_headless_display.sh`
needs `apt-get`, writes `/etc/X11/xorg-ratsim.conf`, edits `/etc/X11/Xwrapper.config`, and
installs a systemd unit running `Xorg :99`. All of that requires root and a persistent machine.
Compute nodes give you neither. **This script is dead on the cluster — do not try to adapt it.**

**But we almost certainly don't need an X server at all.** Audit of `ratsim_unity_project/RatsimUnityProject/Assets/`:

- 2D lidar is `Physics.Raycast`-based (`Assets/Sensors/SemanticLidarSensor.cs`) — pure physics,
  no render pipeline involvement.
- The **only** file in the entire `Assets/` tree referencing `RenderTexture`, `Camera.main`,
  or `ReadPixels` is `Assets/Sensors/RGBDSensor.cs`.
- `RGBDSensor` is attached only when an agent preset lists the `rgbd` sensor. The presets that
  do (`sphereagent_rgb.yaml`, `sphereagent_rgbd.yaml`) are **not referenced by any run def** in
  `ratsim_experiments/defs/` — every current def uses a `sphereagent_2d_lidar*` or
  `sphereagent_2d_fullsar` preset.

So for the current experiment set, `-batchmode -nographics` should work with no code changes.
`CLAUDE.md` already flags this as a "possible future improvement… worth trying" — it is now the
recommended primary path, and it is the **single highest-value thing to test first** because it
collapses the scariest-looking risk into a deleted dependency.

**Fallback ladder** (only if `-nographics` misbehaves):
1. `xvfb-run -a <bin> -port $PORT` — user-space virtual framebuffer. No root, no systemd,
   no persistent service. Needs `xvfb` present (module or container).
2. Singularity/Apptainer container carrying xvfb + the build. RCI explicitly supports Singularity
   with `--nv` for GPU passthrough (images live under `/mnt/appl/singularity_images/`).
   Build the `.sif` on a machine where you have root and copy it over.

### The one genuine `apt`-shaped risk

The Unity Linux player **dynamically links X11 libraries at load time**. `-nographics` removes the
need for an X *server*, but the `.so` files may still need to exist on the node. If they're absent
there is no `apt` to fix it. Cheap check from an interactive job (see §5):

```bash
ldd /mnt/personal/$USER/ForagerSimBuild.x86_64 | grep "not found"
```

Empty output → clear. Anything listed → find a module providing it, or go the container route
(which is the robust answer regardless).

---

## 2. Blocking issues, ranked

| # | Issue | Evidence | Severity |
|---|---|---|---|
| 1 | **Hardcoded TCP ports** | `ratsim/ratsim/unity_launcher.py:40` `PERSISTENT_PORT = 9000`, `:46` `FRESH_PORT_BASE = 9100`; `ratsim_experiments/scheduler/ports.py:20` `PortAllocator(start=9100, window_size=10)` | High |
| 2 | **Pidfiles / logs in shared `/tmp`** | `unity_launcher.py:49` `PIDFILE_TEMPLATE = "/tmp/ratsim_{port}.pid"`; `start_ratsim_headless.sh:56` same path, default log `/tmp/ratsim_<port>.log` | High |
| 3 | **`install.sh` hardcodes `$HOME` paths** | `install.sh:20` `GIT_DIR="${HOME}/git"`, `:22` `VENV_DIR="${HOME}/ratvenv"` | Medium |
| 4 | **`install.sh` preflight demands apt** | `install.sh:53-63` errors out telling you to `sudo apt install python3-venv` | Medium (trivial fix) |
| 5 | **CUDA / driver version matching** | `CLAUDE.md` § "CUDA / driver compatibility" — has bitten this project repeatedly | Medium |
| 6 | **`setup_headless_display.sh` unusable** | Needs root + systemd; see §1 | Medium (superseded by `-nographics`) |
| 7 | **Checkpoints only at stage boundaries** | `ratsim_experiments/train.py:612` saves `stage_{i}.zip` after each stage; no periodic `CheckpointCallback` | Low–Medium |
| 8 | **Results dir fixed inside the repo** | `scheduler/scheduler.py:57` `RESULTS_DIR = EXPERIMENTS_DIR / "results"` | Low |
| 9 | **No internet on compute nodes [VERIFY]** | Common on clusters; `install.sh` clones + pip-installs | Low (workflow constraint) |
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

### Phase 0 — De-risk (do first; ~1h total, mostly waiting)

**0a. Local, no cluster needed — the go/no-go test.**
```bash
<build>/ForagerSimBuild.x86_64 -batchmode -nographics -port 9500 -logFile /tmp/ng.log
# then point a lidar-preset run at 9500, e.g.
cd ratsim && python -m ratsim.fps_test --world <w> --agent sphereagent_2d_lidar --max-steps 200
```
Success = non-degenerate lidar scans and a clean log. This determines whether §1's happy path
holds or the fallback ladder is needed. **Do not start Phase 1 before this passes.**

**0b. On the cluster, from an interactive job:**
```bash
ssh username@login1.rci.cvut.cz            # password login only works on login1
srun -p gpufast --gres=gpu:1 --pty nvidia-smi
```
Record driver version + GPU model **per partition you intend to use**. Feeds the wheel selection
in Phase 1.

**0c. Copy the build over, then in an interactive job:**
```bash
ldd /mnt/personal/$USER/ForagerSimBuild.x86_64 | grep "not found"
module avail Python                # record exact module names
echo "$TMPDIR"                     # does SLURM set a per-job TMPDIR here?
```
Decides venv-vs-container and settles the `$TMPDIR` question for Phase 1.

**Exit criteria:** `-nographics` verdict known; driver versions recorded; `ldd` clean or container
decision made; Python module name known.

### Phase 1 — First single-GPU training job (~1 day)

1. `install.sh`: make `GIT_DIR` / `VENV_DIR` overridable via env vars (`:20`, `:22`), defaulting to
   today's `$HOME` paths so laptop behaviour is unchanged.
2. `install.sh`: reword the preflight failure (`:53-63`) to mention `ml Python/...` alongside the
   apt hint.
3. Install on **`login1`** (compute nodes may lack internet — issue #9). Target `/mnt/personal`.
4. Select torch/JAX wheels from the Phase 0b driver reading, per the `CLAUDE.md` compatibility table.
5. Drop `-batchmode -nographics` into `start_ratsim_headless.sh`, gated by a flag or env var
   (`RATSIM_NOGRAPHICS=1`) so laptop GUI debugging is unaffected. Bypass the X-socket precondition
   check at `:58` in that mode.
6. Write a minimal `sbatch` wrapper: `ml` line, venv activate, `export RATSIM_UNITY_BIN=...`,
   one `train.py` invocation. Run one short job end-to-end on `gpufast`.

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
| Phase 0 de-risk | ~1h (plus queue wait) |
| First training job on RCI | ~1 day |
| Safe concurrent multi-run | +1–2 days |
| Comfortable day-to-day setup | +1–2 days |

The headless-rendering risk that motivated this investigation is, on current evidence, ~1h of
work rather than a redesign. The real cost is the SLURM plumbing in Phase 2, and it's mechanical.

---

## 9. Open questions for whoever picks this up

- **[VERIFY]** Does `-batchmode -nographics` work? (Phase 0a — everything else assumes yes.)
- **[VERIFY]** Driver/CUDA version per partition; are nodes heterogeneous enough to need
  per-partition venvs?
- **[VERIFY]** Does `ldd` on the Unity build come up clean on a compute node?
- **[VERIFY]** Does SLURM set a per-job `$TMPDIR` on RCI? Determines the shape of the #2 fix.
- **[VERIFY]** Do compute nodes have outbound internet? Determines whether all installs must
  happen on `login1`.
- **[VERIFY]** Exact Lmod module names for Python / CUDA.
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
source /mnt/personal/$USER/ratvenv/venv/bin/activate
export RATSIM_UNITY_BIN=/mnt/personal/$USER/ForagerSimBuild.x86_64
export RATSIM_NOGRAPHICS=1
cd /mnt/personal/$USER/git/ratsim_experiments
python train.py def=<rundef> method=ppo
```

Monitoring, from `login1`, while the above runs elsewhere:

```bash
cd /mnt/personal/$USER/git/ratsim_experiments
python scheduler_status.py <exp> --watch 5
```
