# Backlog — deprioritized ideas

Things worth doing eventually, deliberately **not** being worked on now. `roadmap.md` holds the
active plan; this file is where an idea goes when it has been examined and parked, so it doesn't
have to be re-derived from scratch later.

Each entry records **what it is, what it would buy, and why it was parked** — the last part matters
most. Several of these were parked because a measurement contradicted the idea, and that
measurement is easy to forget once the idea has been reduced to a one-line TODO.

---

## Performance

### ❌ DROPPED (2026-08-10) — Cap Unity's thread pool with `-job-worker-count`

**The idea.** Each headless Unity instance spawns exactly 266 threads (measured; dead consistent,
+266 per instance). Unity sizes its worker pools from `/proc/cpuinfo`, i.e. the whole 128-thread
node, not the SLURM cgroup — the same "sees the machine, ignores the allocation" trap as
`OMP_NUM_THREADS`. At `n_envs=4` on 16 allocated threads that is ~1064 Unity threads on 16 CPUs, and
it looked like a strong candidate for *causing* the cores-per-env cliff (17 fps on 4 threads against
688 on 16). Unity accepts `-job-worker-count N`.

**Why it was dropped: the data we already had refutes it.**

- The 7-wide job (11318081) ran **28 Unity instances ≈ 7,400 threads on 112 CPUs** and still held
  **84% of solo fps**. If thread count were the binding constraint, that would have collapsed.
- The cluster runs 266 threads per instance against the laptop's ~30 (16 CPUs) and is *faster*
  per run (723 vs 581 fps).

So at our operating point — 16 threads per run — capping workers buys approximately nothing. The
4-thread cliff is more likely plain starvation: 4 Unity instances plus a Python process on 4 CPUs.

**What would revive it.** Only a push for *denser packing* — e.g. 14 runs × 8 threads instead of
7 × 16. The bar is high: 14 × 251 = 3514 is worse than the 4816 we get today, so 8-thread
throughput would have to climb from 251 to over 340 just to break even, and past ~600 to be worth
the complication. Also note `-job-worker-count` was never confirmed to exist on our build.

### ✅ ANSWERED (2026-08-10) — the ~1000 `rollout_fps` ceiling is latency, not Python compute

**Profiled on the cluster** (job 11326598, `amdfast`, a07, 16 threads, PPO `n_envs=4`, 30k steps),
`py-spy --subprocesses` run twice: once sampling on-CPU only, once with `--idle` for wall time.
Raw stacks kept at `/mnt/personal/musilto8/logs/prof-11326598/`.

**The process is predominantly blocked, not computing.** Every top wall-time frame is a blocking
call — `multiprocessing/connection.py:_recv` 34.6%, `selectors.py:select` 15.2% (the Unity socket),
`threading.py:wait` 10.8%. Total on-CPU samples were 12,119 against 55,893 wall samples.

**Of the CPU that is used, the biggest single item is our own code:**

| on-CPU | what |
|---|---|
| **~18%** | `task_tracker/exploration_tracker.py:update_from_lidar` + `_bresenham_ray` |
| ~14% | connector: `flush_send` 10.5%, `read_messages_from_unity` 2.9%, `json.raw_decode` 1.0% |
| ~8.5% | `SubprocVecEnv` pipe traffic (`_recv` 6.6%, pickle `dumps` 1.9%) |
| ~6% | torch (the optimize phase) |

`update_from_lidar` is a **per-ray Python loop**: `_bresenham_ray` walks cells in a `for` loop, then
~8 small numpy ops run on a few-dozen-element array, per ray, per step, per env. Hundreds of tiny
numpy calls per step, where per-call overhead dominates the arithmetic. Vectorizing across all rays
at once (one `(n_rays, max_cells, 2)` array instead of a loop) is mechanical and would remove most
of it. Note it only runs for volumetric-exploration tasks — which is what the current wells/ortho
paper runs use, so it does apply.

**But fixing it does not buy the 1.4×.** Python compute is the minority of wall-clock, so removing
even the whole 18% is worth single-digit percent overall. The ceiling is the **serialized
round-trip** — policy → pipe → Unity → pipe → policy — where the dominant terms are IPC latency and
Unity's own step time, neither of which is Python being slow.

So the original hoped-for 1.26–1.44× per-run speedup **is not available from optimising our Python**.
If per-run wall-clock ever becomes the binding constraint, the levers are Unity's step time or the
IPC structure (e.g. shared-memory observation transfer instead of pickling through pipes), both far
larger pieces of work than this was scoped as.

⚠️ The precise on-CPU/wall ratio is rough: `--idle` samples per *thread*, and an idle
`multiprocessing.resource_tracker` process contributes 12.4% of wall samples while doing nothing.
The qualitative split — blocked ≫ computing — is robust; treat the exact percentages as indicative.

**Residual cheap win, if anyone is in that file anyway:** vectorize `update_from_lidar`. ~18% of
Python CPU for a mechanical change. Not the ceiling, just free money.

*Background, for context:* rollout is 62% of wall-clock (7-wide run: `fps=593`,
`rollout_fps=962` → 1.04 ms/step rollout vs 0.65 ms/step optimize), so 2× rollout would have been
1.44× overall — 8.3 h → 5.8 h on an 18M-step run. That is the number now shown to be unavailable
from Python-side work.

---

## Scheduler

### Checkpointing only at stage boundaries

A run that outlives its SLURM wall-clock loses the current stage's progress. Dreamer has its own
periodic in-stage checkpointing (embodied writes `ckpt/latest` ~every 10 min), so the RAM-kill
watchdog resumes cheaply, but SB3 methods do not. Currently mitigated by choosing a long enough
partition rather than fixed.

### Multi-GPU verified at 2 cards, not 4

B4 (jobs 11325473, 11325609) proved the `GpuAllocator` on **2** A100s. A whole `g[01-10]` node is 4
cards and 124 threads (`gpu: 4`, `cpu_slot: 124`), which nothing has exercised. Two things could
differ at 4: whether the 72%-of-solo packing cost holds or degrades as more Unity instances contend
for the same cores, and whether `--mem` needs raising (dreamer's `max_ram_gb: 30` × 4 = 120 GB plus
Unity). Not blocking — 2 GPUs works today — but the numbers in `rci_gpu.yaml` for a full node are
extrapolated, not measured.

### `PortAllocator.window_size` is hard-capped at 10

Which caps `n_envs ≤ 10` for scheduler-driven runs. Fine at the pinned `n_envs=4`, but
`defs/memory_houses_paperhparams.yaml` wants 256 envs and is therefore CLI-only.

---

## Small, mechanical

### Vectorize `exploration_tracker.update_from_lidar`

**~18% of all Python CPU time** during a volumetric-exploration run (job 11326598 — see the
rollout-ceiling entry above). `_bresenham_ray` walks cells in a Python `for` loop, then ~8 small
numpy ops run on a few-dozen-element array — per ray, per step, per env. Hundreds of tiny numpy
calls per step, where per-call overhead dominates the arithmetic.

The fix is mechanical: build one `(n_rays, max_cells, 2)` array and do the bounds-clip, free-cell
fill and endpoint write as a handful of whole-array operations instead of a loop.

**Do not expect it to move training throughput much** — Python compute is the minority of
wall-clock, so this is worth single-digit percent overall. Worth doing if someone is in that file
anyway; not worth a dedicated push.
