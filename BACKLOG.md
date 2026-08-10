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

### ⏸ DEFERRED (2026-08-10) — Profile the ~1000 `rollout_fps` per-process ceiling

**The idea.** `rollout_fps` saturates near 1000 env-steps/s at `n_envs=4` on both CPU
architectures, and each of 7 packed runs still hit ~1000 simultaneously — a shared hardware
bottleneck would have divided. So it is per-process, above the simulator, on our side. Candidates:
per-step TCP round-trip in the connector, obs assembly/serialization, the gym wrapper stack,
`SubprocVecEnv` pipe overhead.

**What it would buy.** Rollout is **62% of wall-clock** (from the 7-wide run: `fps=593`,
`rollout_fps=962` → 1.04 ms/step rollout vs 0.65 ms/step optimize). So:

| if rollout gets | overall gain | 18M-step run |
|---|---|---|
| 1.5× faster | 1.26× | 8.3 h → 6.6 h |
| 2× faster | 1.44× | 8.3 h → 5.8 h |
| free (unreachable) | 2.6× | — |

Worth having, not transformative. It bounds **per-run wall-clock**, which is what sets iteration
speed; the 7-wide packing already works around it for *aggregate* throughput across seeds.

**Cost: ~30 min, laptop only, no cluster job.** Attach `py-spy` to a running training. The first
question to answer is whether the process is *burning* CPU in Python or *idle waiting* on Unity's
TCP round-trip — those have completely different fixes, and `py-spy` separates them directly.

A negative result closes the question usefully: if it's idle-waiting on Unity, ~1000 steps/s per
env is simply what the simulator does and we stop thinking about it.

---

## Scheduler

### Checkpointing only at stage boundaries

A run that outlives its SLURM wall-clock loses the current stage's progress. Dreamer has its own
periodic in-stage checkpointing (embodied writes `ckpt/latest` ~every 10 min), so the RAM-kill
watchdog resumes cheaply, but SB3 methods do not. Currently mitigated by choosing a long enough
partition rather than fixed.

### `PortAllocator.window_size` is hard-capped at 10

Which caps `n_envs ≤ 10` for scheduler-driven runs. Fine at the pinned `n_envs=4`, but
`defs/memory_houses_paperhparams.yaml` wants 256 envs and is therefore CLI-only.
