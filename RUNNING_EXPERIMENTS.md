# Running experiments on RCI

You pick **an experiment** and **how long you're willing to give it**. Everything
else — partition, `--cpus-per-task`, `--gres`, `--mem`, and whether the def needs
one job or two — is derived from the def and `scheduler/machines/*.yaml`.

```bash
ssh musilto8@login2.rci.cvut.cz          # login2/3/4 — NOT login1
cd /mnt/personal/$USER/git/meta_ratsim
./rci_port_probes/submit.sh memory_orthomaze --time 3d
```

> **login1 will not work.** It's CentOS 7 / glibc 2.17 and can't run the
> module-provided Python the venvs symlink. You get a misleading
> `libpython3.12.so.1.0: cannot open shared object file`.

Always look before you leap:

```bash
./rci_port_probes/submit.sh memory_orthomaze --time 3d --dry-run
```

---

## Job lengths

`--time` picks the partition. You never type a partition name.

| `--time` | Tier | CPU partition | GPU partition | Aggregate cap |
|---|---|---|---|---|
| ≤ `4h` | fast | `amdfast` | `amdgpufast` | 700 CPUs, 8 GPUs |
| ≤ `1d` | day | `amd` | `amdgpu` | **200 CPUs**, 8 GPUs |
| ≤ `3d` | long | `amdlong` | `amdgpulong` | **200 CPUs**, 6 GPUs |
| ≤ `21d` | extralong | `amdextralong` | `amdgpuextralong` | **200 CPUs**, 8 GPUs |

Accepted formats: `4h`, `90m`, `1d`, `3d`, `3-00:00:00`, `12:00:00`.

**The caps are aggregate across all your running jobs, not per-job.** One
112-thread PPO job plus one 62-thread GPU job is 174 and fits. Two 112-thread
PPO jobs are 224 and don't — the second just pends forever with no error message.
`submit.sh` checks `squeue` and warns before you do this.

### Choosing a length

- **Over-requesting is nearly free.** When the last stage finishes the scheduler
  exits and SLURM releases the allocation immediately; you don't hold the node
  for the remainder. The only real cost is queue position.
- **Under-requesting is nearly free too**, because resume is the default. You
  lose at most the in-flight stage per run.
- So: `4h` when you want to *see something* soon (the fast partitions also
  schedule sooner), `3d` for real work, `21d` when a single run genuinely
  needs it.

The taster-then-grind pattern is the intended workflow:

```bash
./rci_port_probes/submit.sh memory_orthomaze --time 4h --mode bfs   # look at curves
./rci_port_probes/submit.sh memory_orthomaze --time 3d              # then commit
```

Same exp_id, same `.done` markers, same `wandb_id.txt` — the second job continues
the *same* W&B runs. Curves extend; they don't fork.

---

## Job shapes

Every run needs **16 threads** at the pinned `n_envs=4`. That's a cliff, not a
slope — the same run measures 17 fps on 4 threads and 723 on 16. **Only dreamer
needs a GPU.** PPO and RecurrentPPO both stay off the cards and leave them free.

RecurrentPPO moved to CPU on 2026-08-11 after measurement: an A100 is worth
1.6% end-to-end, because a 5000-step LSTM unroll at batch 1–2 is latency-bound
(64× the work for 1.32× the time on a direct batch sweep). Details in
`scheduler/machines/rci.yaml`.

| Machine config | Allocation | Runs at once |
|---|---|---|
| `rci` — AMD CPU node | 112t, 200G | **7 PPO** |
| `rci_gpu2` — 2× A100 | 2 GPU, 62t, 200G | **2 GPU runs** (+1 PPO if the job mixes) |
| `rci_gpu` — 1× A100 | 1 GPU, 31t, 100G | 1 run |
| `default` — laptop | 16t | 1 run |

Measured throughput:

| Setup | steps/s | per 1M steps |
|---|---|---|
| PPO, 1 run | 723 | 0.38 h |
| **PPO, 7-wide** | 608 each, **4253 total** | 0.46 h each, 7 at once |
| Dreamer, 1 A100 | 66 | 4.2 h |
| **Dreamer, 2× A100** | 48 each, 95 total | 5.8 h each, 2 at once |

**Dreamer is ~9× slower per step than PPO.** Plan it in steps, never by analogy.

### One def, two jobs

A def mixing PPO and dreamer becomes two submissions automatically, because one
scheduler process holds one allocation and `rci.yaml` has no dreamer profile at
all (the `amd*` partitions have no accelerators):

```
$ ./rci_port_probes/submit.sh method_compare --time 3d --dry-run
method_compare: 7 runs (1 variations × [ppo×3, dreamer×3, recurrent_ppo×1] seeds),
                10 stages × 1,000,000 steps
  wall clock 3-00:00:00 → long partitions   mode bfs
  → amdlong      112t, 200G          ppo,recurrent_ppo  (4 runs, 4 concurrent)
  → amdgpulong   62t, 2×GPU, 200G    dreamer            (3 runs, 2 concurrent)
```

Both feed the same exp_id: **one W&B group, one `analyze_experiment.py`**. A
single-method def stays a single submission.

Escape hatches: `--machine rci_gpu2` forces everything into one job (PPO gets
masked off the cards); `--only ppo` submits just one method's share.

---

## Worked example: `memory_orthomaze`

As it stands: dreamer only (PPO and RecurrentPPO commented out), 3 seeds, 1
variation → **3 runs**, 40 stages × 300k steps = 12M each.

```bash
./rci_port_probes/submit.sh memory_orthomaze --time 3d --dry-run
```

One GPU job, `rci_gpu2`, 2 cards. Two seeds train concurrently and the third
waits for a slot.

**Rough timing** (from the 2-wide dreamer rate of 48 steps/s — an estimate, not
a measurement of this def): ~1.7 h per stage, ~70 h per run, and with only 2
slots for 3 runs the whole thing lands around **5 days**. That's past
`amdgpulong`'s 3-day limit, so either:

```bash
./rci_port_probes/submit.sh memory_orthomaze --time 5d    # → amdgpuextralong
```

or run `--time 3d` and submit the identical line again when it expires. Both are
fine; resume makes them equivalent in outcome.

At ~1.7 h per stage, a job cut off mid-stage loses well under two hours per
in-flight run — this def is sized sensibly for interruption.

If you uncomment `ppo`, the same command becomes two jobs (112t CPU + 62t GPU =
174 CPUs) and finishes far sooner, because PPO isn't competing for the cards.

---

## Resuming

**Resume is the default. There is no resume flag.** Re-submit the identical line:

```bash
./rci_port_probes/submit.sh memory_orthomaze --time 3d
```

The scheduler scans `checkpoints/stage_<i>.done` under every run and dispatches
only what's missing. There's no scheduler-managed progress state to corrupt — the
markers on disk *are* the state, and a stage killed mid-save simply never got its
marker, so it re-runs from the previous stage's checkpoint.

What an interruption costs:

| Method | Lost work |
|---|---|
| PPO / RecurrentPPO | the **whole in-flight stage** (checkpoints land at stage boundaries) |
| DreamerV3 | **~10 min** (embodied writes a rolling `ckpt/latest`) |

Re-submitting a *finished* experiment is a harmless no-op — every marker is
present, so the scheduler exits immediately.

To wipe and start over instead:

```bash
cd ratsim_experiments && python scheduler_run.py memory_orthomaze --restart
```

---

## Monitoring

```bash
# from login2/3/4
. ~/rci_env.sh && ratsim_activate && cd $RATSIM_GIT_DIR/meta_ratsim/ratsim_experiments
python scheduler_status.py memory_orthomaze --watch 5

# or the raw job log
tail -f /mnt/personal/$USER/logs/sched-<jobid>.out
```

W&B gives you live curves either way — group `<exp_id>`, tagged `rci`.

---

## Editing a def to run longer

Give **any two** of `steps_per_stage`, `total_steps`, `n_stages` — the third is
derived. No mental arithmetic required; pick whichever two you actually have an
opinion about.

**Best: the budget and the granularity.**

```yaml
steps_per_stage: 300_000
total_steps: 12_000_000     # → 40 stages
```

Want to train longer? Raise `total_steps` to 18M. Stage size is unchanged, so
the first 40 stages keep their meaning and 20 new ones append. Nothing already
finished is invalidated.

The two must divide evenly. If they don't, the error names the two nearest
usable totals — again, no arithmetic on your side:

```
total_steps=12,500,000 is not a multiple of steps_per_stage=300,000.
Nearest usable totals: 12,300,000 (41 stages) or 12,600,000 (42 stages).
```

`steps_per_stage` + `n_stages` works the same way (extend by raising
`n_stages`). Giving all three is allowed and cross-checked — a contradiction is
an error, not a silent winner.

**⚠️ The legacy pairing is `total_steps` + `n_stages`**, which most defs
(including `memory_orthomaze`) still use:

```yaml
total_steps: 12_000_000
n_stages: 40                # → 300k per stage, DERIVED
```

Here the *stage size* is the derived quantity, and that's the one thing resume
depends on — `stage_<i>.done` records that stage K finished, never how big it
was. Bump `total_steps` to 18M on a run that's finished 30 of 40 stages and those
30 markers now claim 450k steps of training that never happened; resume builds on
them without complaint. The only safe edit to such a def is to scale both numbers
together so the ratio holds.

Converting is free whenever the derived size is already what you want. For
`memory_orthomaze`, 12M/40 is exactly 300k, so replacing `n_stages: 40` with
`steps_per_stage: 300_000` produces identical stages and existing markers stay
valid.

---

## Flags worth knowing

| Flag | Effect |
|---|---|
| `--time 4h\|1d\|3d\|5d` | wall clock → partition. Required. |
| `--dry-run` | print the sbatch lines, submit nothing |
| `--mode bfs\|dfs` | dispatch order. **Never inferred.** `dfs` finishes runs one at a time; `bfs` advances every run through early stages first — better when a short job will be cut off and you want all methods comparable. |
| `--only ppo` | submit just one method's job |
| `--machine rci_gpu2` | force everything into one job |
| `--step-multiplier 0.01` | 1% of every step count — smoke test |

Anything else is passed through to `scheduler_run.py`.

---

## Gotchas

- **login1 can't run any of this.** Use login2/3/4.
- **`scheduler_job.sbatch` alone defaults to `amdfast`, 4 h** — a smoke-test
  partition that kills real runs mid-stage. `submit.sh` exists partly to remove
  that trap; if you call sbatch by hand, pass `-p` explicitly.
- **The CPU/GPU caps are per-user aggregates**, so a second big job pends
  silently rather than erroring.
- **Don't retarget the `h200*` partitions.** The admin rule is to use them only
  at high utilisation, and one dreamer world model won't saturate one H200.
- **The 4-card config (`gpu: 4, cpu_slot: 124`) is extrapolated, not measured.**
  Two cards is verified (jobs 11325473 / 11325609). Measure before trusting four.
- **Dreamer's RAM watchdog has never fired on the cluster.** It *will* on a long
  run. Cheap proof: set `max_ram_gb: 4` in the machine config so it trips within
  minutes, and check the run resumes.

Details and measurements behind all of the above: `RCI_CLUSTER_PORT.md` (§0.1 is
the operational summary), `ratsim_experiments/scheduler/README.md`.
