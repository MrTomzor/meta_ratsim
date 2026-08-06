# Wells / Foraging task — next steps

Living plan doc for the home-return foraging task family (Widloski & Foster 2022 barrier
maze). Written so the conversation can be compacted without losing decisions.
See also auto-memory: `project_wells_foraging.md`, `unity-mcp-workflow.md`.
Papers: `~/foragersimprint/` (Widloski & Foster 2022 + supplemental).

---

## Status — DONE and verified (via Unity MCP, Unity 6000.1.8f1)

**Well primitive (Phase 1), prefab-only:**
- Unity: `WorldGen/Data/WellData.cs`, `Well.cs`; `WorldGen/Generation/WellLoader.cs`
  (+ `IWellProvider`), `StructureSlotDistribution.cs`; `WorldDataType.Wells`.
  `RewardObjectLoader` refactored to share `StructureSlotDistribution`.
- Wells are **prefab-only** (`wells/prefab_name` → `Resources/WorldGen/WellPrefabs/<name>`,
  default `well_basic`). `WellLoader` instantiates the prefab, adds+configures `WellData`+`Well`.
  Runtime primitive builder was removed.
- **Dispense = spawn REAL reward objects, collected by TOUCH** (fixed the old "phantom pickup"
  where reward was credited on proximity). Rewards spawn in annulus
  `[wells/reward_spawn_min_dist, wells/reward_spawn_max_dist]` (both 0 = on well),
  `wells/rewards_per_dispense`. `wells/arrival_radius` only gates WHEN it dispenses. Well waits
  until all dispensed rewards are collected, then cooldown → re-arm. Always-armed feeder (no
  scheduler yet).
- Debug gizmo (`wells/debug_gizmo`): green=active / yellow=closed / red=depleted + arrival-radius ring.
- Semantic: lidar (`SemanticLidarSensor`) raycasts everything EXCEPT the `WorldGen` layer and reads
  `NamedSemanticObject` off the hit collider. `well_basic` is a Default-layer collider + label `well`.
  Created `Resources/SemanticSets/reward_well_obstacle` = `[reward_obj1, well_basic]` → classes
  `reward_pickup`(0) + `well`(1), else default. Agent preset `sphereagent_2d_lidar_wells.yaml`.
- Scene: `WellLoader` GameObject added to `Wildfire.unity` (saved).
- Test preset: `ratsim/ratsim/config_blender/world_presets/maze_memorymaze_11x11_wells.yaml`
  (maze rooms, rewards swapped for wells, `seed_key: reward` → same slots as reward objects).
- Verified: 4 wells spawn from prefab, dispense-by-touch works, no phantom pickup, lidar reports
  `well`/`reward_pickup` classes, 0 errors.

**Run/verify loop:** Unity Editor open → `unity` MCP. Compile-check:
`refresh_unity(compile)` → poll `mcpforunity://editor/state` `ready_for_tools` → `read_console`.
Drive/step test: `manage_editor(play)`, then from `ratsim/`:
`python -m ratsim.fps_test --world <w> --agent <a> --max-steps N` (auto-steps, non-interactive)
or `python -m ratsim.human_control_test --world <w> --agent <a> --rtf 1.0` (keyboard drive).

---

**Phase 2 — `widloski_maze` layout (built + verified 2026-07-02):**
- NEW `Assets/WorldGen/Generation/WidloskiMazeLayoutLoader.cs` (ILayoutProvider +
  IRoomProvider, active on `layout/mode=widloski_maze`) + `RoomStructureBuilder.cs`
  (shared footprint-only room + `LOD0/rewardSpawnPositions/<slot>` builder;
  `MazeLayoutLoader.SpawnRoomStructure` refactored to use it, behaviour-identical).
- Emits N×N well-rooms (each a `well_room` WorldStructure with one center slot) on a
  grid centered at world origin, and thin stretchable `barrier` prefabs on a per-episode
  subset of the 2N(N-1) inter-room edges. BFS reachability over the cell graph with a
  virtual "margin ring" node when `margin_enabled=1`; reshuffle-retry up to `n_tries`.
- NEW prefab `Resources/WorldGen/BarrierPrefabs/barrier_basic` (unit-cube, Default-layer
  BoxCollider + `NamedSemanticObject("barrier")`; loader sets localScale = span/height/thickness).
  Added `barrier` (class 2) to `Resources/SemanticSets/reward_well_obstacle`.

**Phase 3 — Home/Random scheduler (built + verified 2026-07-02):**
- NEW `Assets/WorldGen/Generation/WellScheduleController.cs` (WorldDataProvider,
  Provides `WellSchedule`, DependsOn `Wells`; `WorldDataType.WellSchedule` added). Active on
  `well_schedule/enabled=1`. Picks a fixed UNCUED Home well (`home_well_id`, -1=auto centroid),
  alternates Home↔Random trials, arms exactly one well at a time (Random cued, Home uncued),
  advances when the target dispenses AND its rewards are collected. Publishes `/well/state`
  as FloatArrayMessage `[target_id, trial_type(0/1), cued, home_id, dispensed]`.
- Well exposes `DispenseCount` + `AwaitingCollection` for completion detection.
- Scene: `WidloskiMazeLayoutLoader` + `WellScheduleController` GameObjects added to
  `Wildfire.unity` (saved). Presets `widloski_maze_3x3.yaml` (10-unit cells, 6/12 barriers,
  scheduler on) + `widloski_maze_11x11.yaml`. Verified via fps_test @60FPS: 9 wells, 6
  barriers reachable attempt 1, agent spawns cleanly, Home=centroid well, first Random trial
  cues one well. 0 errors.
- Spawn note: `agents_spawn_pos in_structures` samples within (cell/2 − spawnSafetyRadius) of
  the room center, which is where the well sits — so cell_size must exceed ~2·spawnSafetyRadius
  for clearance (why the presets use cell_size 10/8, not the paper's tight spacing).

**Cue + inter-trial delay (built + verified 2026-07-02):**
- Cue = abstract "LED" via a `SignalSource` on the `cue` channel (NOT RGB), perceived by the
  existing `SectorSignalSensor`. `WellLoader` attaches a disabled `SignalSource` to every well
  (`wells/cue/channel|range|strength|falloff`); `Well.UpdateCue()` enables it only while
  `armed && cued && !depleted` per `wells/cue/mode`: `until_depleted` (faithful steady LED,
  default) | `timed` (on for `wells/cue/duration_steps` then dark — harder) | `none`. Home
  trials arm uncued → cue stays silent (the memory demand). No occlusion (matches transparent
  barriers). New agent preset `sphereagent_2d_lidar_wells_cue.yaml` adds `sector_signal
  channels: cue`; `env.py` lifts it into obs with no gym changes.
- Inter-trial delay: `WellScheduleController` waits `well_schedule/inter_trial_delay_{min,max}_steps`
  (uniform per trial, both 0 = off) with NO well armed after each collection (paper's 5-15 s gap).
- `/well/state` now published EVERY step (was on-change) so per-step RL obs / late subscribers
  always see `[target_id, trial_type, cued, home_id, dispensed]`.
- Verified via a probe (agent stationary on Home): `/sector_signal/cue` peaks ~0.83 every step
  toward the cued Random well; `/well/state`=[7,0,1,5,0] (Random, cued, home=5). 0 errors.

**Still TODO (not yet done):** drive-test trial ADVANCE in `human_control_test` (fps_test
doesn't navigate to wells); optionally add a *visible* `cueLight` child to `well_basic` (the
`Well` already toggles one if present — currently none, so the cue is signal-only, which is what
the abstract-sensing agent uses anyway); fold `/well/state` into `env.py` obs/logging if desired;
consider filtering the collision penalty for the `barrier`/`well` semantics; the bat-orchard
`patch_depletion` variant.

## Original Phase-2 design notes (as-built matches these): `widloski_maze` layout

**Why a new layout (not reuse maze gen):** `MazeLayoutLoader` materializes walls as chunky
`cell_size` cubes from a binary mask — wrong substrate for the paper's thin "jail-bar" barriers,
and its walls make maze-specific reachability checks awkward. So build a dedicated provider.

**Naming:** `layout/mode = widloski_maze`, class `WidloskiMazeLayoutLoader`. The reconfigurable-
barrier + reward-well-grid apparatus is Widloski & Foster 2022's design (transparent "jail-bar"
barrier concept from Ólafsdóttir et al. 2015 — credit in a code comment). Do NOT call it the
ambiguous `well_maze`.

**What it emits (open arena, paper-faithful):**
- Perimeter via `WorldBoundaryLoader`. Arena size = `N·cell_size + 2·margin`.
- **N×N well-rooms** — regular grid. Each cell **IS a room** (a labeled `WorldStructure` with a
  center reward-spawn slot), exactly like the existing maze rooms. Scales 3×3 → 11×11+.
- **Thin barriers** on inter-room edges. N×N grid has `2·N·(N−1)` edge slots (12 for 3×3 ✓).
  A per-episode barrier-seed picks a subset (`barrier_fraction`, paper = 6/12), reshuffled each
  reset (= paper's per-session reconfiguration; one episode = one "session" of many trials).

**Barriers = switchable prefab, stretchable:**
- `barriers/prefab_name` → a thin `barrier` prefab (Default-layer collider + `NamedSemanticObject`
  `barrier` + configurable visibility), symmetric with the well prefab.
- `barriers/stretch_walls_to_fit_cell_size` (bool): if true, scale the barrier so its across-edge
  dimension (x or z, whichever runs along the cell edge) = `cell_side_length ×
  barriers/stretch_walls_cell_size_percentage` (default `1.0` = spans the full cell side; paper
  ≈ 0.85, leaving impassable end-gaps). Handles making cell size wider/narrower.

**Connectivity / reachability check (paper-faithful open field):**
- BFS over the **cell grid** from a start cell (e.g. agent spawn cell). Spread to an orthogonally-
  adjacent cell UNLESS a barrier sits on the shared edge (a wall stops the spread).
- **Margin toggle** (`widloski_maze/margin_enabled`, on/off — easy switch): when ON, any two cells
  on the OUTER edge of the map are also connected (you can travel around via the outer margin band).
  Model as: perimeter cells all join one component (e.g. an extra "margin ring" node linking all
  edge cells), so BFS can route around blocked interior edges via the perimeter.
- This catches large enclosures at scale (e.g. in a 10×10, barriers can wall off a 2×2 block of
  cells entirely). Require every well-room reachable from start; if not, **reshuffle the barrier
  seed and retry up to `widloski_maze/n_tries`** (same pattern as existing gen variants), then
  error loudly.

**Reuse — DO NOT duplicate "place a well in the center of a room":**
- Both `widloski_maze` AND the existing maze gen produce room `WorldStructure`s with a canonical
  **center slot**. Factor the room-structure builder (currently `MazeLayoutLoader.SpawnRoomStructure`
  + a well-at-room-center rule) into a **shared helper** so one code path places a well in a room
  center for both layouts (extraction pattern like `StructureSlotDistribution`). WellLoader then
  works unchanged for both via `wells/allowed_structures: well_room`.
- Room **labels** only for now (e.g. `well_room`, later `scent_room`): don't implement scent/special-
  object spawning, just make sure the layout emits labeled rooms so future providers can target them.

**Draft config params:**
```
layout/mode: widloski_maze
widloski_maze/grid_n: 3            # N (→ N×N wells); scale up e.g. 11
widloski_maze/cell_size: <units>  # room/cell side; well spacing
widloski_maze/margin: <units>     # outer open band
widloski_maze/margin_enabled: 1   # perimeter routing on/off
widloski_maze/barrier_fraction: 0.5   # or n_barriers (paper 6/12)
widloski_maze/n_tries: <int>      # reachability reshuffle attempts
barriers/prefab_name: barrier_basic
barriers/stretch_walls_to_fit_cell_size: 1
barriers/stretch_walls_cell_size_percentage: 1.0   # paper ≈ 0.85
wells/enabled: 1
wells/prefab_name: well_basic
wells/allowed_structures: well_room
# ... wells/* dispense params as today
```
Provider: `ILayoutProvider` + `IRoomProvider`, active on `layout/mode=widloski_maze`, all eager in
`Generate()` (bounded arena, no chunk streaming). Barriers placed by the layout (it owns the slots);
barrier prefab authored like `well_basic`.

---

## Other TODO (not started)

- **Phase 3 — Home/Random alternation scheduler** (the memory-demanding core). Decided earlier:
  schedule logic lives in **Unity C#** (`WellScheduleController`, a `WorldDataProvider`). Alternates
  Home→Random→Home→Random trials; Home = one fixed unmarked well per episode (must be RECALLED →
  memory demand); Random = a cued well (LED). 5–15 s (→ steps) delay after arrival before dispense.
  Reward still flows via `Pickupable`→`/reward_pickup`→TaskTracker. Publish `/well/state` (target id,
  trial type, cued?) for Python obs/logging. Also `patch_depletion` mode for the bat-orchard case.
- **Bat-orchard variant:** tree-look well prefab carrying `VegetationModification{Remove}` (clears
  nearby trees so dispensed rewards have room); `well_maze`-independent (anchors in `orchard` OBBs
  via footprint-scatter); `patch_depletion` schedule (roster-free → composes with lazy registration). [NOTE - reuse existing abstract tree prefab used for vegetation spawning in some world configs]
- **Presets to finish:** faithful `widloski_maze` world preset (3×3, 6 barriers, Home/Random) +
  scaled variants (11×11) + bat orchard.

---

## Sources (naming attribution)
- Widloski & Foster 2022, Neuron — https://pmc.ncbi.nlm.nih.gov/articles/PMC9473153/
- (later, different apparatus) Omniroute maze 2025 — https://www.biorxiv.org/content/10.1101/2025.09.01.672969.full.pdf
