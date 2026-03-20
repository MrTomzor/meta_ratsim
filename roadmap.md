[name in progress, now ratsim] is a simulation framework/set of packages allowing simulation of agents moving in extremely large-scale environments that is solvable both through an AI Gym interface and through ROS2. 

Core Features:

- **2D and 3D** — Easy switching (in a user’s package extending either the RL env or in a ROS2 package extending the ROS2 core package) between 2D lidar-based setup (much faster simulation without rendering) and 3D RGBD setup for solving the same world setup. (e.g. both a 2D agent and a 3D agent will share the world layout generation, but in the 2D scenario the height generation will be set to the “superflat” mode and agent will be configured to only rotate along the vertical axis). This is made possible through the Layout generation of the Unity WorldGen being the first thing to be generated, and height being determined by the layout (and thus modifiable to be flat)
- **Procedural generation** - implemented in Unity and controller through a world config allows training RL algorithms and testing robustness of mobile robotics algorithms (SLAM/navigation/…).
- **Connection to AI Gym**
- **Connection to ROS2**
- **Unified world+agent configuration across Gym and ROS2** - the config controls the world and agent setup in unity (and the Gym env will read it to correctly construct the observation and action space, while the ROS2 node probably just needs to map between ratsim and ROS2 messages as they come and go). This is the core that allows the same “problem” (world+agents setup) to be addressed both by the robotics and the RL community

CURRENTLY - working towards a workshop paper (deadline - may 5th). Priorities/Milestones:

- **SAR DOG EXAMPLE** — can simulate a large-scale search-and-rescue mission within a city environment, where we can change the layouts of the cities and the properties of the houses during different stages of a curriculum. (e.g. making all doors blocked in a new phase → would test the agent’s ability to adapt and instead try pulling through the rubble if trained on that) + can show training of separate skills in a “lab” environment with e.g. a single house spawned at a time, with both rubble (one wall being turned to rubble) and doors, or only rubble, or only doors. (would test compositional learning).
    - primarily an RL env example, but could show the same mission in ROS2 - same world and agent config
    - should be able to show e.g. a result of a trained PPO agent on doors-only / rubble-only / all-together / rule-switch (suddenly all house doors being locked or very different house structure) example
    - reward objects = abstract boxes with a semantic class “reward”, same as with bat
- **BAT FORAGING EXAMPLE** — can simulate bat foraging mission in RL and ROS2 (varying scales) - with a cave, distant mountains, fruit patches (new type of worldstructure), a CAVE (as the home, to introduce multi-scale of the environment) and in general VERY large scale. This would serve to
    - primarily a ROS2 environment, I think I won’t have time to implement RGBD observations into an RL env and train anything reasonable
- **LARGE-SCALE WORLD DYNAMICS EXAMPLE** — should be able to show that the envs can be very simply manipulated (e.g. simply adding a global vegetation density reduction modifier - would simulate a forest after a storm). This would be interesting to test the adaptability to SLAM/navigation stacks (and also RL algorithms) to changes in an environment.

new

**SIM CONFIGURATION:** 

- world config - contains all info related to world generation
- agents config - contains info describing a list of agents, each agent contains:
    - a given core “prefab name” (for spawning the core prefab in Unity)
    - name_prefix — attached to all topics in and out of agent (e.g. a publisher publishing on “lidar2d” on an agent named “peter” should publish on “/peter/lidar2d”
    - modifiable actuators mode (core = velocity / acceleration)
    - list of sensors (each with different config variables, core = the 2D semantic lidar sensor, RGBD, odometry sensor), also allow setting their transformation relative to agent’s gameobject
- NEW: task config

**CURRENT TODOS:**
- [x]  **TaskTracker refactor** — `ratsim/ratsim/task_tracker/` submodule added. `TaskTracker` class accepts a task config dict (loaded from `task_presets/default.json`), exposes `reset()`, `update_with_unity_msgs()`, `get_this_step_score()`, `get_total_score()`, `is_terminated()`. Integrated into `env.py` — `reward_config` replaced by `task_config`. Curricula, train_ppo.py, test_manual_control.py updated accordingly.
    - [x]  Task 1 - implement the TaskTracker and integrate it with env.py in ratsim_wildfire_gym_env
        - [x]  Default task config json: episode_max_steps, foraging_settings, collision_settings, termination_settings
        - [x]  In env.py, replace all reward and termination calculation with the TaskTracker

- [ ]  **ROS2 package restructuring/generalization — should allow multiple agents!**

---

## AssetManager Refactor

Introduce a thin abstraction layer over `Resources.Load()` so that all asset loading goes through `AssetManager.Load<T>(path)` where the path comes from config JSON. This decouples world gen modules from hardcoded prefab references and enables config-driven asset swapping (visual fidelity profiles, contributor asset packs). Future migration to Unity Addressables becomes a one-method change.

### Design

```csharp
public static class AssetManager
{
    public static T Load<T>(string path) where T : UnityEngine.Object
    {
        return Resources.Load<T>(path);
    }
}
```

Works for all asset types: prefabs, textures, materials, meshes, terrain layers.

World gen modules read asset paths from config instead of hardcoding them:
```json
{"key": "vegetation/allowed_prefabs", "value": "tree_oak,tree_pine"},
{"key": "terrain_texture/grass", "value": "Textures/Terrain/grass_realistic"}
```

### Asset Organization

```
Resources/
├── AgentPrefabs/               — agent prefabs (sphere, quadruped, drone)
├── WorldGen/
│   ├── WorldStructurePrefabs/  — structures (city, house, road, etc.)
│   ├── VegetationPrefabs/      — trees, grass, ground cover
│   ├── RewardObjectPrefabs/    — reward objects
│   └── HouseModulePrefabs/     — doors, cars, rubble
├── Textures/                   — terrain textures, materials
└── ...
```

Contributors add assets to the appropriate folder, reference them by path in config presets. Licensed/purchased assets go in gitignored folders.

### Implementation

- [ ] Create `AssetManager.cs` static class with `Load<T>(string path)`
- [ ] Replace all `Resources.Load()` calls across world gen modules with `AssetManager.Load()`
  - WorldLayoutLoader (structure prefabs)
  - WorldBoundaryLoader (boundary prefab)
  - TreeLoader (vegetation prefabs)
  - CityLoader (house prefabs)
  - HouseLoader (door, car, rubble prefabs)
  - RewardObjectLoader (reward prefabs)
  - AgentLoader (agent prefabs)
- [ ] Verify all prefab paths are already read from config (most already are — TreeLoader, HouseLoader, RewardObjectLoader use config-driven paths; CityLoader hardcodes `"house"` prefix; AgentLoader reads `prefab_name` from config)
- [ ] Add `.gitignore` patterns for local/purchased asset folders

### Scope

This is a purely mechanical refactor — no behavior change, no new config keys, no Python-side changes. It's independent of the layer-based loading refactor below.

---

## Architecture Refactor: Layer-Based World Loading

Refactor the world generation pipeline from a flat module list with broadcast chunk requests into a **dependency-graph-ordered, per-layer chunk loading system** with multiple requestors. This is the foundation for open-source extensibility — contributors can add new data providers, new requestors, and configure per-layer load distances from Python config.

### Design Principles

- **Data providers, not modules.** Each provider declares what data type it produces (`Provides`) and what it needs (`DependsOn`). The system topologically sorts providers and dispatches chunk requests only to the relevant provider.
- **Per-layer spatial loading.** Each data type (terrain mesh, vegetation, structures, etc.) has its own configurable load radius and LOD distances, controlled from Python world config JSON.
- **Multiple requestors.** Any entity that needs world data around it (agent, fire simulation, physics objects, autonomous vehicles) is a requestor. Each declares which layers it needs and at what radius. The system unions all demands — data stays loaded as long as any requestor needs it.
- **Service interfaces for cross-module queries.** Static singletons (`WorldHeightLoader.instance`) replaced by a service locator (`WorldServices.Get<IHeightProvider>()`), enabling swappable implementations.

### WorldDataType Enum

```
Height, Layout, Boundaries, StructureEvents, StructureContent,
Rewards, Agents, Vegetation, TerrainMesh, TerrainTexture, Lighting
```

Contributors adding new data types (Biomes, Rivers, Dynamics) add to this enum — a deliberate, reviewed architectural decision.

### Dependency Graph

```
Layout ──→ Height
         ↘ StructureEvents ──→ StructureContent (City, House)
                             ↘ Rewards
         ↘ Boundaries
StructureContent + Height ──→ Agents
Height + StructureContent + Agents ──→ Vegetation
Height ──→ TerrainMesh ──→ TerrainTexture
(independent) ──→ Lighting
```

### Provider Types

| Provider | Type | Provides | DependsOn |
|---|---|---|---|
| WorldHeightLoader | Global (queryable field) | Height | — |
| WorldLayoutLoader | Global | Layout | Height |
| WorldBoundaryLoader | Global | Boundaries | Height, Layout |
| StructureLoadingCoordinator | Spatial | StructureEvents | Layout |
| CityLoader | Structure-triggered | StructureContent | StructureEvents |
| HouseLoader | Structure-triggered | StructureContent | StructureEvents |
| RewardObjectLoader | Spatial + Structure | Rewards | Height, StructureEvents |
| AgentLoader | Global | Agents | Height, StructureContent |
| TreeLoader | Spatial (LOD0) | Vegetation | Height, StructureContent, Agents |
| TerrainMeshLoader | Spatial (all LODs) | TerrainMesh | Height |
| TerrainTextureLoader | Spatial | TerrainTexture | TerrainMesh, Height |
| LightingAndFogLoader | Global | Lighting | — |

### Per-Layer Config (from Python)

```json
{
  "layer/terrain_mesh/load_radius": 500,
  "layer/terrain_mesh/lod_distances": "100,300,500",
  "layer/vegetation/load_radius": 150,
  "layer/structure_events/load_radius": 400,
  "layer/terrain_texture/load_radius": 500,
  "layer/rewards/load_radius": 150
}
```

2D lidar RL training preset — zero visual layers for maximum speed:
```json
{
  "layer/terrain_mesh/load_radius": 0,
  "layer/terrain_texture/load_radius": 0,
  "layer/vegetation/load_radius": 0,
  "layer/lighting/load_radius": 0
}
```

### Multiple Requestors

Any entity that interacts with the world registers a requestor declaring which layers it needs:

- **Agent** — all layers, radii from config
- **Fire simulation** — VegetationDensity + StructureContent, large radius
- **Physics objects** — Height + StructureContent (collision geometry), small radius
- **Autonomous vehicles** — Layout + Height, medium radius

Requestors are created/destroyed dynamically (fire starts, boulder comes to rest). The chunk controller unions all active requestors' demands per layer.

### Implementation Phases

**Phase 1 — Foundation (no behavior change)** ✅
- [x] Define `WorldDataType` enum
- [x] Create `WorldServices` static service locator + `IHeightProvider`, `ITerrainMeshProvider`, `ILayoutProvider` interfaces
- [x] Create `WorldDataProvider` abstract base class (Provides, DependsOn, Generate, GenerateChunk, ClearChunk, Clear)
- [x] Create `WorldStructureProvider` base class (replaces WorldStructureLoader)

**Phase 2 — Migrate existing modules (incremental, one at a time)** ✅
- [x] WorldHeightLoader → WorldDataProvider + IHeightProvider
- [x] WorldLayoutLoader → WorldDataProvider + ILayoutProvider
- [x] WorldBoundaryLoader → WorldDataProvider
- [x] StructureLoadingCoordinator → WorldDataProvider
- [x] CityLoader → WorldStructureProvider
- [x] HouseLoader → WorldStructureProvider
- [x] RewardObjectLoader → WorldStructureProvider
- [x] AgentLoader → WorldDataProvider
- [x] TreeLoader → WorldDataProvider
- [x] TerrainMeshLoader → WorldDataProvider + ITerrainMeshProvider
- [x] TerrainTextureLoader → WorldDataProvider
- [x] LightingAndFogLoader → WorldDataProvider
- [x] SimpleStructureLoader → WorldStructureProvider
- [x] Replace all `ModuleName.instance` calls with `WorldServices.Get<IInterface>()`
- [x] Add topological sort (Kahn's algorithm) to WorldLoadingController
- [x] Fix CityLoader.Generate() collection-modified-during-enumeration bug

**Phase 3 — New chunk loading system**
- [ ] Refactor `ChunkLoadingRequestor` into abstract base with `GetPosition()`, `RequiredLayers`, `GetLoadRadius(layer)`, `GetLodDistances(layer)`
- [ ] Create `AgentChunkRequestor` (concrete, reads radii from `layer/*/load_radius` config)
- [ ] Create `ChunkLoadingController` — topological sort, per-layer chunk tracking, requestor demand merging, dependency expansion
- [ ] Simplify `WorldLoadingController` — remove `InitializeAllModules()` loop, delegate chunk logic

**Phase 4 — Cleanup** ✅
- [x] Delete `WorldLoadingModule.cs`, `WorldStructureLoader.cs`
- [x] Remove all `static instance` singletons from providers (7 removed)
- [x] Update all stale comments referencing old class names
- [ ] Remove `_generated` / `_spawned` / `_paramsLoaded` guards (controller guarantees single execution)

**Phase 5 — Config integration (Python side)**
- [ ] Add `layer/*/load_radius` and `layer/*/lod_distances` keys to `world_presets/default.json`
- [ ] Create `world_presets/fast_2d.json` preset (zero visual layers)
- [ ] Defaults reproduce current behavior (radius=3 chunks, longRangeRadius=8 chunks × chunkWidth) so existing presets work unchanged

### Migration Safety

- Phases 1, 2, 4 complete — all modules migrated, old base classes deleted, singletons removed
- Phase 3 is the next step — per-layer chunk loading with multiple requestors
- Default layer radii reproduce current two-tier LOD behavior if no `layer/` config keys are present
- Python side (env.py, curricula.py, train_ppo.py) is unaffected — it just sends config keys over TCP

### Future Extensions (enabled by this architecture)

- **Multiple requestors:** fire, physics objects, autonomous vehicles, multi-agent
- **New data types:** Biomes, Rivers, Weather, Dynamics — add enum value + provider
- **Swappable providers:** GPU-instanced vegetation, erosion-based height, biome-driven texturing — implement same interface, configure which is active
- **Prefab/asset swapping:** config-driven asset paths + AssetManager enable visual profile switching without code changes
- **Parallel chunk loading:** dependency graph identifies independent layers that could generate concurrently
