# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working across the ratsim project.

## Project Overview
This project (name WIP) is a playground / simulator for researching large-scale long-horizon behavior, inspired primarily by the foraging capabilities of many types of animals (from bats traveling tens of kilometers to fruit trees, to dogs being trained to search for people in search-and-rescue missions over hour long missions and often requiring manipulation of the environment). Its primary goal is to pose foraging tasks/scenarios that can be attacked from multiple angles - robotics (ROS2 connection) and reinforcement learning (Gym env) and to find middle-ground solutions. The simulator offers large-scale, procedurally generated and loaded environments of many different types, while supporting as modular extensions as possible (new sensing, actuation, env generation, prefabs or textures etc).

## Sub-Repo CLAUDE.md Files

Before modifying code in any sub-repository, you MUST read that repo's CLAUDE.md file first:
- `ratsim/CLAUDE.md`
- `ratsim_unity_project/CLAUDE.md`
- `ratsim_wildfire_gym_env/CLAUDE.md`
- `ratsim_ros2/CLAUDE.md`
- `ratsim_experiments/CLAUDE.md`

## Meta-Repo Structure

This is a meta-repo containing symlinks to five independent git repositories that together form the ratsim simulation framework. Each repo has its own CLAUDE.md with detailed architecture docs.

| Directory | Repo | Role |
|-----------|------|------|
| `ratsim/` | Python SDK | TCP connector, message definitions, config blender, human control, visualization |
| `ratsim_unity_project/` | Unity project | Simulator: worldgen, sensors, actuators, TCP server |
| `ratsim_wildfire_gym_env/` | Gym environment | Gymnasium RL env wrapping the Unity sim via ratsim SDK |
| `ratsim_ros2/` | ROS 2 package | ROS 2 bridge to the Unity sim via ratsim SDK |
| `ratsim_experiments/` | Experiment automation | Train/test scripts, run definitions, method-invariant result recording |

## How the Repos Connect

```
Unity sim (TCP:9000)
    ├── ratsim (Python SDK) ← message defs, connector, config, human control
    │       ├── ratsim_wildfire_gym_env (Gym env) ← RL env wrapper
    │       ├── ratsim_ros2 (ROS 2 bridge) ← robotics integration
    │       └── ratsim_experiments ← train/test automation, imports gym env + SDK
```

- **ratsim** is the shared dependency. The Gym env and ROS2 package both import `ratsim.roslike_unity_connector`.
- **Message types** must stay in sync between Unity (`MessageDefs.cs` + `MessageRegistry.cs`) and Python (`message_definitions.py`). Python messages are auto-generated from C# via `generate_python_msgs.py` in `ratsim/`.
- **Config** (world + agent JSON) is authored via the config blender in `ratsim/config_blender/`, sent over TCP, and parsed by Unity's `WorldLoadingController`.

## Config Flow

Both world and agent configs use the same JSON entries format: `{"entries": [{"key": "k", "value": "v"}, ...]}`.

**World config** is sent each reset:
1. `train_ppo.py`: `blend_presets("world", ["default"])` → override dict → pass to env
2. `env.py` `reset()`: `to_entries_json(cfg)` → publish to `/sim_control/world_config` → publish reset

**Agent config** is sent once during env construction:
1. `train_ppo.py`: `blend_presets("agents", ["sphereagent_2d_lidar"])` → pass to env
2. `env.py` `__init__()`: `to_entries_json(agent_config)` → publish to `/sim_control/agent_config`
3. Unity: `WorldLoadingController` stores it; `AgentLoader.Generate()` reads it each episode

**Episode lifecycle in Unity:**
`StartEpisode()` → `ClearAllWorldData()` (providers `Clear()` in reverse dependency order) → `InitializeAllModules()` (providers `Generate()` in dependency order via topological sort, AgentLoader spawns agent here) → `ChunkLoadingRequestor.Tick()` (agent's requestor triggers chunk loading)

**Agent preset keys:** `prefab_name`, `name_prefix`, `sensors` (comma-separated), `actuators`, plus sensor/actuator param overrides like `lidar2d/maxRange` or `velocity/maxLinearVelocity`.

## Cross-Repo Change Patterns

**Adding a new message type:**
1. Unity: define class in `MessageDefs.cs`, register in `MessageRegistry.cs`
2. ratsim: regenerate Python messages (`generate_python_msgs.py`) or add manually to `message_definitions.py`
3. Gym/ROS2: use the new message type in env or bridge code

**Adding a new sensor:**
1. Unity: create sensor MonoBehaviour in `Assets/Sensors/`, publish on a topic. Store latest data in public fields for UI visualization.
2. Unity: register in `AgentLoader.SensorNameToType`
3. ratsim: ensure message type exists in Python
4. Gym env: subscribe to the topic in `env.py`, add to observation space
5. ROS2: map to appropriate ROS2 message type

**Changing world generation config:**
1. Unity: add param handling in the relevant `WorldDataProvider`
2. ratsim: add/update preset JSON in `config_blender/world_presets/`
3. Gym env: expose in `worldgen_config` dict in `curricula.py`

## Coordinate Convention

All Python/ROS code uses ROS standard: **x=forward, y=left, z=up**. Unity uses x=right, y=up, z=forward internally, but all sensors/actuators convert via `CoordConversion.cs` so that data crossing the TCP boundary is always in ROS frame.

**Message types for pose and motion:**
- `PoseMessage` (x, y, z, qx, qy, qz, qw) — position + orientation quaternion. Used by sensors (AbsolutePose2DSensor, RelativePoseSensor, Odom2DSensor), teleport actuator, and goal position publishing.
- `TwistMessage` (linear_x/y/z, angular_x/y/z) — velocity/acceleration commands. Used by Twist2DActuator for `/cmd_vel` and `/cmd_accel` topics.

Python quaternion/euler utilities live in `ratsim/transforms.py` (`quat_from_yaw`, `yaw_from_quat`, `quat_to_rpy`, `rpy_to_quat`).

## Running the Stack

1. Open Unity project and enter Play mode (or run a build) — listens on TCP:9000
2. From experiments repo: `python train.py def=<rundef> method=ppo` or `python test.py def=<rundef> model=<path>`
3. Human control testing: `python -m ratsim.human_control_test --world_preset default --rtf 1.0`
4. From ROS2: launch via `ros2 launch ratsim_ros2 <launch_file>`

## Running on a Headless / Cloud Box

### Bootstrap

`install.sh` at the meta-repo root clones sibling repos into `~/git/`, symlinks them into `meta_ratsim/`, creates two venvs (`~/ratvenv/venv` for SB3, `~/ratvenv/dreamer_venv` for DreamerV3), installs requirements for both, and editable-installs the ratsim packages. Auto-detects GPU via `nvidia-smi` unless `--gpu` / `--cpu` is forced. Idempotent; `--force` recreates venvs. Requires `python3-venv` and `python3-pip` apt packages — the script preflights this.

Note: `ratsim_experiments` is intentionally **not** pip-installed. It's a scripts directory (train.py, test.py); `cd` into it and run scripts directly. (Its `pyproject.toml` currently trips setuptools flat-layout discovery — worth fixing with an explicit packages list if we ever want to install it.)

### Headless display for the Unity binary

Unity needs an X server even in compute-only scenes (for now). `ratsim/scripts/setup_headless_display.sh` (run once, with sudo) sets up a virtual X server on `:99`:
- **NVIDIA path**: binds Xorg to the GPU's PCI bus; renders on the GPU.
- **Fallback path**: `xserver-xorg-video-dummy` + Mesa llvmpipe (CPU rendering).

The Xorg config is cached at `/etc/X11/xorg-ratsim.conf` — **re-run the setup script after driver changes**, otherwise it'll keep using the path it took on first run (e.g. llvmpipe from before the NVIDIA driver was installed). Verify with `DISPLAY=:99 glxinfo | grep renderer`.

`start_ratsim_headless.sh <binary>` attaches the Unity binary to `DISPLAY=:99` and waits for TCP:9000.

**Possible future improvement**: if no cameras are ever used in the build, adding `-batchmode -nographics` to the launcher lets Unity skip rendering entirely and drops the X-server dependency — worth trying for perf on compute-only runs.

### GPU vs CPU per method

(UNVERIFIED, POSSIBLY INCORRECT): For single-env RL (`n_envs=1`) the bottleneck is Unity stepping + Python overhead, not policy compute. CPU↔GPU transfer cost per step can dominate matmul savings for small policies. Measured on this project:

| Method | Best device | Why |
|---|---|---|
| PPO (small MLP, `n_envs=1`) | **CPU** | GPU transfer per step > forward-pass savings. Laptop CPU beats cloud GPU on same model. Pass `device="cpu"` to the SB3 `PPO(...)` constructor. |
| DreamerV3 | **GPU** | World model + actor-critic are large enough to amortize transfer. ~1.7x on `fps/train`, ~1.6x on `fps/policy`. Override via `method.jax.platform=cuda` (default in `train_dreamerv3.py:173` is `cpu`). |

`fps/policy` in Dreamer logs is the env step rate — capped by Unity single-env stepping regardless of GPU. `fps/train` is world-model update throughput.

### CUDA / driver compatibility

Version-matching here is fiddly and has bitten us repeatedly:

- **NVIDIA driver version → max supported CUDA**: driver 550 → CUDA 12.4, driver 545 → 12.3, driver 525 → 12.0. `nvidia-smi` shows both.
- **Torch** must match the driver. Default PyPI torch currently targets CUDA 12.6/12.8 (needs driver ≥560). For driver 550 use `pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision`; for older drivers use `cu118`.
- **JAX** `0.4.33` with `jax[cuda12]` bundles CUDA 12.3, so it needs driver ≥545. Works on 550 as-is.
- Error decode: torch's "found version NNNNN" = major·1000 + minor·10 + patch, so `12040` = CUDA 12.4.

### Viewing TensorBoard over SSH

`ratsim_experiments/tensorboard.sh` binds to `localhost:6006`. Forward via SSH rather than opening firewall ports:

```bash
ssh -N -f -L 6006:localhost:6006 user@<server>
# on server, inside ratsim_experiments: ./tensorboard.sh
# on laptop: http://localhost:6006
```

## Planning

See `roadmap.md` for current project goals, milestones, and TODOs.
