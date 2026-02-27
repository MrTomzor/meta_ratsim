# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working across the ratsim project.

## Meta-Repo Structure

This is a meta-repo containing symlinks to four independent git repositories that together form the ratsim simulation framework. Each repo has its own CLAUDE.md with detailed architecture docs.

| Directory | Repo | Role |
|-----------|------|------|
| `ratsim/` | Python SDK | TCP connector, message definitions, config blender, visualization |
| `ratsim_unity_project/` | Unity project | Simulator: worldgen, sensors, actuators, TCP server |
| `ratsim_wildfire_gym_env/` | Gym environment | Gymnasium RL env wrapping the Unity sim via ratsim SDK |
| `ratsim_ros2/` | ROS 2 package | ROS 2 bridge to the Unity sim via ratsim SDK |

## How the Repos Connect

```
Unity sim (TCP:9000)
    ├── ratsim (Python SDK) ← message defs, connector, config
    │       ├── ratsim_wildfire_gym_env (Gym env) ← RL training
    │       └── ratsim_ros2 (ROS 2 bridge) ← robotics integration
```

- **ratsim** is the shared dependency. The Gym env and ROS2 package both import `ratsim.roslike_unity_connector`.
- **Message types** must stay in sync between Unity (`MessageDefs.cs` + `MessageRegistry.cs`) and Python (`message_definitions.py`). Python messages are auto-generated from C# via `generate_python_msgs.py` in `ratsim/`.
- **Config** (world + agent JSON) is authored via the config blender in `ratsim/config_blender/`, sent over TCP, and parsed by Unity's `WorldLoadingController`.

## Cross-Repo Change Patterns

**Adding a new message type:**
1. Unity: define class in `MessageDefs.cs`, register in `MessageRegistry.cs`
2. ratsim: regenerate Python messages (`generate_python_msgs.py`) or add manually to `message_definitions.py`
3. Gym/ROS2: use the new message type in env or bridge code

**Adding a new sensor:**
1. Unity: create sensor MonoBehaviour in `Assets/Sensors/`, publish on a topic
2. ratsim: ensure message type exists in Python
3. Gym env: subscribe to the topic in `env.py`, add to observation space
4. ROS2: map to appropriate ROS2 message type

**Changing world generation config:**
1. Unity: add param handling in the relevant `WorldLoadingModule`
2. ratsim: add/update preset JSON in `config_blender/world_presets/`
3. Gym env: expose in `worldgen_config` dict in `curricula.py`

## Running the Stack

1. Open Unity project and enter Play mode (or run a build) — listens on TCP:9000
2. From the Gym env: `python train_ppo.py` or `python test_manual_control.py`
3. From ROS2: launch via `ros2 launch ratsim_ros2 <launch_file>`

## Planning

See `roadmap.md` for current project goals, milestones, and TODOs.
