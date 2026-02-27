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

**CURRENT TODOS:**

- [x]  **UNITY - WorldStructure loading standardization**
    - [x]  generalize structures (city can generate house structures) → gets caught by StructureObjectLoaderSimple (structure has either prefab OR just label?)
- [x]  **Standardize config loading**
    - [x]  UNITY - add loading from MSG (json) or from file (path given thru a msg on some topic). The config loading should be additive and refresh when new configs are sent? - thus also we might want to generalize the WorldLoadingModule to have some function OnWorldConfigChanged, which is by default called in Awake, but also triggered by the Controller if a json with new config param values is received? (Controller might get a config AND a world reset request. Usually, e.g. during RL training, it will just receive a new seed as the only config, this should however also probably propagate to the loading modules).
    - [x]  UNITY - make the new worldgen loading controller be as controllable as the old WildfireWorldManager (through the generation request topic AND the new form of sending the config through a Json). (separate topic for world and agent config).
    - [x]  META - move the current world config file from Unity to the config blender in ratsim, name it e.g. default.json.
    - [x]  META - create some basic agent config which will reflect the current agent (named rat1, with a lidar2d sensor, odom sensor), name should say sthg like default_lidar2d
    - [x]  Ratsim - implement the config blender - it should allow stacking a given set of presets for both world and agent configuration.
    - [x]  UNITY - AgentLoader WorldLoadingModule: spawns agent prefabs from Resources based on agent config, manages sensor enable/disable + param overrides via reflection, finds safe spawn positions. Uses Initialize() lifecycle hook (called before chunk loading) to solve the agent-carries-ChunkLoadingRequestor dependency.
    - [x]  GYM ENV - env.py accepts agent_config dict, sends it to Unity via /sim_control/agent_config before first reset. train_ppo.py loads agent preset via blend_presets("agents", [...]).
    - [x]  COMPLETION = can run the train_ppo.py with the config being created in train_ppo.py using the config blender.
- [ ]  **MSG RESTRUCTURING TO 3D** - transfer from 2D Twist messages into general Pose messages (will save us a lot of pain when developing both for 2D and 3D navigation scenarios)
    - [ ]  UNITY - remove the Twist2D, replace with Pose msg and CORRECTLY handle the difference between Unity’s coordinate system (z=forward, x=right, y=up) and the one used in robotics (x=forward, y=left, z=up)
    - [ ]  UNITY - change all usage of the Twist2D to Pose (e.g. in odom sensor, but also all others).
    - [ ]  RATSIM (human input to run script requied?) - regenerate the msgs
    - [ ]  Ratsim ROS2 - correctly map any ratsim Pose msgs to ROS2 pose msgs
    - [ ]  RL - since we are just dealing with RL in 2D, just map to 2D in the same way that we used the 2D msg
    - [ ]  COMPLETION = can run the train_ppo.py script
- [ ]  **ROS2 package restructuring/generalization — should allow multiple agents!**
