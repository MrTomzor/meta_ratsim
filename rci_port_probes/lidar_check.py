"""Verify lidar data from a running Unity instance is non-degenerate.

A sim can happily stream lidar messages that are all-zero or all-maxRange when
rendering is broken, so "messages arrive" is not sufficient evidence that the
headless mode is sound. This checks the distribution of returned ranges.
"""
import sys
import numpy as np

from ratsim.roslike_unity_connector.connector import RoslikeUnityConnector
from ratsim.roslike_unity_connector.message_definitions import (
    StringMessage, BoolMessage, Lidar2DMessage)
from ratsim.config_blender import blend_presets, to_entries_json

WORLD = sys.argv[1] if len(sys.argv) > 1 else "maze_memorymaze_11x11_wells"
AGENT = sys.argv[2] if len(sys.argv) > 2 else "sphereagent_2d_lidar_wells"
STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 60

world_config = blend_presets("world", [WORLD])
agent_config = blend_presets("agents", [AGENT])

conn = RoslikeUnityConnector(verbose=False)
conn.connect()
conn.publish(StringMessage(data="Wildfire"), "/sim_control/scene_select")
conn.send_messages_and_step(enable_physics_step=False)
conn.read_messages_from_unity()
conn.publish(StringMessage(data=to_entries_json(agent_config)), "/sim_control/agent_config")
conn.send_messages_and_step(enable_physics_step=False)
conn.read_messages_from_unity()
# Mirror fps_test.run_fps_test exactly: world_config + reset published
# together, then stepped WITH physics enabled.
conn.publish(StringMessage(data=to_entries_json(world_config)), "/sim_control/world_config")
conn.publish(BoolMessage(data=True), "/sim_control/reset_episode")
conn.send_messages_and_step(enable_physics_step=True)
conn.read_messages_from_unity()
conn.process_worldgen_status()

seen_topics = set()
scans, descriptor_sets = [], []
for _ in range(STEPS):
    conn.send_messages_and_step(enable_physics_step=True)
    msgs = conn.read_messages_from_unity()   # {topic: [Message, ...]}
    for _topic, msg_list in (msgs or {}).items():
        seen_topics.add(_topic)
        for m in msg_list:
            if isinstance(m, Lidar2DMessage) and m.ranges:
                scans.append(np.asarray(m.ranges, dtype=float))
                if m.descriptors:
                    descriptor_sets.append(np.asarray(m.descriptors, dtype=float))

if not scans:
    print("FAIL: no Lidar2DMessage received at all")
    print("topics seen:", sorted(seen_topics))
    raise SystemExit(1)

arr = np.vstack(scans)
mx = float(np.nanmax(arr))
finite = arr[np.isfinite(arr)]
# A scan that is entirely maxRange means every ray escaped -> geometry missing.
frac_at_max = float(np.mean(np.isclose(finite, mx, rtol=1e-3)))
frac_zero = float(np.mean(finite == 0.0))

print(f"scans captured:      {len(scans)}  (beams/scan = {arr.shape[1]})")
print(f"range min/mean/max:  {finite.min():.3f} / {finite.mean():.3f} / {mx:.3f}")
print(f"unique values:       {len(np.unique(np.round(finite, 3)))}")
print(f"fraction at max:     {frac_at_max:.3f}")
print(f"fraction exactly 0:  {frac_zero:.3f}")
if descriptor_sets:
    d = np.concatenate(descriptor_sets)
    print(f"semantic classes:    {sorted(set(d.tolist()))}")

ok = (len(np.unique(np.round(finite, 3))) > 10) and frac_at_max < 0.95 and frac_zero < 0.5
print()
print("VERDICT:", "PASS - lidar returns varied geometry" if ok
      else "FAIL - lidar looks degenerate")
raise SystemExit(0 if ok else 1)
