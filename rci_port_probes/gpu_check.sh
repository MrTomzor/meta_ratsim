#!/bin/bash
set -uo pipefail
. "$HOME/rci_env.sh"
echo "host=$(hostname)"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

echo "=== main venv: torch ==="
ratsim_activate
python - <<'PY'
import torch
print("torch:", torch.__version__, "| built for CUDA:", torch.version.cuda)
ok = torch.cuda.is_available()
print("cuda available:", ok)
if ok:
    import torch as t
    print("device:", t.cuda.get_device_name(0))
    a = t.randn(512,512,device='cuda')
    print("matmul finite:", bool(t.isfinite((a@a).sum())))
    print("bf16 supported:", t.cuda.is_bf16_supported())
else:
    print("!! GPU UNUSABLE")
import stable_baselines3, sb3_contrib, gymnasium, ratsim
print("sb3:", stable_baselines3.__version__, "| gym:", gymnasium.__version__, "| ratsim OK")
PY
deactivate

echo "=== dreamer venv: jax ==="
ratsim_activate_dreamer
python - <<'PY' 2>/dev/null
import jax, dreamerv3
print("jax:", jax.__version__, "devices:", jax.devices())
print("dreamerv3 import OK")
PY
deactivate
echo "=== GPUCHECK DONE ==="
