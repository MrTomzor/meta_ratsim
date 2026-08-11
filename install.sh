#!/usr/bin/env bash
# Bootstrap the ratsim stack on a fresh machine.
#
# Clones sibling repos into ~/git/, symlinks them into this meta-repo,
# creates two venvs under ~/ratvenv/ (main + dreamer), installs requirements,
# and editable-installs the ratsim packages.
#
# Idempotent: re-running skips what's already in place. Use --force to
# recreate venvs from scratch.
#
# Usage:
#   ./install.sh                    # auto-detect GPU, skip Unity + ROS2
#   ./install.sh --gpu              # force GPU JAX/TF install
#   ./install.sh --cpu              # force CPU JAX/TF install
#   ./install.sh --with-unity       # also clone ratsim_unity_project (~GB)
#   ./install.sh --with-ros2        # also clone ratsim_ros2
#   ./install.sh --force            # delete and recreate venvs
#
# Env overrides (for clusters / non-$HOME layouts):
#   RATSIM_GIT_DIR=/mnt/personal/$USER/git
#   RATSIM_VENV_DIR=/mnt/personal/$USER/ratvenv
set -euo pipefail

# Overridable so the stack can live off $HOME — needed on HPC clusters where the
# home filesystem is small/slow and data belongs elsewhere (e.g. RCI's
# /mnt/personal/$USER). Defaults keep laptop behaviour unchanged.
GIT_DIR="${RATSIM_GIT_DIR:-${HOME}/git}"
META_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${RATSIM_VENV_DIR:-${HOME}/ratvenv}"
MAIN_VENV="${VENV_DIR}/venv"
DREAMER_VENV="${VENV_DIR}/dreamer_venv"

REPOS=(
  "ratsim"
  "ratsim_wildfire_gym_env"
  "ratsim_experiments"
)

WITH_UNITY=0
WITH_ROS2=0
FORCE=0
GPU_MODE="auto"  # auto | gpu | cpu

for arg in "$@"; do
  case "$arg" in
    --with-unity) WITH_UNITY=1 ;;
    --with-ros2) WITH_ROS2=1 ;;
    --force) FORCE=1 ;;
    --gpu) GPU_MODE="gpu" ;;
    --cpu) GPU_MODE="cpu" ;;
    -h|--help)
      sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "Unknown arg: $arg" >&2; exit 1 ;;
  esac
done

[[ "$WITH_UNITY" == 1 ]] && REPOS+=("ratsim_unity_project")
[[ "$WITH_ROS2" == 1 ]] && REPOS+=("ratsim_ros2")

# ----- Preflight: python3 must be new enough AND have venv ------------------
# Version floor first: ratsim_wildfire_gym_env declares requires-python >=3.9,
# and an old interpreter passes the venv check below while failing later in
# confusing ways (CentOS 7 ships 3.6.8, which does have venv).
MIN_PY_MINOR=9
if ! python3 -c "import sys; sys.exit(0 if sys.version_info[:2] >= (3, ${MIN_PY_MINOR}) else 1)" 2>/dev/null; then
  cat >&2 <<EOF
[install] ERROR: python3 is too old ($(python3 -V 2>&1)); need >= 3.${MIN_PY_MINOR}.
On an HPC cluster, load a newer one from the module system, e.g.:
    module avail Python            # list what's there
    ml Python/3.12.3-GCCcore-13.3.0
Then re-run this script. Use the IDENTICAL ml line in every job script — a venv
symlinks the interpreter that created it and breaks confusingly without it.
On Debian/Ubuntu, install a newer python3 from your package manager instead.
EOF
  exit 1
fi

if ! python3 -c "import venv, ensurepip" 2>/dev/null; then
  PYVER="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  cat >&2 <<EOF
[install] ERROR: python3 venv/ensurepip module is missing.
On Debian/Ubuntu, install it with:
    sudo apt update && sudo apt install -y python3-venv python3.${PYVER#*.}-venv python3-pip
(You may only need one of the two venv packages depending on your distro.)
On an HPC cluster without root, load a module-provided Python instead:
    ml Python/3.12.3-GCCcore-13.3.0
EOF
  exit 1
fi

# ----- GPU detection --------------------------------------------------------
if [[ "$GPU_MODE" == "auto" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    GPU_MODE="gpu"
    echo "[install] Detected NVIDIA GPU → installing CUDA JAX + TF."
  else
    GPU_MODE="cpu"
    echo "[install] No GPU detected → installing CPU JAX + TF."
  fi
else
  echo "[install] GPU mode forced: $GPU_MODE"
fi

# ----- Clone repos into ~/git/ ---------------------------------------------
mkdir -p "$GIT_DIR"
clone_if_missing() {
  local name="$1" url="$2"
  if [[ -d "$GIT_DIR/$name/.git" ]]; then
    echo "[install] $name already cloned, skipping."
  else
    echo "[install] Cloning $name..."
    git clone "$url" "$GIT_DIR/$name"
  fi
}

for repo in "${REPOS[@]}"; do
  clone_if_missing "$repo" "https://github.com/MrTomzor/${repo}.git"
done
clone_if_missing "dreamerv3" "https://github.com/danijar/dreamerv3.git"

# ----- Symlink into meta_ratsim/ -------------------------------------------
for repo in "${REPOS[@]}"; do
  link="$META_DIR/$repo"
  target="$GIT_DIR/$repo"
  if [[ -L "$link" ]]; then
    current="$(readlink "$link")"
    # Compare resolved paths, not link text: the repo commits *relative* links
    # (../ratsim). A literal comparison against the absolute $target would
    # rewrite them every run, dirtying the working tree and breaking git pull.
    if [[ "$current" == "$target" ]] || [[ "$(readlink -f "$link")" == "$(readlink -f "$target")" ]]; then
      echo "[install] Symlink $repo already correct, skipping."
    else
      echo "[install] Repointing symlink $repo: $current -> $target"
      ln -sfn "$target" "$link"
    fi
  elif [[ -e "$link" ]]; then
    echo "[install] WARN: $link exists and is not a symlink, leaving alone."
  else
    echo "[install] Linking $repo -> $target"
    ln -s "$target" "$link"
  fi
done

# ----- Create / recreate venvs ---------------------------------------------
mkdir -p "$VENV_DIR"
create_venv() {
  local path="$1"
  if [[ -d "$path" ]] && [[ "$FORCE" == 1 ]]; then
    echo "[install] --force: removing $path"
    rm -rf "$path"
  fi
  # Treat a dir with no bin/activate as a broken partial venv and recreate.
  if [[ -d "$path" ]] && [[ ! -f "$path/bin/activate" ]]; then
    echo "[install] Found partial venv at $path (no bin/activate); recreating."
    rm -rf "$path"
  fi
  if [[ ! -d "$path" ]]; then
    echo "[install] Creating venv $path"
    python3 -m venv "$path"
  else
    echo "[install] Venv $path already exists, skipping create."
  fi
  if [[ ! -f "$path/bin/activate" ]]; then
    echo "[install] ERROR: venv at $path is missing bin/activate." >&2
    echo "[install] python3 -m venv did not complete. Check python3-venv is installed." >&2
    exit 1
  fi
}
create_venv "$MAIN_VENV"
create_venv "$DREAMER_VENV"

# ----- Main venv (SB3 / ratsrc) --------------------------------------------
echo ""
echo "[install] === Main venv ==="
# shellcheck disable=SC1091
source "$MAIN_VENV/bin/activate"
pip install --upgrade pip

# torch arrives as a transitive dep of stable_baselines3[extra], which would take
# it from the DEFAULT PyPI index. That is not safe: torch 2.13 defaults to CUDA 13
# wheels, which need driver >= 580. On an older driver you get a silent
# torch.cuda.is_available() == False. So pin the CUDA variant up front, and let
# SB3 then see torch as already satisfied.
#
# CPU mode is deliberately left alone so laptop behaviour is unchanged.
if [[ "$GPU_MODE" == "gpu" ]]; then
  TORCH_INDEX="${RATSIM_TORCH_INDEX:-}"
  DRV_MAJ=""
  if [[ -z "$TORCH_INDEX" ]]; then
    # nvidia-smi is often PRESENT on GPU-less nodes and fails at runtime, printing
    # its error to STDOUT (not stderr) -- so test that it actually works, and then
    # accept the result only if it is purely numeric. Without both guards the
    # error text lands in DRV_MAJ and the (( )) comparisons below abort the script.
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
      DRV_MAJ="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f1)"
    fi
    [[ "$DRV_MAJ" =~ ^[0-9]+$ ]] || DRV_MAJ=""

    # GPU compute capability matters as much as the driver, because the choice is
    # constrained from BOTH ends:
    #   - too NEW a wheel DROPS old GPU archs. Measured: cu128 ships only
    #     CC >= 9.0, so on a V100 (sm_70) torch.cuda.is_available() is True and
    #     the first matmul dies with "no kernel image is available".
    #   - too NEW a CUDA needs a newer DRIVER (cu130 wants driver >= 580).
    # cu126 ships sm_50..sm_90 and needs only driver >= 525, so it is the safe
    # default for anything up to Hopper. Blackwell (sm_100+) needs cu128/cu130.
    CC_MAJ=""
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
      CC_MAJ="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f1)"
      [[ "$CC_MAJ" =~ ^[0-9]+$ ]] || CC_MAJ=""
    fi

    if [[ -z "$DRV_MAJ" ]]; then
      # Happens when installing on a CPU node for later use on GPU nodes (HPC).
      echo "[install] WARN: --gpu forced but nvidia-smi is unavailable here."
      echo "[install]       Defaulting torch to cu126 (widest GPU arch coverage)."
      echo "[install]       Override with RATSIM_TORCH_INDEX=<url|default> for Blackwell."
      TORCH_INDEX="https://download.pytorch.org/whl/cu126"
    elif (( DRV_MAJ < 525 )); then  TORCH_INDEX="https://download.pytorch.org/whl/cu118"
    elif [[ -n "$CC_MAJ" ]] && (( CC_MAJ >= 10 )) && (( DRV_MAJ >= 580 )); then
      TORCH_INDEX="default"
    elif [[ -n "$CC_MAJ" ]] && (( CC_MAJ >= 9 )) && (( DRV_MAJ >= 550 )); then
      TORCH_INDEX="https://download.pytorch.org/whl/cu128"
    elif (( DRV_MAJ >= 525 )); then TORCH_INDEX="https://download.pytorch.org/whl/cu126"
    fi
  fi
  if [[ "$TORCH_INDEX" == "default" || "$TORCH_INDEX" == "none" ]]; then
    echo "[install] torch: using default PyPI index (driver ${DRV_MAJ:-forced})."
  else
    echo "[install] torch: $TORCH_INDEX (driver ${DRV_MAJ:-unknown})"
    pip install --index-url "$TORCH_INDEX" torch
  fi
fi

pip install \
  scipy \
  scikit-learn \
  gymnasium \
  "stable_baselines3[extra]" \
  sb3-contrib \
  tensorboard \
  wandb
pip install -e "$GIT_DIR/ratsim"
pip install --no-deps -e "$GIT_DIR/ratsim_wildfire_gym_env"
# ratsim_experiments is a scripts dir (train.py / test.py), not a package —
# run scripts from that directory, no pip install needed.
deactivate

# ----- Dreamer venv --------------------------------------------------------
echo ""
echo "[install] === Dreamer venv ==="
# shellcheck disable=SC1091
source "$DREAMER_VENV/bin/activate"
pip install --upgrade pip

# Install requirements.txt *without* jax/tensorflow — we inject the right
# variant (cpu vs cuda12) below so the same file works for both.
DREAMER_REQ="$META_DIR/ratsim_experiments/methods/dreamerv3/requirements.txt"
FILTERED_REQ="$(mktemp)"
grep -viE '^(jax==|tensorflow-?cpu==|tensorflow==)' "$DREAMER_REQ" > "$FILTERED_REQ"
pip install -r "$FILTERED_REQ"
rm -f "$FILTERED_REQ"

if [[ "$GPU_MODE" == "gpu" ]]; then
  pip install "jax[cuda12]==0.4.33"
  pip install "tensorflow==2.19.1"
else
  pip install "jax==0.4.33"
  pip install "tensorflow-cpu==2.19.1"
fi

pip install --no-deps -e "$GIT_DIR/dreamerv3"
pip install --no-deps -e "$GIT_DIR/ratsim"
pip install --no-deps -e "$GIT_DIR/ratsim_wildfire_gym_env"
deactivate

# ----- Final notes ----------------------------------------------------------
cat <<EOF

[install] Done.

Add these to your ~/.bashrc (or ~/.zshrc) if you want shortcuts:

    alias ratsrc='source ${MAIN_VENV}/bin/activate'
    alias drsrc='source ${DREAMER_VENV}/bin/activate'

Main venv:    ${MAIN_VENV}
Dreamer venv: ${DREAMER_VENV}
Mode:         ${GPU_MODE}
EOF
