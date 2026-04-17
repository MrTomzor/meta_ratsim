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
set -euo pipefail

GIT_DIR="${HOME}/git"
META_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${HOME}/ratvenv"
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

# ----- Preflight: python3-venv must be functional ---------------------------
if ! python3 -c "import venv, ensurepip" 2>/dev/null; then
  PYVER="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  cat >&2 <<EOF
[install] ERROR: python3 venv/ensurepip module is missing.
On Debian/Ubuntu, install it with:
    sudo apt update && sudo apt install -y python3-venv python3.${PYVER#*.}-venv python3-pip
(You may only need one of the two venv packages depending on your distro.)
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
    echo "[install] Symlink $repo already exists, skipping."
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
pip install \
  scipy \
  scikit-learn \
  gymnasium \
  "stable_baselines3[extra]" \
  sb3-contrib \
  tensorboard
pip install -e "$GIT_DIR/ratsim"
pip install --no-deps -e "$GIT_DIR/ratsim_wildfire_gym_env"
pip install --no-deps -e "$GIT_DIR/ratsim_experiments"
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
pip install --no-deps -e "$GIT_DIR/ratsim_experiments"
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
