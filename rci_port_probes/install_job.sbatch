#!/bin/bash
# Install / update the ratsim stack on RCI.  Submit with:  sbatch install_job.sbatch
#
# Runs on a CPU node deliberately. login1 is CentOS 7 (glibc 2.17) and pip
# resolves wheels for whatever machine it runs on, so installing there pulls
# wheels the compute nodes (glibc 2.28) can't use. Compute nodes have internet.
#SBATCH --partition=cpufast
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --job-name=ratsim-install
#SBATCH --output=/mnt/personal/%u/logs/install-%j.out
set -euo pipefail

# Sets up Lmod (which SLURM jobs do NOT get by default), modules, and paths.
. "$HOME/rci_env.sh"

echo "host=$(hostname) python=$(python3 -V 2>&1) glibc=$(ldd --version | head -1 | awk '{print $NF}')"

cd "$RATSIM_GIT_DIR/meta_ratsim"
# --gpu forced: this is a CPU node, so auto-detect would install CPU JAX/TF
# wheels even though the training nodes have GPUs. install.sh handles the
# "no nvidia-smi here" case by defaulting torch to cu126.
./install.sh --gpu

echo "=== INSTALL EXIT OK ==="
