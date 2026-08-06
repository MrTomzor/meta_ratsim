#!/usr/bin/env bash
# Can this machine host a headless ratsim Unity instance?
#
# Designed to be run on an RCI compute node inside an interactive SLURM job:
#
#     ssh username@login1.rci.cvut.cz
#     srun -p gpufast --gres=gpu:1 --pty bash -i
#     ./check_cluster_node.sh                      # capability checks only
#     ./check_cluster_node.sh /path/to/Build.x86_64  # + definitive launch test
#
# TIER 1 (no build needed) reports what the node can do.
# TIER 2 (build path given) actually launches Unity under xvfb and is the only
# conclusive answer -- see the note on ldd below for why static checks lie.
#
# Exit code 0 = the definitive test passed (or tier 1 found no blockers when
# run without a build). Non-zero = something needs attention. Read the output.
set -u

BUILD="${1:-}"
PASS=0; WARN=0; FAIL=0
say()  { printf '%s\n' "$*"; }
ok()   { printf '  [ OK ]   %s\n' "$*"; PASS=$((PASS+1)); }
warn() { printf '  [WARN]   %s\n' "$*"; WARN=$((WARN+1)); }
bad()  { printf '  [FAIL]   %s\n' "$*"; FAIL=$((FAIL+1)); }
info() { printf '  .        %s\n' "$*"; }

say "==============================================="
say " ratsim cluster-node capability probe"
say "==============================================="
say "host:      $(hostname)"
say "date:      $(date -Is 2>/dev/null || date)"
say "SLURM job: ${SLURM_JOB_ID:-<not in a slurm job>}"
say "partition: ${SLURM_JOB_PARTITION:-<none>}"
say ""

# --- 1. Are we somewhere we're allowed to compute? -------------------------
say "1. Execution context"
case "$(hostname)" in
  login*) bad "You are on a LOGIN node. Never run training here."
          info "Get a compute node: srun -p gpufast --gres=gpu:1 --pty bash -i" ;;
  *)      if [ -n "${SLURM_JOB_ID:-}" ]; then ok "on compute node inside SLURM job ${SLURM_JOB_ID}"
          else warn "not inside a SLURM allocation -- results may not reflect a real job"; fi ;;
esac
say ""

# --- 2. Virtual X server ---------------------------------------------------
# This is the crux: Unity -nographics still needs a REAL X server (measured;
# see RCI_CLUSTER_PORT.md section 1). xvfb-run provides one without root.
say "2. Virtual X server (xvfb) -- the critical dependency"
XVFB_OK=0

# RCI provides Xvfb as an Lmod module, so it is NOT on PATH by default.
# Try to load it before declaring it missing. Lmod exports `module` as a shell
# function to child shells, but fall back to $LMOD_CMD if that didn't survive.
if ! command -v Xvfb >/dev/null 2>&1; then
  if command -v module >/dev/null 2>&1; then
    module load Xvfb >/dev/null 2>&1 || true
  elif [ -n "${LMOD_CMD:-}" ]; then
    eval "$("$LMOD_CMD" bash load Xvfb 2>/dev/null)" || true
  fi
  command -v Xvfb >/dev/null 2>&1 && info "loaded the Xvfb module (remember 'ml Xvfb' in your sbatch script)"
fi

# Only `Xvfb` itself is essential. `xvfb-run` is a convenience wrapper shipped
# by the Debian package; module builds often omit it. Starting Xvfb by hand and
# exporting DISPLAY is equivalent and is a measured-working path.
XVFB_RUN=0
if command -v Xvfb >/dev/null 2>&1; then
  ok "Xvfb present ($(command -v Xvfb))"
  XVFB_OK=1
  if command -v xvfb-run >/dev/null 2>&1; then
    XVFB_RUN=1
    info "xvfb-run wrapper also available"
  else
    warn "xvfb-run wrapper absent -- start Xvfb manually instead (equivalent):"
    info "    Xvfb :99 -screen 0 1024x768x24 -nolisten tcp &"
    info "    DISPLAY=:99 <build> -batchmode -nographics -port <p>"
  fi
else
  bad "Xvfb not found on PATH (and 'module load Xvfb' did not provide it)"
  info "Find the exact module name: module spider Xvfb"
  info "Note: 'module avail | grep Xvfb' shows column neighbours too -- the"
  info "      grep-highlighted word is the real match, the rest is noise."
  info "If no module provides it, use a Singularity container carrying xvfb."
fi

# Prove Xvfb can actually START here (present != usable; some nodes lack
# /tmp/.X11-unix or forbid the lock files).
if [ "$XVFB_OK" = 1 ]; then
  XVFB_TEST_DISPLAY=":$(( 90 + (RANDOM % 9) ))"
  Xvfb "$XVFB_TEST_DISPLAY" -screen 0 640x480x24 -nolisten tcp >/dev/null 2>&1 &
  XPID=$!
  sleep 2
  if kill -0 "$XPID" 2>/dev/null; then
    ok "Xvfb starts successfully on $XVFB_TEST_DISPLAY"
    kill -9 "$XPID" 2>/dev/null
  else
    bad "Xvfb is installed but FAILED to start -- container route likely needed"
    XVFB_OK=0
  fi
  wait "$XPID" 2>/dev/null
fi
say ""

# --- 3. Scratch / TMPDIR ---------------------------------------------------
# Pidfiles and Unity logs must not collide between jobs sharing a node.
say "3. Per-job scratch (pidfile/log isolation)"
if [ -n "${TMPDIR:-}" ] && [ "$TMPDIR" != "/tmp" ]; then
  ok "TMPDIR=$TMPDIR (per-job -- use this for pidfiles/logs)"
elif [ -n "${SLURM_JOB_ID:-}" ] && [ -d "/mnt/job-${SLURM_JOB_ID}" ]; then
  # RCI's documented per-job node-local scratch, cleaned up when the job ends.
  ok "/mnt/job-${SLURM_JOB_ID} exists (per-job scratch -- use this for pidfiles/logs)"
  info "TMPDIR is unset, so set it yourself: export TMPDIR=/mnt/job-\$SLURM_JOB_ID"
elif [ -d "/data/temporary" ]; then
  warn "TMPDIR unset/=/tmp, but /data/temporary exists -- use that, scoped by \$SLURM_JOB_ID"
else
  warn "No per-job TMPDIR and no /data/temporary; scope paths by \$SLURM_JOB_ID manually"
fi
[ -w "/tmp" ] && info "/tmp writable (shared with other users' jobs -- do NOT use bare /tmp/ratsim_<port>.pid)"
say ""

# --- 4. GPU / driver -------------------------------------------------------
say "4. GPU and driver"
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
  GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
  NGPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
  ok "$NGPU GPU(s) visible: $GPU (driver $DRV)"
  DRVMAJ=${DRV%%.*}
  if   [ "${DRVMAJ:-0}" -ge 560 ] 2>/dev/null; then info "driver >=560 -> default PyPI torch (cu126/128) is fine"
  elif [ "${DRVMAJ:-0}" -ge 550 ] 2>/dev/null; then info "driver 550 -> torch cu124: pip install --index-url https://download.pytorch.org/whl/cu124 torch"
  elif [ "${DRVMAJ:-0}" -ge 545 ] 2>/dev/null; then info "driver 545 -> torch cu123-ish; jax[cuda12] 0.4.33 OK (bundles 12.3)"
  else info "driver <545 -> use torch cu118; jax[cuda12] 0.4.33 will NOT work"; fi
else
  warn "no GPU visible (fine for CPU-only runs; on a gpu partition you must pass --gres=gpu:N)"
fi
say ""

# --- 5. Python -------------------------------------------------------------
say "5. Python / venv"
if python3 -c "import venv, ensurepip" >/dev/null 2>&1; then
  ok "python3 $(python3 -V 2>&1 | awk '{print $2}') has venv+ensurepip (install.sh preflight will pass)"
else
  warn "current python3 lacks venv/ensurepip -- load a module first"
  info "module avail Python   # then: ml Python/<version>"
fi
command -v module >/dev/null 2>&1 || [ -n "${LMOD_CMD:-}" ] \
  && ok "module system available" || warn "no 'module' command found"
say ""

# --- 6. Networking ---------------------------------------------------------
say "6. Networking"
if command -v ss >/dev/null 2>&1; then ok "ss present (start_ratsim_headless.sh requires it)"
else bad "ss missing (iproute2) -- start_ratsim_headless.sh will refuse to run"; fi
INUSE=$(ss -tln 2>/dev/null | grep -cE ':(9[0-9]{3}) ' || true)
if [ "${INUSE:-0}" -gt 0 ]; then
  warn "$INUSE listener(s) already on ports 9000-9999 on this node -- port collision risk is REAL here"
  ss -tln 2>/dev/null | grep -E ':(9[0-9]{3}) ' | sed 's/^/           /'
else
  ok "no listeners on 9000-9999 right now"
fi
if timeout 5 bash -c 'exec 3<>/dev/tcp/pypi.org/443' 2>/dev/null; then
  ok "outbound internet works from this node (pip install possible here)"
else
  warn "no outbound internet from this node -- run install.sh/pip on login1 instead"
fi
say ""

# --- 7. Definitive test ----------------------------------------------------
say "7. Definitive launch test"
if [ -z "$BUILD" ]; then
  info "skipped -- no build path given."
  info "Re-run as: $0 /mnt/personal/\$USER/ForagerSimBuildV1.x86_64"
elif [ ! -x "$BUILD" ]; then
  bad "not an executable file: $BUILD"
elif [ "$XVFB_OK" != 1 ]; then
  bad "skipping launch test -- no working xvfb (fix that first)"
else
  # NOTE: `ldd` is NOT a valid check here. Unity dlopen()s libX11/libGL at
  # runtime, so ldd reports clean even on a node that cannot run the binary.
  # Launching it is the only real test.
  PORT=$(( 9300 + (${SLURM_JOB_ID:-$$} % 300) ))
  LOG="${TMPDIR:-/tmp}/ratsim_probe_$$.log"
  OWN_XPID=""
  if [ "$XVFB_RUN" = 1 ]; then
    info "launching under xvfb-run on port $PORT (log: $LOG)"
    setsid xvfb-run -a "$BUILD" -batchmode -nographics -port "$PORT" -logFile "$LOG" \
      >/dev/null 2>&1 &
  else
    # No wrapper: run our own Xvfb and point DISPLAY at it. Same thing.
    PROBE_DISP=":$(( 80 + (RANDOM % 10) ))"
    info "no xvfb-run; starting Xvfb on $PROBE_DISP, launching on port $PORT (log: $LOG)"
    Xvfb "$PROBE_DISP" -screen 0 1024x768x24 -nolisten tcp >/dev/null 2>&1 &
    OWN_XPID=$!
    sleep 3
    if ! kill -0 "$OWN_XPID" 2>/dev/null; then
      bad "our Xvfb died immediately on $PROBE_DISP"; OWN_XPID=""
    fi
    DISPLAY="$PROBE_DISP" setsid "$BUILD" -batchmode -nographics -port "$PORT" \
      -logFile "$LOG" >/dev/null 2>&1 &
  fi

  # Unity's comm name is the basename truncated to 15 chars (kernel limit).
  # Match on that with pgrep -x; never `pgrep -f <buildname>`, which also
  # matches THIS script (our own argv contains the build path).
  COMM=$(basename "${BUILD%.x86_64}" | cut -c1-15)
  UPID=""
  for _ in $(seq 1 10); do
    UPID=$(pgrep -n -x "$COMM" 2>/dev/null || true)
    [ -n "$UPID" ] && break
    sleep 1
  done
  [ -z "$UPID" ] && info "warning: could not resolve Unity pid (comm '$COMM'); relying on port+log only"

  UP=0; i=0
  for i in $(seq 1 40); do
    sleep 1
    if ss -tln 2>/dev/null | grep -q ":${PORT} "; then UP=1; break; fi
    if [ -n "$UPID" ] && ! kill -0 "$UPID" 2>/dev/null; then break; fi
  done

  # `grep -c` already prints 0 when there are no matches; it just exits 1.
  # Piping it through `|| echo 0` would emit TWO lines and break the -eq tests.
  SEGV=$(grep -c 'Caught fatal signal' "$LOG" 2>/dev/null)
  SEGV=${SEGV:-0}
  if [ "$UP" = 1 ] && [ "$SEGV" -eq 0 ]; then
    ok "Unity booted headless and opened TCP :$PORT after ${i}s -- THIS NODE WORKS"
  elif [ "$SEGV" -gt 0 ]; then
    bad "Unity SEGFAULTED under xvfb (${SEGV} fatal signal(s) in log)"
    info "last 15 log lines:"; tail -15 "$LOG" 2>/dev/null | sed 's/^/           /'
  else
    bad "Unity never opened port :$PORT (timeout)"
    info "last 15 log lines:"; tail -15 "$LOG" 2>/dev/null | sed 's/^/           /'
  fi

  # Clean up: kill Unity, then any Xvfb we spawned. xvfb-run's child survives
  # killing the wrapper, so target the Unity pid directly.
  [ -n "${UPID:-}" ] && kill -9 "$UPID" 2>/dev/null
  sleep 1
  [ -n "${OWN_XPID:-}" ] && kill -9 "$OWN_XPID" 2>/dev/null
  for p in $(pgrep -x Xvfb 2>/dev/null); do
    if [ "$(ps -o user= -p "$p" 2>/dev/null | tr -d ' ')" = "$(id -un)" ]; then
      # only reap servers with no client left; ours has none now
      kill -9 "$p" 2>/dev/null
    fi
  done
fi
say ""

# --- verdict ---------------------------------------------------------------
say "==============================================="
printf ' %d ok, %d warn, %d fail\n' "$PASS" "$WARN" "$FAIL"
if [ "$FAIL" -gt 0 ]; then
  say " VERDICT: blockers found -- see [FAIL] lines above."
  say "==============================================="
  exit 1
fi
if [ -z "$BUILD" ]; then
  say " VERDICT: no blockers in capability checks."
  say " Re-run with the build path for the conclusive answer."
else
  say " VERDICT: this node can host headless ratsim."
fi
say "==============================================="
exit 0
