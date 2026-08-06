#!/usr/bin/env bash
# Rigorous single-variant probe for Unity headless modes.
#
# Launches the build in its own process group, then verifies that the process
# LISTENING on the port is actually a descendant of what we launched — the
# naive `ss | grep :9000` check gives false positives when a previous test
# leaked a process that still holds the port.
#
# Usage: probe.sh <name> <logdir> -- <command...>
set -u

NAME="$1"; LOGDIR="$2"; shift 3   # shift past name, logdir, and the literal --
PORT=9000
LOG="$LOGDIR/${NAME}.log"
rm -f "$LOG"

# Precondition: port must be free, else the result is meaningless.
if ss -tln | grep -q ":${PORT} "; then
  echo "[$NAME] ABORT: port $PORT already held before launch"
  exit 3
fi

setsid "$@" -port "$PORT" -logFile "$LOG" >/dev/null 2>&1 &
LAUNCHED=$!
PGID=$(ps -o pgid= -p "$LAUNCHED" 2>/dev/null | tr -d ' ')

verdict="TIMEOUT"
elapsed=0
for i in $(seq 1 25); do
  sleep 1; elapsed=$i
  # Is the listener ours? Compare the listening pid's process group to ours.
  lpid=$(ss -tlnp 2>/dev/null | grep ":${PORT} " | grep -oP 'pid=\K[0-9]+' | head -1)
  if [[ -n "${lpid:-}" ]]; then
    lpgid=$(ps -o pgid= -p "$lpid" 2>/dev/null | tr -d ' ')
    if [[ "$lpgid" == "$PGID" ]]; then verdict="SUCCESS"; else verdict="FOREIGN_LISTENER"; fi
    break
  fi
  # Nothing listening yet — is anything from our group still alive?
  if ! pgrep -g "$PGID" >/dev/null 2>&1; then verdict="DIED"; break; fi
done

segv=$(grep -c 'Caught fatal signal' "$LOG" 2>/dev/null || echo 0)
printf "[%-24s] %-16s in %2ss   segv_in_log=%s\n" "$NAME" "$verdict" "$elapsed" "$segv"

# Kill the entire process group — this is what the earlier harness got wrong
# (killing `xvfb-run` left its Unity child alive holding the port).
kill -9 -"$PGID" 2>/dev/null
sleep 2
[[ "$verdict" == "SUCCESS" ]] && exit 0 || exit 1
