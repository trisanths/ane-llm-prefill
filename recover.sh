#!/bin/sh
# Submit-and-recover for ANE experiments.
#
#   ./recover.sh submit <python-file> [args...]   run a test in isolation with a
#                                               timeout, then verify the engine
#   ./recover.sh probe                            just check the engine is alive
#   ./recover.sh recover                          escalate until the probe passes
#
# Escalation order, least to most disruptive. Steps 3 and 4 need sudo and are
# printed rather than run, so a human decides.
#   1. kill the offending child process
#   2. wait for aned to settle, re-probe
#   3. sudo launchctl kickstart -k system/com.apple.aned      (restart the daemon)
#   4. reboot
#
# Written for Experiment 0. Nothing in that experiment hung the engine: NaN and
# Inf weights, corrupted manifests and deleted bundle files all either ran or
# were rejected cleanly. The escalation path is documented from the launchd
# layout, not from a recovery that was actually needed.

set -u
cd "$(dirname "$0")"
PY=.venv/bin/python
TIMEOUT=${ANE_TIMEOUT:-120}

probe() {
  # Known-good tiny program in a fresh process. Exit 0 only if it runs and
  # returns finite, correct output.
  $PY - <<'EOF' 2>/dev/null
import warnings, numpy as np, sys
warnings.filterwarnings("ignore")
import aneforge as af
W = (np.random.default_rng(0).standard_normal((256, 256)) / 16).astype(np.float16)
x = af.input([8, 256]); p = af.compile(x.linear(W))
xi = np.random.default_rng(1).standard_normal((8, 256)).astype(np.float16)
out = np.asarray(p(xi), dtype=np.float32); p.release()
ref = xi.astype(np.float32) @ W.astype(np.float32).T
ok = np.isfinite(out).all() and np.abs(out - ref).max() / np.abs(ref).max() < 1e-2
print("PROBE_OK" if ok else "PROBE_BAD")
sys.exit(0 if ok else 2)
EOF
}

run_with_timeout() {
  # portable timeout: run in background, kill on deadline
  "$@" &
  pid=$!
  ( sleep "$TIMEOUT"; kill -0 "$pid" 2>/dev/null && { echo "TIMEOUT after ${TIMEOUT}s, killing $pid"; kill -9 "$pid"; } ) &
  watcher=$!
  wait "$pid"; rc=$?
  kill "$watcher" 2>/dev/null; wait "$watcher" 2>/dev/null
  return $rc
}

daemon_state() {
  echo "aned: $(pgrep -x aned | head -1 || echo none)  ANECompilerService: $(pgrep -f ANECompilerService | head -1 || echo none)"
}

recover() {
  echo "[recover] step 1: killing stray experiment children"
  pkill -f "exp0_|aneforge_roofline|block2b|hybrid" 2>/dev/null || true
  sleep 2
  echo "[recover] step 2: daemon state, then probe"
  daemon_state
  if probe | grep -q PROBE_OK; then echo "[recover] engine healthy"; return 0; fi
  echo "[recover] probe failed. Steps 3 and 4 need sudo; run by hand:"
  echo "    sudo launchctl kickstart -k system/com.apple.aned"
  echo "    then: ./recover.sh probe"
  echo "    if still failing: reboot"
  return 1
}

case "${1:-}" in
  probe)   probe ;;
  recover) recover ;;
  submit)
    shift
    echo "[submit] $* (timeout ${TIMEOUT}s)"
    run_with_timeout $PY "$@"; rc=$?
    echo "[submit] child exit $rc"
    if probe | grep -q PROBE_OK; then echo "[submit] engine healthy after test"; exit $rc; fi
    echo "[submit] engine probe FAILED after test; escalating"
    recover; exit 3 ;;
  *) echo "usage: $0 submit <file.py> [args] | probe | recover"; exit 64 ;;
esac
