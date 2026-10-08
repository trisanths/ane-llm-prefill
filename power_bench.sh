#!/bin/sh
# Capture ANE / GPU / CPU rail power while a benchmark runs, for one engine or
# both back to back under a single password prompt.
#   ./power_bench.sh            both engines, default shapes
#   ./power_bench.sh ane        ANE only
#   ./power_bench.sh gpu        GPU only
set -e
cd "$(dirname "$0")"
WHICH=${1:-both}
LOG=results/power_${WHICH}.log
echo "Starting powermetrics (needs your password once), logging to $LOG"
sudo powermetrics --samplers ane_power,gpu_power,cpu_power -i 300 > "$LOG" &
PM=$!
sleep 2
if [ "$WHICH" = "ane" ] || [ "$WHICH" = "both" ]; then
  echo "--- ANE phase"
  .venv/bin/python chain_tiled.py --M 1024 --K 4096 --N 4096 --layers 8 \
    --kt 1024 --precision fp16 --iters 150 --warmup 5 \
    --out results/power_bench_ane.jsonl 2>/dev/null | tail -1
fi
if [ "$WHICH" = "gpu" ] || [ "$WHICH" = "both" ]; then
  echo "--- idle gap"
  /bin/sleep 3
  echo "--- GPU phase"
  .venv/bin/python mlx_bench.py --M 1024 --K 4096 --N 4096 --layers 8 \
    --iters 150 --warmup 5 --out results/power_bench_gpu.jsonl 2>/dev/null | tail -1
fi
/bin/sleep 1
sudo kill $PM 2>/dev/null || true
wait $PM 2>/dev/null || true
echo
.venv/bin/python analyze_power.py "$LOG"
