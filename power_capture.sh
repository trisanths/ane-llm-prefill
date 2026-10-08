#!/bin/sh
# Samples ANE, GPU and CPU power every 500 ms while a benchmark runs.
# Needs sudo once. Usage:
#   ./power_capture.sh power_int8.log &  then run the benchmark, then kill %1
# or give a sample count as the second argument to stop on its own.
OUT=${1:-power.log}
N=${2:-0}
if [ "$N" -gt 0 ]; then
  sudo powermetrics --samplers ane_power,gpu_power,cpu_power -i 500 -n "$N" > "$OUT"
else
  sudo powermetrics --samplers ane_power,gpu_power,cpu_power -i 500 > "$OUT"
fi
