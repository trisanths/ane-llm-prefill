#!/bin/sh
# Sustained 60 s GPU load under powermetrics.
#
# The 'both' capture caught only 5 GPU samples above 3 W across 285 s, with an
# obvious dropout mid-burst, so a short benchmark cannot be trusted on this rail.
# This drives the GPU continuously for 60 s: even a lossy sampler accumulates
# enough valid samples to give a median.
#
# powermetrics self-terminates via -n, so no sudo kill is needed (the previous
# script leaked three root processes because `sudo kill` cannot prompt).
set -e
cd "$(dirname "$0")"
LOG=results/power_gpu60.log
echo "powermetrics for 75 s -> $LOG (password once)"
sudo powermetrics --samplers ane_power,gpu_power,cpu_power -i 300 -n 250 > "$LOG" &
PM=$!
sleep 3
echo "driving the GPU for 60 s..."
.venv/bin/python sustain_gpu.py --seconds 60
wait $PM
echo
.venv/bin/python analyze_power.py "$LOG"
