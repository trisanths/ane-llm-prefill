#!/bin/sh
# Sustained 60 s ANE load under powermetrics, the symmetric counterpart to
# power_gpu.sh. The ANE figure currently rests on 12 samples; this gives it the
# same footing as the GPU's 379.
set -e
cd "$(dirname "$0")"
LOG=results/power_ane60.log
echo "powermetrics for 75 s -> $LOG (password once)"
sudo powermetrics --samplers ane_power,gpu_power,cpu_power -i 300 -n 250 > "$LOG" &
PM=$!
sleep 3
echo "driving the ANE for 60 s..."
.venv/bin/python sustain_ane.py --seconds 60
wait $PM
echo
.venv/bin/python analyze_power.py "$LOG"
