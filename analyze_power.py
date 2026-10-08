"""Phase-aware powermetrics summary.

A whole-log median is meaningless when the capture outlives the benchmark, so
this finds contiguous loaded runs per rail and reports those.
"""
import re
import sys

import numpy as np

txt = open(sys.argv[1]).read()


def rail(name):
    return np.array([int(m) for m in re.findall(rf"{name} Power:\s+(\d+)\s*mW", txt)])


def runs(a, thresh, min_len=3):
    """Contiguous index ranges where a > thresh."""
    hot = a > thresh
    out, start = [], None
    for i, v in enumerate(hot):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= min_len:
                out.append((start, i))
            start = None
    if start is not None and len(a) - start >= min_len:
        out.append((start, len(a)))
    return out


ane, gpu, cpu = rail("ANE"), rail("GPU"), rail("CPU")
dt = 0.3
print(f"capture {len(ane) * dt:.0f} s, {len(ane)} samples at {dt * 1000:.0f} ms\n")

for label, a, thresh in (("ANE", ane, 2000), ("GPU", gpu, 5000)):
    rs = runs(a, thresh)
    if not rs:
        print(f"{label}: never exceeded {thresh} mW (peak {a.max()} mW)")
        continue
    s, e = max(rs, key=lambda r: r[1] - r[0])          # the longest loaded burst
    seg = a[s:e]
    print(f"{label} loaded phase: t={s * dt:.1f}-{e * dt:.1f} s, {len(seg)} samples")
    print(f"     median {int(np.median(seg))} mW   mean {seg.mean():.0f} mW   peak {seg.max()} mW")
    idle = a[a <= 200]
    print(f"     idle floor: {int(np.median(idle)) if len(idle) else 'n/a'} mW over {len(idle)} samples")

# efficiency: pass measured TFLOPS via argv[2] (ane) and argv[3] (gpu); defaults are this project's
for label, w, tops in (("ANE  fp16 tiled", None, 12.15779679713063),
                       ("GPU  fp16 MLX  ", None, 4.614545635233252)):
    a = ane if "ANE" in label else gpu
    th = 2000 if "ANE" in label else 5000
    rs = runs(a, th)
    if not rs:
        continue
    s, e = max(rs, key=lambda r: r[1] - r[0])
    med = float(np.median(a[s:e])) / 1000
    print(f"\n{label}: {tops:5.2f} TFLOPS at {med:5.2f} W  ->  {tops / med:.2f} TFLOPS/W")
