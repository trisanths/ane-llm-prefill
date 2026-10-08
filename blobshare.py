"""Weight-sharing step 1: is a baked ANEForge blob mapped, or copied?

If the blob is a mapped file region and physical footprint does not grow by its
size, the same bytes could in principle back a Metal buffer via bytesNoCopy, and
the hybrid would stop paying for two copies of the weights. If the driver wires
or copies the pages, that plan is dead and the fallback is baked-ANE plus
duplicated attention only.
"""
import json
import os
import subprocess
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
import aneforge as af  # noqa: E402

TARGET_GB = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
PID = os.getpid()


def footprint():
    try:
        out = subprocess.check_output(["footprint", "-p", str(PID)], text=True,
                                      stderr=subprocess.DEVNULL)
        return out.strip().splitlines()[-3:]
    except Exception as e:
        return [f"footprint unavailable: {e}"]


def vmmap_summary():
    try:
        out = subprocess.check_output(["vmmap", "-summary", str(PID)], text=True,
                                      stderr=subprocess.DEVNULL)
        return out
    except Exception as e:
        return f"vmmap unavailable: {e}"


def phys_mb():
    """Resident (phys footprint) in MB from footprint, else RSS."""
    try:
        out = subprocess.check_output(["footprint", "-p", str(PID)], text=True,
                                      stderr=subprocess.DEVNULL)
        for line in out.splitlines():
            if "phys_footprint" in line.lower() or "physical footprint" in line.lower():
                import re
                m = re.search(r"([\d.]+)\s*([KMG])B", line)
                if m:
                    v = float(m.group(1))
                    return v * {"K": 1 / 1024, "M": 1, "G": 1024}[m.group(2)]
    except Exception:
        pass
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


# Build a program whose baked weights are about TARGET_GB
N = K = 4096
per = N * K * 2
nlayers = max(1, int(TARGET_GB * 1e9 // per))
print(json.dumps({"stage": "plan", "layers": nlayers,
                  "baked_gb": round(nlayers * per / 1e9, 2)}), flush=True)

print(json.dumps({"stage": "before_weights", "phys_mb": round(phys_mb(), 1)}), flush=True)
rng = np.random.default_rng(0)
Ws = [(rng.standard_normal((N, K)) / 64).astype(np.float16) for _ in range(nlayers)]
print(json.dumps({"stage": "host_weights_built", "phys_mb": round(phys_mb(), 1),
                  "host_gb": round(sum(w.nbytes for w in Ws) / 1e9, 2)}), flush=True)

x = af.input([64, K])
y = x
for W in Ws:
    y = y.linear(W)
t0 = time.perf_counter()
prog = af.compile(y)
print(json.dumps({"stage": "after_compile", "phys_mb": round(phys_mb(), 1),
                  "build_s": round(time.perf_counter() - t0, 1)}), flush=True)

del Ws
import gc
gc.collect()
print(json.dumps({"stage": "after_free_host", "phys_mb": round(phys_mb(), 1)}), flush=True)

xi = rng.standard_normal((64, K)).astype(np.float16)
prog(xi)
print(json.dumps({"stage": "after_first_run", "phys_mb": round(phys_mb(), 1)}), flush=True)
for _ in range(5):
    prog(xi)
print(json.dumps({"stage": "after_runs", "phys_mb": round(phys_mb(), 1)}), flush=True)

vm = vmmap_summary()
open("/tmp/vmmap_blob.txt", "w").write(vm)
print("=== vmmap summary (regions >= 100 MB or interesting) ===", flush=True)
for line in vm.splitlines():
    low = line.lower()
    if any(k in low for k in ("mapped file", "ane", "iosurface", "wired", "physical footprint",
                              "total", "malloc", "region type", "====")):
        print("   " + line.rstrip(), flush=True)
print("=== footprint tail ===", flush=True)
for l in footprint():
    print("   " + l, flush=True)
prog.release()
