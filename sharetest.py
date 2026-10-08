"""Is the ANE copy made at compile or at load, and is MLX's copy file-backed?"""
import json, os, subprocess, sys, gc, warnings
import numpy as np
warnings.filterwarnings("ignore")
import aneforge as af

PID = os.getpid()


def regions():
    """Dirty anonymous vs clean file-backed, from vmmap -summary."""
    out = subprocess.check_output(["vmmap", "-summary", str(PID)], text=True,
                                  stderr=subprocess.DEVNULL)
    res = {}
    for line in out.splitlines():
        for key in ("Malloc Large", "mapped file", "IOSurface", "Physical footprint:"):
            if line.strip().startswith(key):
                res[key] = line.rstrip()
    return res


def phys():
    out = subprocess.check_output(["footprint", "-p", str(PID)], text=True,
                                  stderr=subprocess.DEVNULL)
    d = {}
    for line in out.splitlines():
        if "phys_footprint:" in line:
            d["phys"] = line.split(":")[1].strip()
        if "neural" in line and "peak" in line:
            d["neural_peak"] = line.split(":")[1].strip()
    return d


mode = sys.argv[1] if len(sys.argv) > 1 else "compile"
N = K = 4096
nlayers = 12
rng = np.random.default_rng(0)
Ws = [(rng.standard_normal((N, K)) / 64).astype(np.float16) for _ in range(nlayers)]
print(json.dumps({"stage": "weights_on_host", **phys()}), flush=True)

x = af.input([64, K])
y = x
for W in Ws:
    y = y.linear(W)
prog = af.compile(y)
del Ws
gc.collect()
print(json.dumps({"stage": f"{mode}_done", **phys()}), flush=True)
for k, v in regions().items():
    print("   " + v, flush=True)
xi = rng.standard_normal((64, K)).astype(np.float16)
prog(xi)
print(json.dumps({"stage": "after_run", **phys()}), flush=True)
prog.release()
gc.collect()
print(json.dumps({"stage": "after_release", **phys()}), flush=True)
for k, v in regions().items():
    print("   " + v, flush=True)
