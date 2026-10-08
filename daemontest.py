import json, os, subprocess, gc, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
import aneforge as af
import ctypes, ctypes.util

def daemon_rss():
    out={}
    try:
        pids=subprocess.check_output(["pgrep","-f","aned"],text=True).split()
    except Exception:
        pids=[]
    for p in pids:
        try:
            r=subprocess.check_output(["ps","-o","rss=","-p",p],text=True).strip()
            out[p]=int(r)/1024
        except Exception: pass
    return out

def self_phys():
    try:
        o=subprocess.check_output(["footprint","-p",str(os.getpid())],text=True,stderr=subprocess.DEVNULL)
        for l in o.splitlines():
            if "phys_footprint:" in l: return l.split(":")[1].strip()
    except Exception: pass
    return "?"

libc=ctypes.CDLL(ctypes.util.find_library("c"))
def relieve():
    try:
        libc.malloc_zone_pressure_relief(ctypes.c_void_p(0), ctypes.c_size_t(0))
        return True
    except Exception: return False

print(json.dumps({"stage":"start","self":self_phys(),"aned_mb":daemon_rss()}),flush=True)
N=K=4096; nl=12
rng=np.random.default_rng(0)
Ws=[(rng.standard_normal((N,K))/64).astype(np.float16) for _ in range(nl)]
x=af.input([64,K]); y=x
for W in Ws: y=y.linear(W)
prog=af.compile(y)
print(json.dumps({"stage":"compiled","self":self_phys(),"aned_mb":daemon_rss()}),flush=True)
del Ws; gc.collect()
print(json.dumps({"stage":"host_freed","self":self_phys(),"aned_mb":daemon_rss()}),flush=True)
ok=relieve(); gc.collect()
print(json.dumps({"stage":"after_pressure_relief","relieved":ok,"self":self_phys(),"aned_mb":daemon_rss()}),flush=True)
prog(rng.standard_normal((64,K)).astype(np.float16))
print(json.dumps({"stage":"after_run","self":self_phys(),"aned_mb":daemon_rss()}),flush=True)
relieve()
print(json.dumps({"stage":"final","self":self_phys(),"aned_mb":daemon_rss()}),flush=True)
