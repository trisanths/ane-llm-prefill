"""Is int4 residency real or deferred? Are the weights file-backed?"""
import json, os, re, subprocess, gc, warnings
import numpy as np
warnings.filterwarnings("ignore")
import aneforge as af
PID=os.getpid()
CACHE=os.path.expanduser("~/Models/.aneforge-cache")

def footprint():
    o=subprocess.check_output(["footprint","-p",str(PID)],text=True,stderr=subprocess.DEVNULL)
    d={}
    for l in o.splitlines():
        if "phys_footprint:" in l: d["phys"]=l.split(":")[1].strip()
        if "neural" in l.lower() and ":" in l:
            parts=l.split(":",1)
            if len(parts)==2 and parts[1].strip(): d[parts[0].strip()]=parts[1].strip()
    return d

def rss(): return int(subprocess.check_output(["ps","-o","rss=","-p",str(PID)],text=True).strip())/1024

def mapped_regions():
    """Every mapped-file region, with path, resident and dirty."""
    out=subprocess.check_output(["vmmap","-w",str(PID)],text=True,stderr=subprocess.DEVNULL)
    rows=[]
    for line in out.splitlines():
        if "aneforge-cache" in line or "weights.bin" in line or "e5bundlecache" in line or "Models" in line:
            rows.append(line.rstrip()[:190])
    return rows

def region_totals():
    out=subprocess.check_output(["vmmap","-summary",str(PID)],text=True,stderr=subprocess.DEVNULL)
    keep=[]
    for line in out.splitlines():
        s=line.strip()
        if s.startswith(("mapped file","Malloc Large","IOSurface","Physical footprint","__DATA","VM_ALLOCATE","ANE","neural")):
            keep.append(s[:150])
    return keep

N=K=4096
rng=np.random.default_rng(0)
codes=rng.integers(-1,2,size=(N,K)).astype(np.float32)
sc=(rng.random((N,K//128))*0.05+0.005).astype(np.float32)
W=(codes.reshape(N,K//128,128)*sc[:,:,None]).reshape(N,K).astype(np.float16)
del codes, sc; gc.collect()
print(json.dumps({"stage":"weights_on_host","rss_MB":round(rss(),1),**footprint()}),flush=True)

x=af.input([64,K])
prog=af.compile(x.linear(W), compress="int4")
del W; gc.collect()
print(json.dumps({"stage":"after_compile","rss_MB":round(rss(),1),**footprint()}),flush=True)

xi=rng.standard_normal((64,K)).astype(np.float16)
prog(xi)
print(json.dumps({"stage":"after_run_1","rss_MB":round(rss(),1),**footprint()}),flush=True)
for _ in range(19):
    prog(xi)
print(json.dumps({"stage":"after_run_20","rss_MB":round(rss(),1),**footprint()}),flush=True)

print("=== mapped regions referencing the compile cache ===",flush=True)
mr=mapped_regions()
if mr:
    for r in mr: print("   "+r,flush=True)
else:
    print("   NONE — no mapped-file region points into ~/Models/.aneforge-cache",flush=True)
print("=== region summary ===",flush=True)
for r in region_totals(): print("   "+r,flush=True)
open("/tmp/vmmap_int4.txt","w").write(subprocess.check_output(["vmmap","-w",str(PID)],text=True,stderr=subprocess.DEVNULL))
