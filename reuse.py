import json, os, subprocess, gc, warnings
import numpy as np
warnings.filterwarnings("ignore")
import aneforge as af
def phys():
    o=subprocess.check_output(["footprint","-p",str(os.getpid())],text=True,stderr=subprocess.DEVNULL)
    for l in o.splitlines():
        if "phys_footprint:" in l:
            v=l.split(":")[1].strip(); n=float(v.split()[0]); u=v.split()[1]
            return n*{"KB":1/1024,"MB":1,"GB":1024}[u]
    return -1
N=K=4096; PER=8
rng=np.random.default_rng(0)
print(f"{'round':>6} {'baked_MB':>9} {'phys_MB':>9} {'delta':>8}  mode")
prev=phys()
progs=[]
# A: keep every program resident
for r in range(3):
    Ws=[(rng.standard_normal((N,K))/64).astype(np.float16) for _ in range(PER)]
    x=af.input([64,K]); y=x
    for W in Ws: y=y.linear(W)
    p=af.compile(y); progs.append(p); del Ws; gc.collect()
    p(rng.standard_normal((64,K)).astype(np.float16))
    cur=phys(); print(f"{r:6} {PER*N*K*2/1e6:9.1f} {cur:9.1f} {cur-prev:8.1f}  keep resident",flush=True); prev=cur
for p in progs: p.release()
progs=[]; gc.collect()
cur=phys(); print(f"{'rel':>6} {'':9} {cur:9.1f} {cur-prev:8.1f}  after releasing all",flush=True); prev=cur
# B: release each before compiling the next
for r in range(3):
    Ws=[(rng.standard_normal((N,K))/64).astype(np.float16) for _ in range(PER)]
    x=af.input([64,K]); y=x
    for W in Ws: y=y.linear(W)
    p=af.compile(y); del Ws; gc.collect()
    p(rng.standard_normal((64,K)).astype(np.float16))
    p.release(); del p; gc.collect()
    cur=phys(); print(f"{r:6} {PER*N*K*2/1e6:9.1f} {cur:9.1f} {cur-prev:8.1f}  release each",flush=True); prev=cur
