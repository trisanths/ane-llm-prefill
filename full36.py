"""Item 3: actual residency for 36 Bonsai-shaped layers, 1 layer/program, tiled, int4."""
import json, os, subprocess, gc, sys, warnings
import numpy as np
warnings.filterwarnings("ignore")
import aneforge as af

D, FF, KVD = 4096, 12288, 1024
NL   = int(sys.argv[1]) if len(sys.argv) > 1 else 36
TILE = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
COMP = sys.argv[3] if len(sys.argv) > 3 else "int4"
M = 64
rng = np.random.default_rng(0)

def rss():
    return int(subprocess.check_output(["ps","-o","rss=","-p",str(os.getpid())],text=True).strip())/1024

def neural_mb():
    try:
        o=subprocess.check_output(["footprint","-p",str(os.getpid())],text=True,stderr=subprocess.DEVNULL)
        for l in o.splitlines():
            if l.strip().startswith("neural:"):
                v=l.split(":")[1].strip(); n=float(v.split()[0]); u=v.split()[1]
                return n*{"KB":1/1024,"MB":1,"GB":1024}[u]
    except Exception: pass
    return -1

def wgt(n,k):
    c=rng.integers(-1,2,size=(n,k)).astype(np.float32)
    g=min(128,k); s=(rng.random((n,k//g))*0.05+0.005).astype(np.float32)
    w=(c.reshape(n,k//g,g)*s[:,:,None]).reshape(n,k).astype(np.float16)
    del c,s
    return w

def lin(x,n,k):
    """Split the output axis so no baked tensor exceeds TILE rows."""
    if n<=TILE:
        w=wgt(n,k); y=x.linear(w); del w; return y
    parts=[]
    for s in range(0,n,TILE):
        r=min(TILE,n-s); w=wgt(r,k); parts.append(x.linear(w)); del w
    return af.concat(parts,axis=-1)

base=rss(); keep=[]
params_per_layer = D*D + 2*KVD*D + D*D + 2*FF*D + D*FF
print(json.dumps({"stage":"base","rss_MB":round(base,1),"layers":NL,"tile":TILE,
                  "compress":COMP,"params_per_layer_M":round(params_per_layer/1e6,1)}),flush=True)
for li in range(NL):
    x=af.input([M,D])
    q=lin(x,D,D); k=lin(x,KVD,D); v=lin(x,KVD,D)
    h=x+lin(q,D,D)
    g=lin(h,FF,D).silu(); u=lin(h,FF,D)
    y=h+lin(g*u,D,FF)
    p=af.compile(y, **({} if COMP=="fp16" else {"compress":COMP}))
    p(rng.standard_normal((M,D)).astype(np.float16))
    keep.append(p); gc.collect()
    if li%6==0 or li==NL-1:
        print(json.dumps({"layer":li,"rss_MB":round(rss(),1),"neural_MB":round(neural_mb(),1),
                          "weights_so_far_GB":round((li+1)*params_per_layer*2/1e9,2)}),flush=True)
tot=NL*params_per_layer
print(json.dumps({"summary":True,"layers":NL,"total_params_B":round(tot/1e9,2),
                  "fp16_equiv_GB":round(tot*2/1e9,2),
                  "final_rss_GB":round(rss()/1024,2),
                  "neural_GB":round(neural_mb()/1024,2),
                  "rss_growth_GB":round((rss()-base)/1024,2)}),flush=True)
