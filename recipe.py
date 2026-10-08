"""The memory recipe: 1 layer/program + int4 + every baked tensor tiled to a bounded shape."""
import json, os, subprocess, gc, sys, warnings
import numpy as np
warnings.filterwarnings("ignore")
import aneforge as af

def rss():
    return int(subprocess.check_output(["ps","-o","rss=","-p",str(os.getpid())],text=True).strip())/1024

D, FF, KVD, NL = 4096, 12288, 1024, int(sys.argv[1]) if len(sys.argv)>1 else 4
TILE_N = int(sys.argv[2]) if len(sys.argv)>2 else 1024   # cap on the output rows per baked tensor
COMP = sys.argv[3] if len(sys.argv)>3 else "int4"
rng=np.random.default_rng(0)

def wgt(n,k):
    codes=rng.integers(-1,2,size=(n,k)).astype(np.float32)
    g=min(128,k); sc=(rng.random((n,k//g))*0.05+0.005).astype(np.float32)
    return (codes.reshape(n,k//g,g)*sc[:,:,None]).reshape(n,k).astype(np.float16)

def tiled_linear_rows(x, n, k, M):
    """Split the OUTPUT axis so no baked tensor exceeds TILE_N rows, then concat."""
    parts=[]
    for s in range(0, n, TILE_N):
        r=min(TILE_N, n-s)
        parts.append(x.linear(wgt(r,k)))
    return af.concat(parts, axis=-1) if len(parts)>1 else parts[0]

M=64
base=rss(); keep=[]; prev=base
print(json.dumps({"stage":"base","rss_MB":round(base,1),"tile_rows":TILE_N,"compress":COMP}),flush=True)
for li in range(NL):
    x=af.input([M,D])
    q=tiled_linear_rows(x,D,D,M)
    k=tiled_linear_rows(x,KVD,D,M)
    v=tiled_linear_rows(x,KVD,D,M)
    o=tiled_linear_rows(q,D,D,M)
    h=x+o
    g=tiled_linear_rows(h,FF,D,M).silu()
    u=tiled_linear_rows(h,FF,D,M)
    y=h+tiled_linear_rows(g*u,D,FF,M)
    p=af.compile(y, **({} if COMP=="fp16" else {"compress":COMP}))
    p(rng.standard_normal((M,D)).astype(np.float16))
    keep.append(p); gc.collect()
    cur=rss()
    params=(D*D + 2*KVD*D + D*D + 2*FF*D + D*FF)
    print(json.dumps({"layer":li,"rss_MB":round(cur,1),"delta_MB":round(cur-prev,1),
                      "layer_params_M":round(params/1e6,1),
                      "fp16_equiv_MB":round(params*2/1e6,1)}),flush=True)
    prev=cur
tot=(D*D + 2*KVD*D + D*D + 2*FF*D + D*FF)*NL
print(json.dumps({"summary":True,"layers":NL,"total_params_M":round(tot/1e6,1),
                  "fp16_equiv_GB":round(tot*2/1e9,2),
                  "rss_growth_GB":round((rss()-base)/1024,2),
                  "ratio_vs_fp16":round((rss()-base)*1e6/(tot*2),3)}),flush=True)
