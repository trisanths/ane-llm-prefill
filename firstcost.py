import json, os, subprocess, gc, sys, warnings
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
rng=np.random.default_rng(0)
def build(nl,N=4096,K=4096):
    Ws=[(rng.standard_normal((N,K))/64).astype(np.float16) for _ in range(nl)]
    x=af.input([64,K]); y=x
    for W in Ws: y=y.linear(W)
    p=af.compile(y); del Ws; gc.collect()
    p(rng.standard_normal((64,K)).astype(np.float16))
    return p, nl*N*K*2/1e6
mode=sys.argv[1]
b=phys(); print(json.dumps({"stage":"base","phys":b}),flush=True)
keep=[]
if mode=="tinyfirst":
    W=(rng.standard_normal((64,64))/8).astype(np.float16)
    x=af.input([8,64]); p0=af.compile(x.linear(W)); p0(rng.standard_normal((8,64)).astype(np.float16))
    keep.append(p0); a=phys()
    print(json.dumps({"stage":"after_tiny_compile","phys":a,"delta":round(a-b,1)}),flush=True); b=a
p,baked=build(8)
keep.append(p)
a=phys(); print(json.dumps({"stage":"after_8layer","baked_MB":round(baked,1),"phys":a,"delta":round(a-b,1),"ratio":round((a-b)/baked,2)}),flush=True); b=a
p2,baked2=build(8); keep.append(p2)
a=phys(); print(json.dumps({"stage":"second_8layer","baked_MB":round(baked2,1),"phys":a,"delta":round(a-b,1),"ratio":round((a-b)/baked2,2)}),flush=True)
