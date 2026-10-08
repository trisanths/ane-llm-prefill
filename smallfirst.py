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
N=K=4096
per_layer_mb=N*K*2/1e6
nprog=int(sys.argv[1]) if len(sys.argv)>1 else 24
layers_each=int(sys.argv[2]) if len(sys.argv)>2 else 1
keep=[]; b0=phys()
print(f"base {b0:.0f} MB, {nprog} programs x {layers_each} layer(s) = {nprog*layers_each*per_layer_mb:.0f} MB of weights")
prev=b0
for i in range(nprog):
    Ws=[(rng.standard_normal((N,K))/64).astype(np.float16) for _ in range(layers_each)]
    x=af.input([64,K]); y=x
    for W in Ws: y=y.linear(W)
    p=af.compile(y); del Ws; gc.collect()
    p(rng.standard_normal((64,K)).astype(np.float16))
    keep.append(p)
    if i in (0,1,2,nprog//2,nprog-1):
        cur=phys()
        print(json.dumps({"prog":i,"phys_MB":round(cur,1),"delta":round(cur-prev,1),
                          "weights_so_far_MB":round((i+1)*layers_each*per_layer_mb,1),
                          "ratio_total":round((cur-b0)/((i+1)*layers_each*per_layer_mb),2)}),flush=True)
        prev=cur
    else:
        prev=phys()
