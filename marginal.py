import json, os, subprocess, gc, sys, warnings
import numpy as np
warnings.filterwarnings("ignore")
import aneforge as af
def phys_mb():
    o=subprocess.check_output(["footprint","-p",str(os.getpid())],text=True,stderr=subprocess.DEVNULL)
    for l in o.splitlines():
        if "phys_footprint:" in l:
            v=l.split(":")[1].strip()
            n=float(v.split()[0]); u=v.split()[1]
            return n*{"KB":1/1024,"MB":1,"GB":1024}[u]
    return -1
nl=int(sys.argv[1]); stream=len(sys.argv)>2 and sys.argv[2]=="stream"
N=K=4096
base=phys_mb()
rng=np.random.default_rng(0)
if stream:
    # build layer by layer, freeing each host array right after it is baked in
    x=af.input([64,K]); y=x
    for i in range(nl):
        W=(rng.standard_normal((N,K))/64).astype(np.float16)
        y=y.linear(W)
        del W
    peak_host=phys_mb()
    prog=af.compile(y)
else:
    Ws=[(rng.standard_normal((N,K))/64).astype(np.float16) for _ in range(nl)]
    x=af.input([64,K]); y=x
    for W in Ws: y=y.linear(W)
    peak_host=phys_mb()
    prog=af.compile(y)
    del Ws
gc.collect()
after=phys_mb()
prog(rng.standard_normal((64,K)).astype(np.float16))
run=phys_mb()
baked=nl*N*K*2/1e6
print(json.dumps({"layers":nl,"stream":stream,"baked_MB":round(baked,1),
                  "base_MB":round(base,1),"peak_host_MB":round(peak_host,1),
                  "after_compile_MB":round(after,1),"after_run_MB":round(run,1),
                  "overhead_ratio":round((run-base)/baked,2)}),flush=True)
