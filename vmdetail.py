import json, os, subprocess, gc, warnings, re
import numpy as np
warnings.filterwarnings("ignore")
import aneforge as af
PID=os.getpid()
def big_regions(minmb=50):
    out=subprocess.check_output(["vmmap","-w",str(PID)],text=True,stderr=subprocess.DEVNULL)
    rows=[]
    for line in out.splitlines():
        m=re.search(r"^(\S[\S ]{0,30}?)\s+([0-9a-f]+)-([0-9a-f]+)\s+\[\s*([\d.]+)([KMG])\s+([\d.]+)([KMG])\s+([\d.]+)([KMG])\s+([\d.]+)([KMG])\]", line)
        if not m: continue
        def tomb(v,u): return float(v)*{"K":1/1024,"M":1,"G":1024}[u]
        size=tomb(m.group(4),m.group(5)); res=tomb(m.group(6),m.group(7)); dirty=tomb(m.group(8),m.group(9))
        if res>=minmb or dirty>=minmb:
            rows.append((m.group(1).strip(), round(size,1), round(res,1), round(dirty,1), line.split("]")[-1].strip()[:46]))
    return rows
N=K=4096; nl=12
rng=np.random.default_rng(0)
Ws=[(rng.standard_normal((N,K))/64).astype(np.float16) for _ in range(nl)]
x=af.input([64,K]); y=x
for W in Ws: y=y.linear(W)
prog=af.compile(y); del Ws; gc.collect()
prog(rng.standard_normal((64,K)).astype(np.float16))
print(f"baked weights = {nl*N*K*2/1e6:.0f} MB")
print(f"{'region':32} {'size':>8} {'res':>8} {'dirty':>8}  detail")
for r in big_regions(30):
    print(f"{r[0]:32} {r[1]:8.1f} {r[2]:8.1f} {r[3]:8.1f}  {r[4]}")
