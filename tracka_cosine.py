"""Track A: Bonsai 8B per-layer cosine, int4 (single codebook) vs blockwise int8,
along the residual stream. Drift compounding is the thing to watch.

For each layer we dequantize Bonsai's real 2-bit weights to fp16 (the reference),
then compile the layer's projections under int4 and under blockwise, and measure
cosine of each projection's output vs the fp16 reference. We track the running
product down the stack as a proxy for residual drift.
"""
import json, warnings, sys
import numpy as np, mlx.core as mx, aneforge as af
warnings.filterwarnings("ignore")
sys.argv=["x"]
import diskguard; diskguard.guard(limit_gb=4.0, min_free_gb=8.0)
from mlx_lm import load
model,_ = load("prism-ml/Ternary-Bonsai-8B-mlx-2bit")
NL = int(sys.argv[1]) if len(sys.argv)>1 else 36
gs,bits=128,2
M=64
def deq(mod): 
    w=mx.dequantize(mod.weight,mod.scales,mod.biases,group_size=gs,bits=bits); mx.eval(w)
    return np.array(w).astype(np.float16)
def cos(a,b):
    a,b=a.ravel().astype(np.float64),b.ravel().astype(np.float64)
    return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)))

rng=np.random.default_rng(0)
res={"int4":[], "blockwise":[]}
import gc
for li in range(NL):
    L=model.model.layers[li]
    # use the down_proj as the representative wide reduction (worst case for int4)
    W=deq(L.mlp.down_proj)
    x=rng.standard_normal((M,W.shape[1])).astype(np.float16)
    ref=x.astype(np.float32)@W.astype(np.float32).T
    xin=af.input([M,W.shape[1]])
    for comp in ("int4","blockwise"):
        p=af.compile(xin.linear(W), compress=comp)
        got=np.asarray(p(x),dtype=np.float32)
        res[comp].append(cos(got,ref)); p.release()
    del W; gc.collect()
    if li%6==0 or li==NL-1:
        print(json.dumps({"layer":li,"int4_cos":round(res['int4'][-1],6),
                          "blockwise_cos":round(res['blockwise'][-1],6)}),flush=True)
summ={"bench":"bonsai_cosine","layers":NL,
      "int4_min":round(min(res['int4']),6),"int4_mean":round(float(np.mean(res['int4'])),6),
      "blockwise_min":round(min(res['blockwise']),6),"blockwise_mean":round(float(np.mean(res['blockwise'])),6),
      "int4_product":round(float(np.prod(res['int4'])),6),
      "blockwise_product":round(float(np.prod(res['blockwise'])),6)}
print(json.dumps(summ),flush=True)
open("results/tracka.jsonl","a").write(json.dumps(summ)+"\n")
open("results/tracka_percos.json","w").write(json.dumps(res))
