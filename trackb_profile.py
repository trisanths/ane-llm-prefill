"""Track B.2: Bonsai 8B decode roofline + per-token profile.

Roofline: 2-bit weights must be streamed once per token. tok/s ceiling =
achievable_bw / weight_bytes. Gap to that ceiling is Python overhead, kernel
launch latency, KV growth, and sampling.
"""
import time, json, sys
import numpy as np
import mlx.core as mx

sys.argv = ["x"]
import diskguard, memguard
from mlx_lm import load

t0 = time.perf_counter()
model, tok = load("prism-ml/Ternary-Bonsai-8B-mlx-2bit")
load_s = time.perf_counter() - t0

def tree_bytes(t):
    if isinstance(t, dict): return sum(tree_bytes(v) for v in t.values())
    if isinstance(t, (list, tuple)): return sum(tree_bytes(v) for v in t)
    return t.nbytes if hasattr(t, "nbytes") else 0
wbytes = tree_bytes(model.parameters())

from mlx_lm.models.cache import make_prompt_cache
prompt = mx.array([[1]*32])
cache = make_prompt_cache(model)
logits = model(prompt, cache=cache); mx.eval(logits)
first = int(mx.argmax(logits[0,-1]).item())

# steady-state decode timing
def decode_n(n):
    tk = mx.array([[first]]); out=[]
    t = time.perf_counter()
    for _ in range(n):
        lg = model(tk, cache=cache); mx.eval(lg)
        nt = int(mx.argmax(lg[0,-1]).item()); out.append(nt); tk = mx.array([[nt]])
    return (time.perf_counter()-t)/n
for _ in range(5): decode_n(1)
per_tok = np.median([decode_n(8) for _ in range(5)])

# component isolation: forward-only (no argmax/python token loop)
def fwd_only(n):
    tk = mx.array([[first]])
    t = time.perf_counter()
    for _ in range(n):
        lg = model(tk, cache=cache); mx.eval(lg)
    return (time.perf_counter()-t)/n
fwd = np.median([fwd_only(16) for _ in range(5)])

# sampling+python overhead = per_tok - fwd
bw_ceiling_tok_s = 159.2e9 / wbytes
r = {"bench":"bonsai_roofline", "load_s":round(load_s,1),
     "weight_GB":round(wbytes/1e9,2),
     "decode_tok_s":round(1/per_tok,2), "per_tok_ms":round(per_tok*1e3,3),
     "fwd_only_tok_s":round(1/fwd,2), "fwd_only_ms":round(fwd*1e3,3),
     "sampling_python_ms":round((per_tok-fwd)*1e3,3),
     "bw_ceiling_tok_s":round(bw_ceiling_tok_s,1),
     "achieved_frac_of_roofline":round((1/per_tok)/bw_ceiling_tok_s,3),
     "implied_bw_GBps":round(wbytes*(1/per_tok)/1e9,1),
     "mem":memguard.stamp()}
print(json.dumps(r), flush=True)
open("results/trackb.jsonl","a").write(json.dumps(r)+"\n")
