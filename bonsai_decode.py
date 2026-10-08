"""Item 2: MLX 2-bit decode throughput and RSS for Ternary Bonsai 8B."""
import argparse
import json
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
import mlx.core as mx  # noqa: E402
import memguard  # noqa: E402
import diskguard  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="prism-ml/Ternary-Bonsai-8B-mlx-2bit")
ap.add_argument("--prompt-len", type=int, default=512)
ap.add_argument("--gen", type=int, default=64)
ap.add_argument("--repeats", type=int, default=3)
ap.add_argument("--out", default=None)
a = ap.parse_args()

stages = {"start": memguard.stamp()}
from mlx_lm import load  # noqa: E402

t0 = time.perf_counter()
model, tok = load(a.model)
load_s = time.perf_counter() - t0
stages["after_load"] = memguard.stamp()

nb = 0
for _, p in model.parameters().items() if hasattr(model, "parameters") else []:
    pass


def tree_bytes(t):
    tot = 0
    if isinstance(t, dict):
        for v in t.values():
            tot += tree_bytes(v)
    elif isinstance(t, (list, tuple)):
        for v in t:
            tot += tree_bytes(v)
    elif hasattr(t, "nbytes"):
        tot += t.nbytes
    return tot


weight_gb = tree_bytes(model.parameters()) / 1e9

rng = np.random.default_rng(0)
ids = mx.array(rng.integers(1000, 140000, size=a.prompt_len).tolist())

# prefill (MLX-only baseline for item 3 later)
def mlx_prefill():
    from mlx_lm.models.cache import make_prompt_cache
    cache = make_prompt_cache(model)
    logits = model(ids[None], cache=cache)
    mx.eval(logits)
    return logits, cache


mlx_prefill()
pt = []
for _ in range(a.repeats):
    t0 = time.perf_counter()
    mlx_prefill()
    pt.append(time.perf_counter() - t0)
prefill_s = float(np.median(pt))
stages["after_prefill"] = memguard.stamp()

logits, cache = mlx_prefill()
first = int(mx.argmax(logits[0, -1]).item())


def decode(n):
    tokn = mx.array([[first]])
    out = []
    for _ in range(n):
        lg = model(tokn, cache=cache)
        mx.eval(lg)
        t = int(mx.argmax(lg[0, -1]).item())
        out.append(t)
        tokn = mx.array([[t]])
    return out


decode(8)
dt = []
for _ in range(a.repeats):
    _, cache = mlx_prefill()
    t0 = time.perf_counter()
    decode(a.gen)
    dt.append(time.perf_counter() - t0)
decode_s = float(np.median(dt))
stages["after_decode"] = memguard.stamp()

r = {"bench": "bonsai_mlx", "model": a.model, "load_s": round(load_s, 1),
     "weight_gb": round(weight_gb, 2),
     "prompt_len": a.prompt_len, "gen": a.gen,
     "mlx_prefill_s": prefill_s, "mlx_prefill_tok_s": a.prompt_len / prefill_s,
     "decode_s": decode_s, "decode_tok_s": a.gen / decode_s,
     "sample": tok.decode(decode(12)), "stages": stages}
diskguard.stamp(r)
memguard.stamp(r)
print(json.dumps(r), flush=True)
if a.out:
    with open(a.out, "a") as f:
        f.write(json.dumps(r) + "\n")
