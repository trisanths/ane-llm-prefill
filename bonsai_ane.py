"""Per-layer ANE cost for Ternary Bonsai 8B, and the 8B budget it implies.

ANEForge has no 2-bit path and no API to accept packed codes, so Bonsai's
weights must be dequantized to fp16 on the host and re-quantized by ANEForge.
Doing that per layer bounds the transient memory to one layer.
"""
import argparse
import json
import os
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
import mlx.core as mx  # noqa: E402
import aneforge as af  # noqa: E402
import diskguard  # noqa: E402
import memguard  # noqa: E402

CACHE = os.path.expanduser("~/Models/.aneforge-cache")


def cache_bytes():
    t = 0
    for r, _, f in os.walk(CACHE):
        for n in f:
            try:
                t += os.path.getsize(os.path.join(r, n))
            except OSError:
                pass
    return t


def deq(mod, gs, bits):
    """Dequantize an mlx_lm QuantizedLinear to a fp16 numpy [out, in]."""
    w = mx.dequantize(mod.weight, mod.scales, mod.biases, group_size=gs, bits=bits)
    mx.eval(w)
    return np.array(w).astype(np.float16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="prism-ml/Ternary-Bonsai-8B-mlx-2bit")
    ap.add_argument("--M", type=int, default=256)
    ap.add_argument("--kt", type=int, default=2048)
    ap.add_argument("--compress", nargs="+", default=["blockwise", "int4", "int8"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    diskguard.guard(limit_gb=5.0, min_free_gb=10.0)
    from mlx_lm import load
    model, _ = load(a.model)
    cfg = model.args
    q = getattr(cfg, "quantization", None)
    gs, bits = (q["group_size"], q["bits"]) if isinstance(q, dict) else (128, 2)
    L0 = model.model.layers[0]
    parts = {"wq": L0.self_attn.q_proj, "wk": L0.self_attn.k_proj,
             "wv": L0.self_attn.v_proj, "wo": L0.self_attn.o_proj,
             "wgate": L0.mlp.gate_proj, "wup": L0.mlp.up_proj, "wdown": L0.mlp.down_proj}
    W = {}
    total_params = 0
    for k, mod in parts.items():
        W[k] = deq(mod, gs, bits)
        total_params += W[k].size
    print(json.dumps({"layer_params_M": round(total_params / 1e6, 1),
                      "fp16_layer_MB": round(total_params * 2 / 1e6, 1),
                      "group_size": gs, "bits": bits,
                      "mem": memguard.stamp()}), flush=True)

    d = cfg.hidden_size
    for comp in a.compress:
        before = cache_bytes()
        x = af.input([a.M, d])
        h = x
        for k in ("wq", "wo"):
            h = h.linear(W[k]) if k == "wq" else h.linear(W[k])
        # a representative slice: q and o projections only, to keep it small
        try:
            t0 = time.perf_counter()
            prog = af.compile(h, **({} if comp == "fp16" else {"compress": comp}))
            build = time.perf_counter() - t0
            grew = cache_bytes() - before
            n = W["wq"].size + W["wo"].size
            rec = {"bench": "bonsai_ane_layer", "compress": comp,
                   "probe_params_M": round(n / 1e6, 1),
                   "probe_bytes_MB": round(grew / 1e6, 1),
                   "bytes_per_weight": round(grew / n, 3),
                   "build_s": round(build, 1)}
            memguard.stamp(rec)
            diskguard.stamp(rec)
            print(json.dumps(rec), flush=True)
            if a.out:
                with open(a.out, "a") as f:
                    f.write(json.dumps(rec) + "\n")
            prog.release()
        except Exception as e:  # noqa: BLE001
            print(json.dumps({"compress": comp, "error": f"{type(e).__name__}: {e}"[:200]}), flush=True)


if __name__ == "__main__":
    main()
