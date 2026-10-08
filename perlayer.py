"""Per-layer cosine along the residual stream, and the W8A16-vs-fp16 comparison.

A stack of L transformer blocks is compiled as ONE program whose output is every
layer's residual concatenated along rows, so drift is measured at each depth
without paying a dispatch per layer. The fp32 numpy reference chains the same
math with the same weights.
"""
import argparse
import json
import time
import warnings

import numpy as np
import aneforge as af

warnings.filterwarnings("ignore", category=af.DispatchFloorWarning)
warnings.filterwarnings("ignore", category=af.PrecisionWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from block2b import CFG, expand_kv, reference, swiglu_mlp, tiled_linear  # noqa: E402


def make_weights(layer, cfg=CFG):
    d, H, H_kv, dh, ff = cfg["d"], cfg["n_heads"], cfg["n_kv"], cfg["head_dim"], cfg["ff"]
    rng = np.random.default_rng(1000 + layer)

    def w(n, k):
        return (rng.standard_normal((n, k)) / np.sqrt(k)).astype(np.float16)

    return {"g1": np.ones(d, np.float16), "g2": np.ones(d, np.float16),
            "q": w(H * dh, d), "k": w(H_kv * dh, d), "v": w(H_kv * dh, d),
            "o": w(d, H * dh), "gate": w(ff, d), "up": w(ff, d), "down": w(d, ff)}


def one_block(x, W, M, kt, ff_chunk, cos, sin, mask, cfg=CFG):
    d, H, H_kv, dh, eps = cfg["d"], cfg["n_heads"], cfg["n_kv"], cfg["head_dim"], cfg["eps"]
    n_rep = H // H_kv
    h = x.rms_norm(W["g1"], eps=eps)
    q = tiled_linear(h, W["q"], M, kt).reshape([M, H, dh]).transpose([1, 0, 2])
    k = tiled_linear(h, W["k"], M, kt).reshape([M, H_kv, dh]).transpose([1, 0, 2])
    v = tiled_linear(h, W["v"], M, kt).reshape([M, H_kv, dh]).transpose([1, 0, 2])
    q, k = af.rope(q, cos, sin), af.rope(k, cos, sin)
    k, v = expand_kv(k, n_rep, H_kv, M, dh), expand_kv(v, n_rep, H_kv, M, dh)
    s = af.einsum("hid,hjd->hij", q, k) * np.float16(1.0 / np.sqrt(dh))
    ctx = af.einsum("hij,hjd->hid", (s + mask).softmax(axis=-1), v)
    ctx = ctx.transpose([1, 0, 2]).reshape([M, H * dh])
    x = x + tiled_linear(ctx, W["o"], M, kt)
    h2 = x.rms_norm(W["g2"], eps=eps)
    return x + swiglu_mlp(h2, W["gate"], W["up"], W["down"], M, kt, ff_chunk)


def build_stack(M, kt, layers, ff_chunk, cfg=CFG):
    """Weights only; each block is compiled as its own program (see build_one)."""
    return None, [make_weights(i) for i in range(layers)], *af.rope_tables(M, cfg["head_dim"])


def build_one(M, kt, W, ff_chunk, cos, sin, cfg=CFG):
    """One program per block, which is the granularity the engine is dispatched at.

    Compiling the whole stack as a single program with every layer's residual
    concatenated was the first attempt; the ANE compiler rejects it (mask=0x4).
    Per-block programs also give the residual at each depth for free.
    """
    mask = np.triu(np.full((M, M), -np.inf, np.float32), k=1).astype(np.float16)
    x = af.input([M, cfg["d"]])
    return one_block(x, W, M, kt, ff_chunk, cos, sin, mask)


def ref_stack(xi, Ws, cos, sin):
    out, h = [], xi
    for W in Ws:
        h = reference(h.astype(np.float16), W, cos, sin)
        out.append(h.copy())
    return out


def cos_sim(a, b):
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def run(M, kt, layers, ff_chunk, precision, iters, warmup, Ws, cos, sin):
    """Run the stack one block-program at a time, capturing the residual at each depth."""
    xi = np.random.default_rng(1).standard_normal((M, CFG["d"])).astype(np.float16)
    kw = {} if precision == "fp16" else {"compress": precision}
    outs, per_block_ms, build_total, ops = [], [], 0.0, 0
    h = xi
    for i in range(layers):
        t0 = time.perf_counter()
        prog = af.compile(build_one(M, kt, Ws[i], ff_chunk, cos, sin), **kw)
        build_total += time.perf_counter() - t0
        ops = prog.n_ops
        for _ in range(warmup):
            prog(h)
        ts = []
        for _ in range(iters):
            t = time.perf_counter()
            prog(h)
            ts.append(time.perf_counter() - t)
        per_block_ms.append(float(np.median(ts)) * 1e3)
        h = np.asarray(prog(h), dtype=np.float16)
        outs.append(h.astype(np.float32))
        prog.release()
    return np.stack(outs), {"n_ops_per_block": ops, "build_s_total": round(build_total, 1),
                            "t_med_ms_per_block": float(np.median(per_block_ms)),
                            "iters": iters, "M": M, "kt": kt, "layers": layers,
                            "precision": precision}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=512)
    ap.add_argument("--kt", type=int, nargs="+", default=[1024, 2048])
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--ff-chunk", type=int, default=2176)
    ap.add_argument("--precision", nargs="+", default=["fp16", "int8"])
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    xi = np.random.default_rng(1).standard_normal((a.M, CFG["d"])).astype(np.float16)
    _, Ws, cos, sin = build_stack(a.M, a.kt[0], a.layers, a.ff_chunk)
    ref = ref_stack(xi.astype(np.float32), Ws, cos, sin)

    store = {}
    for kt in a.kt:
        for p in a.precision:
            try:
                out, meta = run(a.M, kt, a.layers, a.ff_chunk, p, a.iters, a.warmup, Ws, cos, sin)
            except Exception as e:  # noqa: BLE001
                print(json.dumps({"bench": "perlayer", "M": a.M, "kt": kt, "precision": p,
                                  "error": f"{type(e).__name__}: {e}"[:300]}), flush=True)
                continue
            store[(kt, p)] = out
            rec = dict(meta)
            rec["bench"] = "perlayer"
            rec["cos_vs_fp32"] = [cos_sim(out[i], ref[i]) for i in range(a.layers)]
            rec["relerr_vs_fp32"] = [float(np.abs(out[i] - ref[i]).max() / np.abs(ref[i]).max())
                                     for i in range(a.layers)]
            base = store.get((kt, "fp16"))
            if p != "fp16" and base is not None:
                rec["cos_vs_fp16"] = [cos_sim(out[i], base[i]) for i in range(a.layers)]
            print(json.dumps(rec), flush=True)
            if a.out:
                with open(a.out, "a") as f:
                    f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
