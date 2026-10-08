"""Step 2b: one pre-norm transformer block on the ANE via ANEForge.

Shape family is Qwen3-14B-like: d=5120, 40 query heads, 8 KV heads (GQA),
head_dim=128, SwiGLU MLP with ff=17408.

ANE constraints encoded here, each one measured in step 2a or hit while writing
this file. See NOTES.md for the evidence behind each.

C1. Reduction-axis width dominates throughput. A weight tile's byte size barely
    matters; what matters is how wide the contraction is. kt<=2048 is the
    plateau, kt=4096 costs about 1.8x. Every linear below goes through
    tiled_linear(), which splits K and sums partial products.

C2. The output axis is nearly free. A 1024x4096 tile and a 4096x1024 tile are
    the same bytes and differ by 39 percent in throughput, the wide-output one
    being faster. So split K aggressively and leave N whole.

C3. Neither 5120 nor 17408 is a multiple of 2048, so tiling has to handle a
    ragged last tile. 5120 = 2x2560, 17408 = 8x2176. Both land near the kt
    plateau without a remainder, which is luckier than it looks; tiled_linear
    still handles remainders for shapes that are not so kind.

C4. af.sdpa is documented as a graph cut: using it segments the program instead
    of keeping one dispatch. The decomposed variant stays in a single program.
    Both are measured; the cut costs a dispatch floor of about 150 us.

C5. GQA head expansion must not use repeat_interleave, which is slow on this
    path. K and V are expanded by tiling the head axis instead.

C6. RMSNorm is native here via Tensor.rms_norm, so the ANEMLL concat([x,-x])
    LayerNorm trick is not needed. Kept as a flag to measure the difference.

C7. Family 3 (M1/M2) refuses any tensor dimension above 16384. ff=17408 trips
    this on the SwiGLU intermediate, which would be [M, 17408]. The fix is also
    the fast path: split the MLP along ff into chunks, and for each chunk take
    gate, up, the product, and its down-projection contribution, summing the
    contributions. The wide intermediate is never materialized, and each
    down-projection contracts over the chunk width instead of all 17408, which
    C1 wants anyway. This is the same shape as ane-infer's fused FFN.
"""
import argparse
import json
import time
import warnings

import numpy as np
import aneforge as af

warnings.filterwarnings("ignore", category=af.DispatchFloorWarning)
warnings.filterwarnings("ignore", category=af.PrecisionWarning)

CFG = {"d": 5120, "n_heads": 40, "n_kv": 8, "head_dim": 128, "ff": 17408, "eps": 1e-6}


def split_sizes(K, kt):
    """C3: tile widths covering K, last one ragged if kt does not divide K."""
    out, pos = [], 0
    while pos < K:
        out.append(min(kt, K - pos))
        pos += out[-1]
    return out


def tiled_linear(x, W, M, kt):
    """C1 + C2: split the contraction, keep the output axis whole."""
    N, K = W.shape
    sizes = split_sizes(K, kt)
    if len(sizes) == 1:
        return x.linear(W)
    acc, pos = None, 0
    for s in sizes:
        xi = x.slice_by_size([0, pos], [M, s])
        p = xi.linear(np.ascontiguousarray(W[:, pos:pos + s]))
        acc = p if acc is None else acc + p
        pos += s
    return acc


def swiglu_mlp(h, Wg, Wu, Wd, M, kt, ff_chunk):
    """C7 + C1: chunk along ff so [M, ff] never exists, and each down-projection
    contracts over ff_chunk rather than the full ff."""
    acc, pos = None, 0
    for c in split_sizes(Wg.shape[0], ff_chunk):
        g = tiled_linear(h, np.ascontiguousarray(Wg[pos:pos + c]), M, kt).silu()
        u = tiled_linear(h, np.ascontiguousarray(Wu[pos:pos + c]), M, kt)
        d = tiled_linear(g * u, np.ascontiguousarray(Wd[:, pos:pos + c]), M, kt)
        acc = d if acc is None else acc + d
        pos += c
    return acc


def rms_norm(x, gamma, eps, use_ln_trick):
    """C6: native rms_norm, or the ANEMLL concat([x,-x]) LayerNorm equivalent."""
    if not use_ln_trick:
        return x.rms_norm(gamma, eps=eps)
    D = gamma.shape[0]
    doubled = af.concat([x, x * np.float16(-1.0)], axis=-1)
    normed = doubled.layer_norm(np.ones(2 * D, np.float16), np.zeros(2 * D, np.float16), eps=eps)
    half = normed.slice_by_size([0, 0], [x.shape[0], D])
    return half * gamma.astype(np.float16)


def expand_kv(t, n_rep, H_kv, S, dh):
    """C5: GQA expansion without repeat_interleave.

    t is [H_kv, S, dh]. Each KV head must be repeated n_rep times to line up
    with the query heads. Build it by concatenating per-head slices, which
    lowers to slice + concat rather than the slow repeat path.
    """
    if n_rep == 1:
        return t
    parts = []
    for h in range(H_kv):
        row = t.slice_by_size([h, 0, 0], [1, S, dh])
        parts.extend([row] * n_rep)
    return af.concat(parts, axis=0)


def build(M, kt, attn_impl, use_ln_trick, ff_chunk=2176, cfg=CFG, seed=0):
    d, H, H_kv, dh, ff, eps = (cfg["d"], cfg["n_heads"], cfg["n_kv"],
                               cfg["head_dim"], cfg["ff"], cfg["eps"])
    n_rep = H // H_kv
    rng = np.random.default_rng(seed)

    def w(n, k):
        return (rng.standard_normal((n, k)) / np.sqrt(k)).astype(np.float16)

    W = {"g1": np.ones(d, np.float16), "g2": np.ones(d, np.float16),
         "q": w(H * dh, d), "k": w(H_kv * dh, d), "v": w(H_kv * dh, d),
         "o": w(d, H * dh), "gate": w(ff, d), "up": w(ff, d), "down": w(d, ff)}

    cos, sin = af.rope_tables(M, dh)
    mask = np.triu(np.full((M, M), -np.inf, dtype=np.float32), k=1).astype(np.float16)
    scale = 1.0 / np.sqrt(dh)

    x = af.input([M, d])
    h = rms_norm(x, W["g1"], eps, use_ln_trick)
    q = tiled_linear(h, W["q"], M, kt).reshape([M, H, dh]).transpose([1, 0, 2])
    k = tiled_linear(h, W["k"], M, kt).reshape([M, H_kv, dh]).transpose([1, 0, 2])
    v = tiled_linear(h, W["v"], M, kt).reshape([M, H_kv, dh]).transpose([1, 0, 2])
    q = af.rope(q, cos, sin)
    k = af.rope(k, cos, sin)
    k = expand_kv(k, n_rep, H_kv, M, dh)
    v = expand_kv(v, n_rep, H_kv, M, dh)

    if attn_impl == "sdpa":
        # C4: this is a graph cut, so the block stops being one program.
        ctx = af.sdpa(q.reshape([1, H, M, dh]), k.reshape([1, H, M, dh]),
                      v.reshape([1, H, M, dh]), scale=scale, is_causal=True)
        ctx = ctx.reshape([H, M, dh])
    else:
        s = af.einsum("hid,hjd->hij", q, k) * np.float16(scale)
        s = s + mask
        ctx = af.einsum("hij,hjd->hid", s.softmax(axis=-1), v)

    ctx = ctx.transpose([1, 0, 2]).reshape([M, H * dh])
    x = x + tiled_linear(ctx, W["o"], M, kt)
    h2 = rms_norm(x, W["g2"], eps, use_ln_trick)
    y = x + swiglu_mlp(h2, W["gate"], W["up"], W["down"], M, kt, ff_chunk)
    return y, W, cos, sin


def flops(M, cfg=CFG):
    d, H, H_kv, dh, ff = cfg["d"], cfg["n_heads"], cfg["n_kv"], cfg["head_dim"], cfg["ff"]
    qkv = 2.0 * M * d * (H * dh + 2 * H_kv * dh)
    o = 2.0 * M * (H * dh) * d
    mlp = 2.0 * M * d * ff * 3
    attn = 2.0 * 2.0 * M * M * H * dh
    return {"total": qkv + o + mlp + attn, "mlp": mlp, "attn": attn, "proj": qkv + o}


def reference(xi, W, cos, sin, cfg=CFG):
    """fp32 numpy reference of the same block."""
    d, H, H_kv, dh, ff, eps = (cfg["d"], cfg["n_heads"], cfg["n_kv"],
                               cfg["head_dim"], cfg["ff"], cfg["eps"])
    n_rep, M = H // H_kv, xi.shape[0]
    f = lambda a: a.astype(np.float32)  # noqa: E731

    def rms(t, g):
        return t / np.sqrt((t ** 2).mean(-1, keepdims=True) + eps) * f(g)

    def rope_np(t, cos, sin):
        c, s = f(cos)[None], f(sin)[None]
        half = dh // 2
        rot = np.concatenate([-t[..., half:], t[..., :half]], axis=-1)
        return t * c + rot * s

    x = f(xi)
    h = rms(x, W["g1"])
    q = (h @ f(W["q"]).T).reshape(M, H, dh).transpose(1, 0, 2)
    k = (h @ f(W["k"]).T).reshape(M, H_kv, dh).transpose(1, 0, 2)
    v = (h @ f(W["v"]).T).reshape(M, H_kv, dh).transpose(1, 0, 2)
    q, k = rope_np(q, cos, sin), rope_np(k, cos, sin)
    k = np.repeat(k, n_rep, axis=0)
    v = np.repeat(v, n_rep, axis=0)
    s = q @ k.transpose(0, 2, 1) / np.sqrt(dh)
    s = s + np.triu(np.full((M, M), -np.inf, np.float32), k=1)
    s = s - s.max(-1, keepdims=True)
    p = np.exp(s)
    p /= p.sum(-1, keepdims=True)
    ctx = (p @ v).transpose(1, 0, 2).reshape(M, H * dh)
    x = x + ctx @ f(W["o"]).T
    h2 = rms(x, W["g2"])
    g = h2 @ f(W["gate"]).T
    g = g / (1 + np.exp(-g))
    u = h2 @ f(W["up"]).T
    return x + (g * u) @ f(W["down"]).T


def run(M, kt, attn_impl, use_ln_trick, iters, warmup, check, precision="fp16", ff_chunk=2176):
    t0 = time.perf_counter()
    y, W, cos, sin = build(M, kt, attn_impl, use_ln_trick, ff_chunk)
    prog = af.compile(y, **({} if precision == "fp16" else {"compress": precision}))
    build_s = time.perf_counter() - t0
    rng = np.random.default_rng(1)
    xi = rng.standard_normal((M, CFG["d"])).astype(np.float16)
    for _ in range(warmup):
        prog(xi)
    ts = []
    for _ in range(iters):
        t = time.perf_counter()
        prog(xi)
        ts.append(time.perf_counter() - t)
    ts = np.array(ts)
    med = float(np.median(ts))
    fl = flops(M)
    r = {"bench": "block", "M": M, "kt": kt, "ff_chunk": ff_chunk, "attn": attn_impl, "precision": precision,
         "ln_trick": use_ln_trick, "n_ops": prog.n_ops, "build_s": round(build_s, 2),
         "t_med_ms": med * 1e3, "t_min_ms": float(ts.min()) * 1e3,
         "t_sd_ms": float(ts.std()) * 1e3, "iters": iters,
         "gflop_total": fl["total"] / 1e9, "attn_frac": fl["attn"] / fl["total"],
         "tops_total": fl["total"] / med / 1e12}
    if check:
        ref = reference(xi, W, cos, sin)
        got = np.asarray(prog(xi), dtype=np.float32).reshape(ref.shape)
        rf, gf = ref.ravel(), got.ravel()
        r["cosine"] = float(rf @ gf / (np.linalg.norm(rf) * np.linalg.norm(gf)))
        r["max_rel_err"] = float(np.abs(got - ref).max() / np.abs(ref).max())
        r["argmax_match"] = float((ref.argmax(-1) == got.argmax(-1)).mean())
    prog.release()
    return r


def run_mlp_only(M, kt, iters, warmup, precision="fp16", ff_chunk=2176):
    """MLP alone, so the >=6 TFLOPS target can be read without attention in it."""
    d, ff = CFG["d"], CFG["ff"]
    rng = np.random.default_rng(0)
    w = lambda n, k: (rng.standard_normal((n, k)) / np.sqrt(k)).astype(np.float16)  # noqa: E731
    Wg, Wu, Wd = w(ff, d), w(ff, d), w(d, ff)
    x = af.input([M, d])
    h = x.rms_norm(np.ones(d, np.float16), eps=CFG["eps"])
    y = swiglu_mlp(h, Wg, Wu, Wd, M, kt, ff_chunk)
    prog = af.compile(y, **({} if precision == "fp16" else {"compress": precision}))
    xi = rng.standard_normal((M, d)).astype(np.float16)
    for _ in range(warmup):
        prog(xi)
    ts = []
    for _ in range(iters):
        t = time.perf_counter()
        prog(xi)
        ts.append(time.perf_counter() - t)
    ts = np.array(ts)
    med = float(np.median(ts))
    fl = 2.0 * M * d * ff * 3
    r = {"bench": "mlp", "M": M, "kt": kt, "ff_chunk": ff_chunk, "precision": precision, "n_ops": prog.n_ops,
         "t_med_ms": med * 1e3, "t_sd_ms": float(ts.std()) * 1e3, "iters": iters,
         "gflop": fl / 1e9, "tops_med": fl / med / 1e12}
    prog.release()
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, nargs="+", default=[256, 512, 1024])
    ap.add_argument("--kt", type=int, nargs="+", default=[2560])
    ap.add_argument("--attn", nargs="+", default=["manual", "sdpa"])
    ap.add_argument("--ln-trick", action="store_true")
    ap.add_argument("--precision", default="fp16")
    ap.add_argument("--mlp-only", action="store_true")
    ap.add_argument("--ff-chunk", type=int, default=2176)
    ap.add_argument("--iters", type=int, default=9)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    for M in a.M:
        for kt in a.kt:
            jobs = [("mlp", None)] if a.mlp_only else [("block", at) for at in a.attn]
            for kind, at in jobs:
                try:
                    r = (run_mlp_only(M, kt, a.iters, a.warmup, a.precision, a.ff_chunk) if kind == "mlp"
                         else run(M, kt, at, a.ln_trick, a.iters, a.warmup, a.check, a.precision, a.ff_chunk))
                except Exception as e:  # noqa: BLE001
                    r = {"bench": kind, "M": M, "kt": kt, "attn": at, "precision": a.precision,
                         "error": f"{type(e).__name__}: {e}"[:400]}
                print(json.dumps(r), flush=True)
                if a.out:
                    with open(a.out, "a") as f:
                        f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
