"""MLX fp16 reference for the step 2b block, and the GPU baseline for it.

Same weights and same math as block2b.build(), so ANE output can be compared
against an fp16 engine rather than only against fp32 numpy. fp16-vs-fp16 is the
fair comparison for argmax agreement, since fp32 numpy has no rounding of its own.
"""
import time

import mlx.core as mx
import numpy as np

from block2b import CFG


def build_mlx(W, cos, sin, M, cfg=CFG):
    d, H, H_kv, dh, ff, eps = (cfg["d"], cfg["n_heads"], cfg["n_kv"],
                               cfg["head_dim"], cfg["ff"], cfg["eps"])
    n_rep = H // H_kv
    w = {k: mx.array(v) for k, v in W.items()}
    c, s = mx.array(np.asarray(cos, np.float16)), mx.array(np.asarray(sin, np.float16))
    mask = mx.array(np.triu(np.full((M, M), -np.inf, np.float32), k=1).astype(np.float16))

    def rms(t, g):
        return mx.fast.rms_norm(t, g, eps)

    def rope(t):
        half = dh // 2
        rot = mx.concatenate([-t[..., half:], t[..., :half]], axis=-1)
        return t * c[None] + rot * s[None]

    def block(x):
        h = rms(x, w["g1"])
        q = (h @ w["q"].T).reshape(M, H, dh).transpose(1, 0, 2)
        k = (h @ w["k"].T).reshape(M, H_kv, dh).transpose(1, 0, 2)
        v = (h @ w["v"].T).reshape(M, H_kv, dh).transpose(1, 0, 2)
        q, k = rope(q), rope(k)
        k = mx.repeat(k, n_rep, axis=0)
        v = mx.repeat(v, n_rep, axis=0)
        ctx = mx.fast.scaled_dot_product_attention(
            q[None], k[None], v[None], scale=1.0 / np.sqrt(dh), mask=mask)[0]
        ctx = ctx.transpose(1, 0, 2).reshape(M, H * dh)
        x = x + ctx @ w["o"].T
        h2 = rms(x, w["g2"])
        g = h2 @ w["gate"].T
        g = g * mx.sigmoid(g)
        u = h2 @ w["up"].T
        return x + (g * u) @ w["down"].T

    return block


def bench(W, cos, sin, M, iters=9, warmup=3):
    block = build_mlx(W, cos, sin, M)
    x = mx.array(np.random.default_rng(1).standard_normal((M, CFG["d"])).astype(np.float16))
    for _ in range(warmup):
        mx.eval(block(x))
    ts = []
    for _ in range(iters):
        t = time.perf_counter()
        mx.eval(block(x))
        ts.append(time.perf_counter() - t)
    ts = np.array(ts)
    return float(np.median(ts)), np.array(block(x), dtype=np.float32)
