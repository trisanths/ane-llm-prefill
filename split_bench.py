"""Experiment 1b: MLP on the ANE, attention on the GPU, split per layer.

Attention runs at ~1.4-1.8 TFLOPS on the ANE against ~9 for the MLP, so the
obvious question is whether handing attention to the GPU pays once the
round-trip is counted. Unified memory means no PCIe, but it does not mean free:
the ANE returns host numpy and MLX has to adopt it.

Measured pieces, all at the same layer shape:
  A   ANE: RMSNorm + Q/K/V projections            -> q, k, v
  X   host -> MLX adoption of q, k, v
  G   GPU: causal SDPA
  Y   MLX -> host of the context
  B   ANE: output projection + residual + RMSNorm + SwiGLU MLP
versus the monolithic ANE block that does all of it in one program.
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

import mlx.core as mx  # noqa: E402

from block2b import CFG, build, expand_kv, flops, swiglu_mlp, tiled_linear  # noqa: E402


def med(fn, iters, warmup):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(iters):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return float(np.median(ts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=1024)
    ap.add_argument("--kt", type=int, default=2048)
    ap.add_argument("--ff-chunk", type=int, default=2176)
    ap.add_argument("--iters", type=int, default=9)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    M, kt = a.M, a.kt
    d, H, H_kv, dh, ff, eps = (CFG["d"], CFG["n_heads"], CFG["n_kv"],
                               CFG["head_dim"], CFG["ff"], CFG["eps"])
    n_rep = H // H_kv
    rng = np.random.default_rng(0)
    w = lambda n, k: (rng.standard_normal((n, k)) / np.sqrt(k)).astype(np.float16)  # noqa: E731
    Wq, Wk, Wv, Wo = w(H * dh, d), w(H_kv * dh, d), w(H_kv * dh, d), w(d, H * dh)
    Wg, Wu, Wd = w(ff, d), w(ff, d), w(d, ff)
    g1, g2 = np.ones(d, np.float16), np.ones(d, np.float16)
    cos, sin = af.rope_tables(M, dh)
    xi = rng.standard_normal((M, d)).astype(np.float16)

    # --- A: ANE front half, emits q|k|v concatenated on the feature axis
    x = af.input([M, d])
    h = x.rms_norm(g1, eps=eps)
    qh = tiled_linear(h, Wq, M, kt)
    kh = tiled_linear(h, Wk, M, kt)
    vh = tiled_linear(h, Wv, M, kt)
    progA = af.compile(af.concat([qh, kh, vh], axis=-1))

    # --- B: ANE back half, takes x and ctx
    xb = af.input([M, d])
    ctx = af.input([M, H * dh])
    y = xb + tiled_linear(ctx, Wo, M, kt)
    h2 = y.rms_norm(g2, eps=eps)
    progB = af.compile(y + swiglu_mlp(h2, Wg, Wu, Wd, M, kt, a.ff_chunk))

    # --- monolithic baseline
    ymono, _, _, _ = build(M, kt, "manual", False, a.ff_chunk)
    progM = af.compile(ymono)

    t_A = med(lambda: progA(xi), a.iters, a.warmup)
    qkv = np.asarray(progA(xi))
    qn = np.ascontiguousarray(qkv[:, :H * dh]).reshape(M, H, dh).transpose(1, 0, 2)
    kn = np.ascontiguousarray(qkv[:, H * dh:H * dh + H_kv * dh]).reshape(M, H_kv, dh).transpose(1, 0, 2)
    vn = np.ascontiguousarray(qkv[:, H * dh + H_kv * dh:]).reshape(M, H_kv, dh).transpose(1, 0, 2)

    mask = mx.array(np.triu(np.full((M, M), -np.inf, np.float32), k=1).astype(np.float16))
    scale = 1.0 / np.sqrt(dh)

    def to_mlx():
        return mx.array(qn), mx.array(kn), mx.array(vn)

    t_X = med(lambda: mx.eval(to_mlx()), a.iters, a.warmup)
    qm, km, vm = to_mlx()
    km_e, vm_e = mx.repeat(km, n_rep, axis=0), mx.repeat(vm, n_rep, axis=0)
    mx.eval(qm, km_e, vm_e)

    def gpu_attn():
        o = mx.fast.scaled_dot_product_attention(qm[None], km_e[None], vm_e[None],
                                                 scale=scale, mask=mask)[0]
        mx.eval(o)
        return o
    t_G = med(gpu_attn, a.iters, a.warmup)
    o_gpu = gpu_attn()

    def from_mlx():
        return np.array(o_gpu.transpose(1, 0, 2).reshape(M, H * dh))
    t_Y = med(from_mlx, a.iters, a.warmup)
    ctx_np = from_mlx().astype(np.float16)

    t_B = med(lambda: progB(xi, ctx_np), a.iters, a.warmup)
    t_M = med(lambda: progM(xi), a.iters, a.warmup)

    # ANE-only attention at the best block size, for reference
    from attn_bench import attention_blocked
    qa, ka, va = af.input([H, M, dh]), af.input([H, M, dh]), af.input([H, M, dh])
    progAttn = af.compile(attention_blocked(qa, ka, va, M, dh, 512 if M <= 1024 else 256, scale))
    qi3 = [np.random.default_rng(i).standard_normal((H, M, dh)).astype(np.float16) for i in range(3)]
    t_ANEattn = med(lambda: progAttn(*qi3), a.iters, a.warmup)

    fl = flops(M)
    split_total = t_A + t_X + t_G + t_Y + t_B
    r = {"bench": "split", "M": M, "kt": kt,
         "ane_front_ms": t_A * 1e3, "host_to_mlx_ms": t_X * 1e3, "gpu_attn_ms": t_G * 1e3,
         "mlx_to_host_ms": t_Y * 1e3, "ane_back_ms": t_B * 1e3,
         "split_total_ms": split_total * 1e3, "monolithic_ane_ms": t_M * 1e3,
         "ane_attn_blocked_ms": t_ANEattn * 1e3,
         "transfer_ms": (t_X + t_Y) * 1e3,
         "transfer_frac_of_split": (t_X + t_Y) / split_total,
         "split_vs_monolithic": t_M / split_total,
         "gflop_total": fl["total"] / 1e9,
         "split_tflops": fl["total"] / split_total / 1e12,
         "monolithic_tflops": fl["total"] / t_M / 1e12}
    print(json.dumps(r), flush=True)
    if a.out:
        with open(a.out, "a") as f:
            f.write(json.dumps(r) + "\n")
    for p in (progA, progB, progM, progAttn):
        p.release()


if __name__ == "__main__":
    main()
