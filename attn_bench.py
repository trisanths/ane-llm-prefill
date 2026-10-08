"""Experiment 1a: key-axis-blocked attention on the ANE.

Baseline materializes the full [H, M, M] score matrix. The blocked version walks
the key axis in blocks of B, keeping a running max and sum (online / flash
softmax), so the widest intermediate is [H, M, B].

Two reasons to expect a win, and they are separable:
  - the [H, M, M] tensor never exists, which matters as M grows;
  - the PV contraction goes from width M to width B, and step 2a showed
    throughput tracks reduction width. At M=1024 the baseline PV contracts over
    1024, which is past the ~2048 plateau but well above the B=128..512 range.

FLOPs are held identical to the baseline: every block uses all M query rows, so
the upper-triangular waste is the same in both. This is a like-for-like timing
comparison, not a causal-skipping optimization.
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

NEG = np.float16(-1e4)   # fp16-safe stand-in for -inf; -inf minus -inf is nan


def causal_mask(M, j, B):
    """[M, B] additive mask for key block starting at j."""
    qi = np.arange(M)[:, None]
    kj = np.arange(j, j + B)[None, :]
    return np.where(kj <= qi, np.float16(0), NEG).astype(np.float16)


def attention_full(q, k, v, M, scale):
    mask = np.triu(np.full((M, M), -np.inf, np.float32), k=1).astype(np.float16)
    s = af.einsum("hid,hjd->hij", q, k) * np.float16(scale)
    return af.einsum("hij,hjd->hid", (s + mask).softmax(axis=-1), v)


def attention_blocked(q, k, v, M, dh, B, scale):
    """Key-axis blocked attention with a FIXED softmax max of zero.

    The textbook online (flash) form, carrying a running max and rescaling each
    block, does not compile on this ANE compiler. It fails whenever the final
    normalization joins the accumulated output and the accumulated denominator,
    because both descend from the same deep chain; combining either with a
    shallow operand is fine, so it is the shared-deep-ancestor diamond that
    breaks, not the division. See NOTES.md.

    Dropping the running max removes the diamond's depth and compiles at every
    block size. The cost is numerical: scores are exponentiated without being
    recentred, so a score above about 11 overflows fp16. For RMSNorm'd
    activations scores sit near O(1), and the cosine check below is what
    validates it per configuration. This kernel is not safe for unbounded
    scores.
    """
    o = l = None
    for j in range(0, M, B):
        b = min(B, M - j)
        kj = k.slice_by_size([0, j, 0], [k.shape[0], b, dh])
        vj = v.slice_by_size([0, j, 0], [v.shape[0], b, dh])
        s = af.einsum("hid,hjd->hij", q, kj) * np.float16(scale)
        p = (s + causal_mask(M, j, b)).exp()    # masked entries -> exp(-1e4) = 0
        sj = p.sum([-1])
        oj = af.einsum("hij,hjd->hid", p, vj)
        l = sj if l is None else l + sj
        o = oj if o is None else o + oj
    return o * l.pow(np.float16(-1.0))


def ref_attention(q, k, v, M, scale):
    q, k, v = (x.astype(np.float32) for x in (q, k, v))
    s = q @ k.transpose(0, 2, 1) * scale
    s = s + np.triu(np.full((M, M), -np.inf, np.float32), k=1)
    s = s - s.max(-1, keepdims=True)
    p = np.exp(s)
    p /= p.sum(-1, keepdims=True)
    return p @ v


def run(M, H, dh, B, iters, warmup, check):
    scale = 1.0 / np.sqrt(dh)
    q = af.input([H, M, dh])
    k = af.input([H, M, dh])
    v = af.input([H, M, dh])
    y = attention_full(q, k, v, M, scale) if B == 0 else attention_blocked(q, k, v, M, dh, B, scale)
    t0 = time.perf_counter()
    prog = af.compile(y)
    build_s = time.perf_counter() - t0
    qi, ki, vi = (np.random.default_rng(i).standard_normal((H, M, dh)).astype(np.float16)
                  for i in range(3))
    for _ in range(warmup):
        prog(qi, ki, vi)
    ts = []
    for _ in range(iters):
        t = time.perf_counter()
        prog(qi, ki, vi)
        ts.append(time.perf_counter() - t)
    ts = np.array(ts)
    med = float(np.median(ts))
    fl = 4.0 * M * M * H * dh                    # QK^T and PV
    r = {"bench": "attn", "M": M, "H": H, "dh": dh, "block": B or M,
         "mode": "full" if B == 0 else "blocked", "n_ops": prog.n_ops,
         "build_s": round(build_s, 2), "t_med_ms": med * 1e3,
         "t_sd_ms": float(ts.std()) * 1e3, "iters": iters,
         "tflops_med": fl / med / 1e12,
         "peak_score_MB": 2.0 * H * M * (B or M) / 1e6}
    if check:
        ref = ref_attention(qi, ki, vi, M, scale)
        got = np.asarray(prog(qi, ki, vi), dtype=np.float32).reshape(ref.shape)
        a, b_ = ref.ravel().astype(np.float64), got.ravel().astype(np.float64)
        r["cosine"] = float(a @ b_ / (np.linalg.norm(a) * np.linalg.norm(b_)))
        r["max_rel_err"] = float(np.abs(got - ref).max() / np.abs(ref).max())
    prog.release()
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, nargs="+", default=[1024])
    ap.add_argument("--H", type=int, default=40)
    ap.add_argument("--dh", type=int, default=128)
    ap.add_argument("--blocks", type=int, nargs="+", default=[0, 128, 256, 512])
    ap.add_argument("--iters", type=int, default=9)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    for M in a.M:
        for B in a.blocks:
            try:
                r = run(M, a.H, a.dh, B, a.iters, a.warmup, a.check)
            except Exception as e:  # noqa: BLE001
                r = {"bench": "attn", "M": M, "block": B or M,
                     "mode": "full" if B == 0 else "blocked",
                     "error": f"{type(e).__name__}: {e}"[:300]}
            print(json.dumps(r), flush=True)
            if a.out:
                with open(a.out, "a") as f:
                    f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
