"""Item 3: separate reduction width from weight volume.

Step 2a found throughput tracks the reduction-axis tile width kt. Step 2b's
addendum found the W8A16 speedup collapses once the reduction is tiled, which
suggested both relieve one bottleneck, but could not prove the bottleneck is the
reduction rather than the weight bytes, because in every shape tested the two
moved together.

3a separates them: hold weight volume and FLOPs FIXED while varying K by 16x.
K x N stays constant, so every configuration bakes the same bytes and does the
same arithmetic. Only the shape of the contraction changes. If throughput tracks
K, the reduction is the bottleneck. If it is flat, volume is.

Running each in fp16 and W8A16 also tests the mechanism: if int8 helps by
halving the bytes pulled per contraction step, its advantage should be largest
where K is widest and should vanish where K is narrow.

3b holds the tile fixed and sweeps M at two K values, to check whether the
penalty for wide K is a constant factor or grows with work.
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


def run(M, K, N, precision, kt, iters, warmup, check):
    rng = np.random.default_rng(0)
    W = (rng.standard_normal((N, K)) / np.sqrt(K)).astype(np.float16)
    x = af.input([M, K])
    if kt and kt < K:
        acc, pos = None, 0
        while pos < K:
            b = min(kt, K - pos)
            p = x.slice_by_size([0, pos], [M, b]).linear(np.ascontiguousarray(W[:, pos:pos + b]))
            acc = p if acc is None else acc + p
            pos += b
        y = acc
    else:
        y = x.linear(W)
    prog = af.compile(y, **({} if precision == "fp16" else {"compress": precision}))
    xi = rng.standard_normal((M, K)).astype(np.float16)
    for _ in range(warmup):
        prog(xi)
    ts = []
    for _ in range(iters):
        t = time.perf_counter()
        prog(xi)
        ts.append(time.perf_counter() - t)
    ts = np.array(ts)
    m = float(np.median(ts))
    fl = 2.0 * M * K * N
    r = {"bench": "redlaw", "M": M, "K": K, "N": N, "precision": precision,
         "kt": kt or K, "n_ops": prog.n_ops, "t_med_ms": m * 1e3,
         "t_sd_ms": float(ts.std()) * 1e3, "iters": iters,
         "tflops_med": fl / m / 1e12, "weight_MB": 2.0 * K * N / 1e6,
         "gflop": fl / 1e9}
    if check:
        ref = xi.astype(np.float32) @ W.astype(np.float32).T
        got = np.asarray(prog(xi), dtype=np.float32).reshape(ref.shape)
        a, b_ = ref.ravel().astype(np.float64), got.ravel().astype(np.float64)
        r["cosine"] = float(a @ b_ / (np.linalg.norm(a) * np.linalg.norm(b_)))
    prog.release()
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="3a", choices=["3a", "3b"])
    ap.add_argument("--M", type=int, nargs="+", default=[1024])
    ap.add_argument("--volume", type=int, default=4096 * 4096,
                    help="K*N held constant for 3a")
    ap.add_argument("--Ks", type=int, nargs="+", default=[512, 1024, 2048, 4096, 8192, 16384])
    ap.add_argument("--K", type=int, default=4096, help="for 3b")
    ap.add_argument("--N", type=int, default=4096, help="for 3b")
    ap.add_argument("--kt", type=int, default=0, help="0 = untiled")
    ap.add_argument("--precision", nargs="+", default=["fp16", "int8"])
    ap.add_argument("--iters", type=int, default=9)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    jobs = []
    if a.mode == "3a":
        for M in a.M:
            for K in a.Ks:
                N = a.volume // K
                for p in a.precision:
                    jobs.append((M, K, N, p, a.kt))
    else:
        for M in a.M:
            for p in a.precision:
                jobs.append((M, a.K, a.N, p, a.kt))
    for M, K, N, p, kt in jobs:
        try:
            r = run(M, K, N, p, kt, a.iters, a.warmup, a.check)
        except Exception as e:  # noqa: BLE001
            r = {"bench": "redlaw", "M": M, "K": K, "N": N, "precision": p, "kt": kt,
                 "error": f"{type(e).__name__}: {e}"[:250]}
        print(json.dumps(r), flush=True)
        if a.out:
            with open(a.out, "a") as f:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
