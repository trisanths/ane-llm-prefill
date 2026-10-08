"""True 2D tiling test: build an MxK @ KxN GEMM entirely from tiles of a chosen
byte budget, splitting BOTH the reduction axis K and the output axis N.

Column-only chunking leaves each weight tile at K x chunk, which for K=16384
is 134 MB and never tests a 32 MB residency hypothesis. Here tile bytes are
2 * kt * nt, so kt = nt = 4096 gives exactly 32 MB.
"""
import argparse
import json
import time
import warnings

import numpy as np
import aneforge as af

warnings.filterwarnings("ignore", category=af.DispatchFloorWarning)
warnings.filterwarnings("ignore", category=af.PrecisionWarning)


def run(M, K, N, kt, nt, precision, iters, warmup, check):
    rng = np.random.default_rng(0)
    W = (rng.standard_normal((N, K)) / np.sqrt(K)).astype(np.float16)
    nk, nn = K // kt, N // nt
    x = af.input([M, K])
    cols = []
    for j in range(nn):
        acc = None
        for i in range(nk):
            xi = x.slice_by_size([0, i * kt], [M, kt]) if nk > 1 else x
            Wij = np.ascontiguousarray(W[j * nt:(j + 1) * nt, i * kt:(i + 1) * kt])
            p = xi.linear(Wij)
            acc = p if acc is None else acc + p
        cols.append(acc)
    y = af.concat(cols, axis=-1) if nn > 1 else cols[0]
    t0 = time.perf_counter()
    kw = {} if precision == "fp16" else {"compress": precision}
    prog = af.compile(y, **kw)
    build_s = time.perf_counter() - t0
    xi_np = rng.standard_normal((M, K)).astype(np.float16)
    for _ in range(warmup):
        prog(xi_np)
    ts = []
    for _ in range(iters):
        t = time.perf_counter()
        prog(xi_np)
        ts.append(time.perf_counter() - t)
    ts = np.array(ts)
    med = float(np.median(ts))
    flops = 2.0 * M * K * N
    r = {"bench": "tile2d", "engine": "aneforge", "precision": precision,
         "M": M, "K": K, "N": N, "kt": kt, "nt": nt, "n_tiles": nk * nn,
         "tile_MB": 2.0 * kt * nt / 1e6, "n_ops": prog.n_ops,
         "build_s": round(build_s, 2), "t_med_ms": med * 1e3, "t_min_ms": float(ts.min()) * 1e3,
         "t_sd_ms": float(ts.std()) * 1e3, "iters": iters,
         "tops_med": flops / med / 1e12}
    if check:
        ref = xi_np.astype(np.float32) @ W.astype(np.float32).T
        got = np.asarray(prog(xi_np), dtype=np.float32).reshape(ref.shape)
        r["max_rel_err"] = float(np.abs(got - ref).max() / np.abs(ref).max())
    prog.release()
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=1024)
    ap.add_argument("--K", type=int, default=16384)
    ap.add_argument("--N", type=int, default=16384)
    ap.add_argument("--tiles", nargs="+", default=["4096x4096"],
                    help="kt x nt pairs, e.g. 4096x4096 8192x4096 16384x16384")
    ap.add_argument("--precision", default="fp16")
    ap.add_argument("--iters", type=int, default=7)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    for spec in a.tiles:
        kt, nt = (int(v) for v in spec.lower().split("x"))
        try:
            r = run(a.M, a.K, a.N, kt, nt, a.precision, a.iters, a.warmup, a.check)
        except Exception as e:  # noqa: BLE001
            r = {"bench": "tile2d", "M": a.M, "K": a.K, "N": a.N, "kt": kt, "nt": nt,
                 "precision": a.precision, "error": f"{type(e).__name__}: {e}"[:300]}
        print(json.dumps(r), flush=True)
        if a.out:
            with open(a.out, "a") as f:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
