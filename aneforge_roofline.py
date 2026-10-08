"""Step 2a: roofline through ANEForge (direct ANE dispatch, no Core ML placement).

Sweeps M for chained KxN linears in one fused program, at fp16 / int8 / int4,
plus a tiling test (16384-wide built from 4096-wide column chunks) and a
dispatch-floor measurement. Medians over >= 5 runs.
"""
import argparse
import json
import time
import warnings

import numpy as np
import aneforge as af

warnings.filterwarnings("ignore", category=af.DispatchFloorWarning)
warnings.filterwarnings("ignore", category=af.PrecisionWarning)


def timeit(prog, xi, iters, warmup):
    for _ in range(warmup):
        prog(xi)
    ts = []
    for _ in range(iters):
        t = time.perf_counter()
        prog(xi)
        ts.append(time.perf_counter() - t)
    ts = np.array(ts)
    return float(np.median(ts)), float(ts.min()), float(ts.std())


def compile_opts(precision):
    """fp16 | int8 (per-channel weights) | int4 (LUT)."""
    if precision == "fp16":
        return {}
    if precision == "int8":
        return {"compress": "int8"}
    if precision == "int4":
        return {"compress": "int4"}
    raise ValueError(precision)


def chain(M, K, N, layers, precision, iters, warmup, check):
    rng = np.random.default_rng(0)
    # square chain needs K==N to restack; otherwise only first layer is KxN
    Ws = []
    kk = K
    for _ in range(layers):
        Ws.append((rng.standard_normal((N, kk)) / np.sqrt(kk)).astype(np.float16))
        kk = N
    x = af.input([M, K])
    y = x
    for W in Ws:
        y = y.linear(W)
    t0 = time.perf_counter()
    prog = af.compile(y, **compile_opts(precision))
    build_s = time.perf_counter() - t0
    xi = rng.standard_normal((M, K)).astype(np.float16)
    med, best, sd = timeit(prog, xi, iters, warmup)
    flops = 2.0 * M * K * N * layers
    wbytes = 2.0 * K * N * layers
    r = {"bench": "chain", "engine": "aneforge", "precision": precision,
         "M": M, "K": K, "N": N, "layers": layers, "n_ops": prog.n_ops,
         "build_s": round(build_s, 2), "t_med_ms": med * 1e3, "t_min_ms": best * 1e3,
         "t_sd_ms": sd * 1e3, "iters": iters,
         "tops_med": flops / med / 1e12, "tops_best": flops / best / 1e12,
         "weight_MB_fp16": wbytes / 1e6, "weight_GBps_med": wbytes / med / 1e9}
    if check:
        ref = xi.astype(np.float32)
        for W in Ws:
            ref = ref @ W.astype(np.float32).T
        got = np.asarray(prog(xi), dtype=np.float32).reshape(ref.shape)
        r["max_rel_err"] = float(np.abs(got - ref).max() / np.abs(ref).max())
        rf, gf = ref.ravel(), got.ravel()
        r["cosine"] = float(rf @ gf / (np.linalg.norm(rf) * np.linalg.norm(gf)))
    prog.release()
    return r


def tiled(M, K, N, chunk, precision, iters, warmup, check):
    """Build an MxK @ KxN as ceil(N/chunk) column chunks concatenated, one program."""
    rng = np.random.default_rng(0)
    nch = N // chunk
    Ws = [(rng.standard_normal((chunk, K)) / np.sqrt(K)).astype(np.float16) for _ in range(nch)]
    x = af.input([M, K])
    parts = [x.linear(W) for W in Ws]
    y = af.concat(parts, axis=-1) if nch > 1 else parts[0]
    t0 = time.perf_counter()
    prog = af.compile(y, **compile_opts(precision))
    build_s = time.perf_counter() - t0
    xi = rng.standard_normal((M, K)).astype(np.float16)
    med, best, sd = timeit(prog, xi, iters, warmup)
    flops = 2.0 * M * K * N
    r = {"bench": "tiled", "engine": "aneforge", "precision": precision,
         "M": M, "K": K, "N": N, "chunk": chunk, "n_chunks": nch, "n_ops": prog.n_ops,
         "build_s": round(build_s, 2), "t_med_ms": med * 1e3, "t_min_ms": best * 1e3,
         "t_sd_ms": sd * 1e3, "iters": iters,
         "tops_med": flops / med / 1e12, "tops_best": flops / best / 1e12,
         "weight_MB_fp16": 2.0 * K * N / 1e6,
         "chunk_MB_fp16": 2.0 * K * chunk / 1e6}
    if check:
        ref = np.concatenate([xi.astype(np.float32) @ W.astype(np.float32).T for W in Ws], axis=-1)
        got = np.asarray(prog(xi), dtype=np.float32).reshape(ref.shape)
        r["max_rel_err"] = float(np.abs(got - ref).max() / np.abs(ref).max())
    prog.release()
    return r


def dispatch_floor(iters, warmup):
    """Smallest useful program: one tiny linear. Time is nearly all fixed cost."""
    rng = np.random.default_rng(0)
    out = []
    for n in (1, 8, 32):
        W = (rng.standard_normal((n, n)) / np.sqrt(n)).astype(np.float16)
        x = af.input([1, n])
        prog = af.compile(x.linear(W))
        xi = rng.standard_normal((1, n)).astype(np.float16)
        med, best, sd = timeit(prog, xi, iters, warmup)
        out.append({"bench": "dispatch_floor", "engine": "aneforge", "size": n,
                    "n_ops": prog.n_ops, "t_med_us": med * 1e6, "t_min_us": best * 1e6,
                    "t_sd_us": sd * 1e6, "iters": iters})
        prog.release()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="chain", choices=["chain", "tiled", "floor"])
    ap.add_argument("--M", type=int, nargs="+", default=[1, 8, 32, 128, 512, 1024, 2048, 4096])
    ap.add_argument("--K", type=int, default=4096)
    ap.add_argument("--N", type=int, default=4096)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument("--precision", nargs="+", default=["fp16"], choices=["fp16", "int8", "int4"])
    ap.add_argument("--iters", type=int, default=9)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    rows = []
    if a.bench == "floor":
        rows = dispatch_floor(a.iters, a.warmup)
    else:
        for p in a.precision:
            for M in a.M:
                try:
                    if a.bench == "chain":
                        r = chain(M, a.K, a.N, a.layers, p, a.iters, a.warmup, a.check)
                    else:
                        r = tiled(M, a.K, a.N, a.chunk, p, a.iters, a.warmup, a.check)
                except Exception as e:  # noqa: BLE001
                    r = {"bench": a.bench, "engine": "aneforge", "precision": p, "M": M,
                         "K": a.K, "N": a.N, "layers": a.layers, "error": f"{type(e).__name__}: {e}"[:400]}
                rows.append(r)
    for r in rows:
        print(json.dumps(r), flush=True)
        if a.out:
            with open(a.out, "a") as f:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
