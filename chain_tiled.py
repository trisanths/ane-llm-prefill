"""Chained linears where each layer is K-tiled, the configuration a real
transformer block would use. Answers whether tiling and int8 compose."""
import argparse, json, time, warnings
import numpy as np, aneforge as af
warnings.filterwarnings("ignore", category=af.DispatchFloorWarning)
warnings.filterwarnings("ignore", category=af.PrecisionWarning)


def layer(x, W, M, kt):
    N, K = W.shape
    if kt >= K:
        return x.linear(W)
    acc = None
    for i in range(K // kt):
        xi = x.slice_by_size([0, i * kt], [M, kt])
        p = xi.linear(np.ascontiguousarray(W[:, i * kt:(i + 1) * kt]))
        acc = p if acc is None else acc + p
    return acc


def run(M, K, N, layers, kt, precision, iters, warmup, check):
    rng = np.random.default_rng(0)
    Ws = [(rng.standard_normal((N, K)) / np.sqrt(K)).astype(np.float16) for _ in range(layers)]
    x = af.input([M, K])
    y = x
    for W in Ws:
        y = layer(y, W, M, kt)
    t0 = time.perf_counter()
    prog = af.compile(y, **({} if precision == "fp16" else {"compress": precision}))
    build_s = time.perf_counter() - t0
    xi = rng.standard_normal((M, K)).astype(np.float16)
    for _ in range(warmup):
        prog(xi)
    ts = []
    for _ in range(iters):
        t = time.perf_counter(); prog(xi); ts.append(time.perf_counter() - t)
    ts = np.array(ts); med = float(np.median(ts))
    r = {"bench": "chain_tiled", "engine": "aneforge", "precision": precision, "M": M,
         "K": K, "N": N, "layers": layers, "kt": kt, "n_ops": prog.n_ops,
         "build_s": round(build_s, 2), "t_med_ms": med * 1e3, "t_sd_ms": float(ts.std()) * 1e3,
         "iters": iters, "tops_med": 2.0 * M * K * N * layers / med / 1e12}
    if check:
        ref = xi.astype(np.float32)
        for W in Ws:
            ref = ref @ W.astype(np.float32).T
        got = np.asarray(prog(xi), dtype=np.float32).reshape(ref.shape)
        rf, gf = ref.ravel(), got.ravel()
        r["max_rel_err"] = float(np.abs(got - ref).max() / np.abs(ref).max())
        r["cosine"] = float(rf @ gf / (np.linalg.norm(rf) * np.linalg.norm(gf)))
    prog.release()
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=1024); ap.add_argument("--K", type=int, default=4096)
    ap.add_argument("--N", type=int, default=4096); ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--kt", type=int, nargs="+", default=[2048, 4096])
    ap.add_argument("--precision", nargs="+", default=["fp16", "int8"])
    ap.add_argument("--iters", type=int, default=9); ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--check", action="store_true"); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    for p in a.precision:
        for kt in a.kt:
            try:
                r = run(a.M, a.K, a.N, a.layers, kt, p, a.iters, a.warmup, a.check)
            except Exception as e:
                r = {"bench": "chain_tiled", "precision": p, "kt": kt, "M": a.M,
                     "error": f"{type(e).__name__}: {e}"[:300]}
            print(json.dumps(r), flush=True)
            if a.out:
                open(a.out, "a").write(json.dumps(r) + "\n")
