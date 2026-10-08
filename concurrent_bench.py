"""ANE + GPU concurrency.

Mode 'independent': two processes, one driving the ANE (Core ML, 8x4096x4096 fp16
chain) and one driving the GPU (MLX, same chain), each looping for --seconds and
reporting iterations. Run each alone and then both together; compare aggregate.

Mode 'split': one layer of 4096 -> 4096 at fixed M, N split in half. ANE half runs
in a thread via Core ML predict, GPU half via MLX on the main thread, join, concat.
Reports per-layer time vs each half alone and vs the whole layer on one device.
"""
import argparse
import json
import subprocess
import sys
import threading
import time

import numpy as np


def ane_worker(M, K, N, layers, seconds):
    import coremltools as ct
    from ane_bench import build_model
    model, _ = build_model(M, K, N, layers, ct.ComputeUnit.CPU_AND_NE)
    x = np.random.default_rng(1).standard_normal((M, K)).astype(np.float16)
    for _ in range(3):
        model.predict({"x": x})
    print("READY", flush=True)
    sys.stdin.readline()  # wait for go
    n, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        model.predict({"x": x})
        n += 1
    el = time.perf_counter() - t0
    print(json.dumps({"engine": "ane", "iters": n, "seconds": el,
                      "tflops": 2.0 * M * K * N * layers * n / el / 1e12}), flush=True)


def gpu_worker(M, K, N, layers, seconds):
    import mlx.core as mx
    mx.random.seed(0)
    ws = [(mx.random.normal((N, K)) / (K ** 0.5)).astype(mx.float16) for _ in range(layers)]
    x = mx.random.normal((M, K)).astype(mx.float16)
    mx.eval(ws, x)

    def run():
        h = x
        for w in ws:
            h = h @ w.T
        mx.eval(h)
    for _ in range(3):
        run()
    print("READY", flush=True)
    sys.stdin.readline()
    n, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        run()
        n += 1
    el = time.perf_counter() - t0
    print(json.dumps({"engine": "gpu", "iters": n, "seconds": el,
                      "tflops": 2.0 * M * K * N * layers * n / el / 1e12}), flush=True)


def independent(engines, M, K, N, layers, seconds):
    procs = []
    for e in engines:
        p = subprocess.Popen([sys.executable, __file__, "--worker", e, "--M", str(M), "--K", str(K),
                              "--N", str(N), "--layers", str(layers), "--seconds", str(seconds)],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        procs.append(p)
    for p in procs:
        while True:
            line = p.stdout.readline()
            if line.strip() == "READY":
                break
    for p in procs:
        p.stdin.write("go\n")
        p.stdin.flush()
    out = []
    for p in procs:
        for line in p.stdout:
            if line.startswith("{"):
                out.append(json.loads(line))
        p.wait()
    return {"mode": "independent", "engines": engines, "M": M, "K": K, "N": N, "layers": layers,
            "results": out, "aggregate_tflops": sum(r["tflops"] for r in out)}


def split(M, K, N, iters):
    import coremltools as ct
    import mlx.core as mx
    from ane_bench import build_model
    half = N // 2
    ane_half, _ = build_model(M, K, half, 1, ct.ComputeUnit.CPU_AND_NE, seed=0)
    ane_full, _ = build_model(M, K, N, 1, ct.ComputeUnit.CPU_AND_NE, seed=0)
    out_h = ane_half.get_spec().description.output[0].name
    out_f = ane_full.get_spec().description.output[0].name
    x = np.random.default_rng(1).standard_normal((M, K)).astype(np.float16)
    mx.random.seed(0)
    w_half = (mx.random.normal((half, K)) / (K ** 0.5)).astype(mx.float16)
    w_full = (mx.random.normal((N, K)) / (K ** 0.5)).astype(mx.float16)
    xm = mx.array(x)
    mx.eval(w_half, w_full, xm)

    def t_ane(model, name, n):
        for _ in range(2):
            model.predict({"x": x})
        ts = []
        for _ in range(n):
            t = time.perf_counter(); model.predict({"x": x})[name]; ts.append(time.perf_counter() - t)
        return float(np.median(ts)) * 1e3

    def t_gpu(w, n):
        for _ in range(2):
            mx.eval(xm @ w.T)
        ts = []
        for _ in range(n):
            t = time.perf_counter(); mx.eval(xm @ w.T); ts.append(time.perf_counter() - t)
        return float(np.median(ts)) * 1e3

    res = {"mode": "split", "M": M, "K": K, "N": N,
           "ane_full_ms": t_ane(ane_full, out_f, iters), "gpu_full_ms": t_gpu(w_full, iters),
           "ane_half_ms": t_ane(ane_half, out_h, iters), "gpu_half_ms": t_gpu(w_half, iters)}

    # combined: ANE half in a thread, GPU half on main thread, join, concat
    def combined_once():
        box = {}
        def ane_run():
            box["a"] = ane_half.predict({"x": x})[out_h]
        th = threading.Thread(target=ane_run)
        t = time.perf_counter()
        th.start()
        g = xm @ w_half.T
        mx.eval(g)
        th.join()
        y = np.concatenate([np.asarray(box["a"]), np.array(g)], axis=1)
        return time.perf_counter() - t, y
    for _ in range(2):
        combined_once()
    ts = [combined_once()[0] for _ in range(iters)]
    res["combined_ms"] = float(np.median(ts)) * 1e3
    res["ideal_combined_ms"] = max(res["ane_half_ms"], res["gpu_half_ms"])
    res["sync_overhead_ms"] = res["combined_ms"] - res["ideal_combined_ms"]
    res["speedup_vs_gpu_full"] = res["gpu_full_ms"] / res["combined_ms"]
    res["speedup_vs_ane_full"] = res["ane_full_ms"] / res["combined_ms"]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", choices=["ane", "gpu"])
    ap.add_argument("--mode", default="independent", choices=["independent", "split"])
    ap.add_argument("--engines", nargs="+", default=["ane", "gpu"])
    ap.add_argument("--M", type=int, default=1024)
    ap.add_argument("--K", type=int, default=4096)
    ap.add_argument("--N", type=int, default=4096)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--seconds", type=float, default=10)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.worker == "ane":
        return ane_worker(a.M, a.K, a.N, a.layers, a.seconds)
    if a.worker == "gpu":
        return gpu_worker(a.M, a.K, a.N, a.layers, a.seconds)
    r = independent(a.engines, a.M, a.K, a.N, a.layers, a.seconds) if a.mode == "independent" \
        else split(a.M, a.K, a.N, a.iters)
    print(json.dumps(r), flush=True)
    if a.out:
        with open(a.out, "a") as f:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
