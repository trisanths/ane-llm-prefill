"""Same fp16 chained-linear workload as ane_bench.py, run on the GPU with MLX."""
import argparse
import json
import time

import mlx.core as mx
import numpy as np


def bench(M, K, N, layers, iters, warmup):
    mx.random.seed(0)
    ws = [(mx.random.normal((N, K)) / (K ** 0.5)).astype(mx.float16) for _ in range(layers)]
    x = mx.random.normal((M, K)).astype(mx.float16)
    mx.eval(ws, x)

    def run():
        h = x
        for w in ws:
            h = h @ w.T
        mx.eval(h)
        return h

    for _ in range(warmup):
        run()
    times = []
    for _ in range(iters):
        t = time.perf_counter()
        run()
        times.append(time.perf_counter() - t)
    times = np.array(times)
    med, best = float(np.median(times)), float(times.min())
    flops = 2.0 * M * K * N * layers
    wbytes = 2.0 * K * N * layers
    io_bytes = 2.0 * M * K + 2.0 * M * N
    return {
        "units": "mlx_gpu", "M": M, "K": K, "N": N, "layers": layers,
        "t_med_ms": med * 1e3, "t_min_ms": best * 1e3,
        "tflops_med": flops / med / 1e12, "tflops_best": flops / best / 1e12,
        "weight_MB": wbytes / 1e6,
        "weight_GBps_med": wbytes / med / 1e9, "weight_GBps_best": wbytes / best / 1e9,
        "total_GBps_med": (wbytes + io_bytes) / med / 1e9,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, nargs="+", default=[1, 8, 32, 128, 512, 1024, 2048, 4096])
    ap.add_argument("--K", type=int, default=4096)
    ap.add_argument("--N", type=int, default=4096)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    for M in a.M:
        r = bench(M, a.K, a.N, a.layers, a.iters, a.warmup)
        print(json.dumps(r), flush=True)
        if a.out:
            with open(a.out, "a") as f:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
