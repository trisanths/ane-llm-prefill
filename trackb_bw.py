"""Track B.1: achievable GPU read bandwidth, the ceiling for weight-streaming decode."""
import time, json
import mlx.core as mx
import numpy as np

def bench(nbytes, iters=20):
    n = nbytes // 2
    a = mx.random.normal((n,)).astype(mx.float16)
    mx.eval(a)
    for _ in range(3):
        mx.eval(mx.sum(a))
    ts = []
    for _ in range(iters):
        t = time.perf_counter(); mx.eval(mx.sum(a)); ts.append(time.perf_counter() - t)
    return nbytes / float(np.median(ts)) / 1e9

for mb in (256, 1024, 2300):
    print(json.dumps({"bytes_MB": mb, "read_GBps": round(bench(mb * 2**20), 1)}), flush=True)
