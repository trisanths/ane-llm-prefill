"""Drive the GPU with fp16 matmuls continuously for --seconds, reporting achieved TFLOPS."""
import argparse
import time

import mlx.core as mx
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--seconds", type=float, default=60)
ap.add_argument("--M", type=int, default=1024)
ap.add_argument("--K", type=int, default=4096)
ap.add_argument("--N", type=int, default=4096)
ap.add_argument("--layers", type=int, default=8)
a = ap.parse_args()

mx.set_default_device(mx.gpu)
ws = [(mx.random.normal((a.N, a.K)) / (a.K ** 0.5)).astype(mx.float16) for _ in range(a.layers)]
x = mx.random.normal((a.M, a.K)).astype(mx.float16)
mx.eval(ws, x)


def run():
    h = x
    for w in ws:
        h = h @ w.T
    mx.eval(h)


for _ in range(5):
    run()
n, t0 = 0, time.perf_counter()
while time.perf_counter() - t0 < a.seconds:
    run()
    n += 1
el = time.perf_counter() - t0
fl = 2.0 * a.M * a.K * a.N * a.layers * n
print(f"{{\"engine\": \"mlx_gpu_sustained\", \"iters\": {n}, \"seconds\": {el:.2f}, "
      f"\"tflops\": {fl / el / 1e12:.3f}, \"device\": \"{mx.default_device()}\"}}")
