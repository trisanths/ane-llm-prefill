"""Drive the ANE with the tiled fp16 chain continuously for --seconds."""
import argparse
import time
import warnings

import numpy as np
import aneforge as af

warnings.filterwarnings("ignore")
from chain_tiled import layer  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--seconds", type=float, default=60)
ap.add_argument("--M", type=int, default=1024)
ap.add_argument("--K", type=int, default=4096)
ap.add_argument("--N", type=int, default=4096)
ap.add_argument("--layers", type=int, default=8)
ap.add_argument("--kt", type=int, default=1024)
a = ap.parse_args()

rng = np.random.default_rng(0)
Ws = [(rng.standard_normal((a.N, a.K)) / np.sqrt(a.K)).astype(np.float16) for _ in range(a.layers)]
x = af.input([a.M, a.K])
y = x
for W in Ws:
    y = layer(y, W, a.M, a.kt)
prog = af.compile(y)
xi = rng.standard_normal((a.M, a.K)).astype(np.float16)
for _ in range(5):
    prog(xi)
n, t0 = 0, time.perf_counter()
while time.perf_counter() - t0 < a.seconds:
    prog(xi)
    n += 1
el = time.perf_counter() - t0
fl = 2.0 * a.M * a.K * a.N * a.layers * n
print(f'{{"engine": "ane_sustained", "iters": {n}, "seconds": {el:.2f}, '
      f'"tflops": {fl / el / 1e12:.3f}, "kt": {a.kt}}}')
prog.release()
