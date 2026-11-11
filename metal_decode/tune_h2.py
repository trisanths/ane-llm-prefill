import itertools, time, sys
import mlx.core as mx
from qmv_h2 import qmm_h2
def med(fn, reps=30):
    for _ in range(4): mx.eval(fn())
    ts = []
    for _ in range(reps):
        a = time.perf_counter(); mx.eval(fn()); ts.append(time.perf_counter() - a)
    ts.sort(); return ts[reps // 2] * 1e6
N, K = 12288, 4096
W = (mx.random.normal((N, K)) * 0.02).astype(mx.bfloat16)
w, s, b = mx.quantize(W, group_size=64, bits=4)
Wd = mx.dequantize(w, s, b, group_size=64, bits=4).astype(mx.float32)
mx.eval(w, s, b, Wd)
Ms = (1, 2, 4, 6, 8)
xs = {M: mx.random.normal((M, K)).astype(mx.bfloat16) for M in Ms}
big = (mx.random.normal((8, K)) * 300).astype(mx.bfloat16)
mx.eval(list(xs.values()), big)
ref = {M: xs[M].astype(mx.float32) @ Wd.T for M in Ms}
refbig = big.astype(mx.float32) @ Wd.T
def err(o, r): return (mx.linalg.norm(o.astype(mx.float32) - r) / mx.linalg.norm(r)).item()
line = "mlx              "
for M in Ms:
    f = lambda: [mx.quantized_matmul(xs[M], w, s, b, transpose=True, group_size=64, bits=4) for _ in range(10)]
    line += f" M{M} {med(f)/10:6.1f}"
print(line)
for rps, wpt, sp in itertools.product((2, 4, 8), (1, 2), (1, 2, 4)):
    cfg = dict(rps=rps, wpt=wpt, msplit=sp)
    line = f"rps{rps} wpt{wpt} sp{sp}     "
    e = []
    for M in Ms:
        if M % sp:
            line += f" M{M}   -   "; continue
        f = lambda: [qmm_h2(xs[M], w, s, b, cfg) for _ in range(10)]
        line += f" M{M} {med(f)/10:6.1f}"
        e.append(err(f()[0], ref[M]))
    eb = err(qmm_h2(big, w, s, b, dict(cfg, msplit=sp if 8 % sp == 0 else 1)), refbig)
    print(line + f"   err {max(e):.1e}, |x|~300 err {eb:.1e}")
