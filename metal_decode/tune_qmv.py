"""Sweep qmv_multi settings on one 8B-sized matrix and report us per call
for M = 1..8 against MLX, plus error against an fp32 dequantized reference."""
import itertools
import time

import mlx.core as mx

from qmv_multi import qmm_small


def med(fn, reps=30):
    for _ in range(4):
        mx.eval(fn())
    ts = []
    for _ in range(reps):
        a = time.perf_counter()
        mx.eval(fn())
        ts.append(time.perf_counter() - a)
    ts.sort()
    return ts[reps // 2] * 1e6


N, K = 12288, 4096
W = (mx.random.normal((N, K)) * 0.02).astype(mx.bfloat16)
w, s, b = mx.quantize(W, group_size=64, bits=4)
Wd = mx.dequantize(w, s, b, group_size=64, bits=4).astype(mx.float32)
mx.eval(w, s, b, Wd)
xs = {M: mx.random.normal((M, K)).astype(mx.bfloat16) for M in (1, 2, 4, 6, 8)}
mx.eval(list(xs.values()))
ref32 = {M: xs[M].astype(mx.float32) @ Wd.T for M in xs}
mx.eval(list(ref32.values()))


def err(out, M):
    r = ref32[M]
    return (mx.linalg.norm(out.astype(mx.float32) - r) / mx.linalg.norm(r)).item()


line = "mlx      "
for M in xs:
    f = lambda: [mx.quantized_matmul(xs[M], w, s, b, transpose=True, group_size=64, bits=4) for _ in range(10)]
    line += f" M{M} {med(f)/10:7.1f} ({err(f()[0], M):.1e})"
print(line)
for rps, xt, vpt, sp in itertools.product((2, 4, 8), ("float", "half"), (8, 16), (1, 2, 4, 8)):
    cfg = dict(rps=rps, xt=xt, at="float", vpt=vpt, msplit=sp)
    line = f"rps{rps} {xt:5s} v{vpt:<2d} sp{sp}"
    for M in xs:
        if M % sp:
            line += f" M{M}     -        "
            continue
        f = lambda: [qmm_small(xs[M], w, s, b, cfg) for _ in range(10)]
        line += f" M{M} {med(f)/10:7.1f} ({err(f()[0], M):.1e})"
    print(line)
