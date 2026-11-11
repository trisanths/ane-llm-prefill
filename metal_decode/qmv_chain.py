"""Cost of kernel boundaries in batch-1 decode.

Runs the 113 weight matmuls of one Qwen3 decode token (4 merged per layer,
plus the lm_head) three ways, in one eval each:
  chain        each matmul depends on the previous one, as in decode;
  independent  every matmul reads the same input, so MLX can overlap them;
  one          a single matmul over the same bytes (the stream floor).
The gap between chain and independent is the price of the dependency
barriers between kernels, separate from per-shape kernel efficiency.
"""
import sys
import time

import mlx.core as mx
from mlx_lm import load

from fused_qwen3 import Fused, GS, BITS


def qmv(x, w):
    return mx.quantized_matmul(x, *w, transpose=True, group_size=GS, bits=BITS)


def med(fn, reps=40):
    for _ in range(5):
        mx.eval(fn())
    ts = []
    for _ in range(reps):
        a = time.perf_counter()
        mx.eval(fn())
        ts.append(time.perf_counter() - a)
    ts.sort()
    return ts[len(ts) // 2]


def main(path):
    model, _ = load(path)
    f = Fused(model)
    E = f.emb
    head = (E.weight, E.scales, E.biases)
    mats = []
    for L in f.layers:
        mats += [L["qkv"], L["o"], L["gu"], L["down"]]
    nbytes = sum(sum(a.nbytes for a in m) for m in mats + [head])
    D, I = f.D, f.I
    x0 = mx.random.normal((D,)).astype(f.dtype) * 0.05
    xi = mx.random.normal((I,)).astype(f.dtype) * 0.05
    mx.eval(x0, xi)

    def chain():
        x = x0
        for li in range(len(f.layers)):
            L = f.layers[li]
            x = qmv(x, L["qkv"])[:D]
            x = qmv(x, L["o"])
            x = qmv(x, L["gu"])[:I]
            x = qmv(x, L["down"])
        return qmv(x, head)

    def independent():
        outs = []
        for L in f.layers:
            outs += [qmv(x0, L["qkv"]), qmv(x0, L["o"]), qmv(x0, L["gu"]), qmv(xi, L["down"])]
        outs.append(qmv(x0, head))
        return outs

    k = 4096
    rows = int(nbytes / (k * 4.5 / 8))
    wq = mx.quantize(mx.random.normal((rows, k)).astype(f.dtype), group_size=GS, bits=BITS)
    xk = mx.random.normal((k,)).astype(f.dtype)
    mx.eval(wq, xk)
    one_bytes = sum(a.nbytes for a in wq)

    def one():
        return qmv(xk, wq)

    def per_shape(name, w, x, n=28):
        return lambda: [qmv(x, w) for _ in range(1)]

    print(f"bytes per token in matmuls: {nbytes/1e6:.0f} MB, 113 kernels")
    for _ in range(2):
        tc, ti, to = med(chain), med(independent), med(one)
        print(f"chain        {tc*1e3:6.3f} ms  {nbytes/tc/1e9:6.1f} GB/s")
        print(f"independent  {ti*1e3:6.3f} ms  {nbytes/ti/1e9:6.1f} GB/s")
        print(f"one          {to*1e3:6.3f} ms  {one_bytes/to/1e9:6.1f} GB/s")
        print(f"barrier cost per kernel boundary: {(tc-ti)/112*1e6:.2f} us")
    L = f.layers[0]
    print("\nsingle-kernel latency by shape (one qmv per eval, includes eval floor):")
    for name, w, x in (("qkv 4096x2048", L["qkv"], x0), ("o 2048x2048", L["o"], x0),
                       ("gate_up 12288x2048", L["gu"], x0), ("down 2048x6144", L["down"], xi),
                       ("lm_head 151936x2048", head, x0)):
        b = sum(a.nbytes for a in w)
        def run(w=w, x=x):
            return [qmv(x, w) for _ in range(20)]
        def chain20(w=w, x=x):
            outs = [qmv(x, w) for _ in range(20)]
            return outs
        t = med(run) / 20
        print(f"  {name:22s} {b/1e6:7.2f} MB  {t*1e6:8.1f} us amortized over 20 independent  {b/t/1e9:6.1f} GB/s")


if __name__ == "__main__":
    main(sys.argv[1])
