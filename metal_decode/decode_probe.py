"""Split one batch-1 MLX decode token into its costs.

A: normal decode, one eval per token (what mlx_lm.generate does).
B: the same forward K times per eval on independent caches, so the CPU encodes
   while the GPU runs and the sync per token is amortized. Same kernels as A.
C: A with the forward under mx.compile, which fuses elementwise kernels.
S: the stream floor: one quantized matmul over the same number of weight bytes.
D: per-dispatch cost: chains of tiny dependent kernels in one eval.
Results print as ms per token and as achieved GB/s over the weight bytes.
"""
import json
import sys
import time

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache


def weight_bytes(model):
    tot = 0
    for _, m in model.named_modules():
        if isinstance(m, (nn.QuantizedLinear, nn.QuantizedEmbedding)):
            tot += m.weight.nbytes + m.scales.nbytes + m.biases.nbytes
            if isinstance(m, nn.QuantizedEmbedding):
                tot -= m.weight.nbytes + m.scales.nbytes + m.biases.nbytes
    return tot


def lm_head_bytes(model):
    e = model.model.embed_tokens
    return e.weight.nbytes + e.scales.nbytes + e.biases.nbytes


def prefill(model, prompt_len):
    cache = make_prompt_cache(model)
    x = mx.array([[1000 + i for i in range(prompt_len)]])
    mx.eval(model(x, cache=cache))
    return cache


def run_a(fwd, model, n, prompt_len):
    cache = prefill(model, prompt_len)
    tok = mx.array([[42]])
    for _ in range(8):
        tok = mx.argmax(fwd(tok, cache)[:, -1:], -1)
        mx.eval(tok)
    t = time.perf_counter()
    for _ in range(n):
        tok = mx.argmax(fwd(tok, cache)[:, -1:], -1)
        mx.eval(tok)
    return (time.perf_counter() - t) / n


def run_b(model, n, k, prompt_len):
    caches = [prefill(model, prompt_len) for _ in range(k)]
    toks = [mx.array([[42]]) for _ in range(k)]
    for _ in range(4):
        toks = [mx.argmax(model(t, cache=c)[:, -1:], -1) for t, c in zip(toks, caches)]
        mx.eval(toks)
    t = time.perf_counter()
    for _ in range(n // k):
        toks = [mx.argmax(model(t, cache=c)[:, -1:], -1) for t, c in zip(toks, caches)]
        mx.eval(toks)
    return (time.perf_counter() - t) / ((n // k) * k)


def run_s(nbytes, n):
    k = 4096
    rows = int(nbytes / (k * 4.5 / 8))
    w = mx.random.normal((rows, k)).astype(mx.float16)
    wq, s, b = mx.quantize(w, group_size=64, bits=4)
    del w
    x = mx.random.normal((1, k)).astype(mx.float16)
    mx.eval(wq, s, b, x)
    real = wq.nbytes + s.nbytes + b.nbytes
    for _ in range(5):
        mx.eval(mx.quantized_matmul(x, wq, s, b, group_size=64, bits=4))
    t = time.perf_counter()
    for _ in range(n):
        mx.eval(mx.quantized_matmul(x, wq, s, b, group_size=64, bits=4))
    return (time.perf_counter() - t) / n, real


def run_d(chain, reps):
    x = mx.zeros((1, 2048), dtype=mx.float16)
    mx.eval(x)
    out = {}
    for c in (1, chain):
        def f(y):
            for _ in range(c):
                y = y * 1.0001 + 0.5
            return y
        for _ in range(5):
            mx.eval(f(x))
        t = time.perf_counter()
        for _ in range(reps):
            mx.eval(f(x))
        out[c] = (time.perf_counter() - t) / reps
    return (out[chain] - out[1]) / (chain - 1), out[1]


def main(path, n=128, prompt_len=256):
    model, _ = load(path)
    model.eval()
    wb = weight_bytes(model)
    hb = lm_head_bytes(model)
    tok_bytes = wb + hb
    gbs = lambda s: tok_bytes / s / 1e9
    res = {"model": path.rstrip("/").split("/")[-1], "bytes_per_token": tok_bytes, "prompt_len": prompt_len}
    plain = lambda t, c: model(t, cache=c)
    ta = run_a(plain, model, n, prompt_len)
    res["A_ms"] = ta * 1e3
    tb = run_b(model, n, 8, prompt_len)
    res["B_ms"] = tb * 1e3
    ts, sbytes = run_s(tok_bytes, 50)
    res["S_ms"] = ts * 1e3
    res["S_GBs"] = sbytes / ts / 1e9
    per_dispatch, one = run_d(512, 50)
    res["dispatch_us"] = per_dispatch * 1e6
    res["eval_floor_us"] = one * 1e6
    print(f"weights read per token: {tok_bytes/1e6:.0f} MB (lm_head {hb/1e6:.0f} MB)")
    print(f"A  decode, eval per token   {ta*1e3:7.2f} ms  {1/ta:6.1f} tok/s  {gbs(ta):6.1f} GB/s")
    print(f"B  8 streams per eval       {tb*1e3:7.2f} ms  {1/tb:6.1f} tok/s  {gbs(tb):6.1f} GB/s")
    print(f"S  one qmv, same bytes      {ts*1e3:7.2f} ms  {1/ts:6.1f} tok/s  {sbytes/ts/1e9:6.1f} GB/s")
    print(f"D  dependent tiny kernel    {per_dispatch*1e6:7.2f} us each; empty eval {one*1e6:.0f} us")
    print(json.dumps(res))


if __name__ == "__main__":
    main(sys.argv[1])
