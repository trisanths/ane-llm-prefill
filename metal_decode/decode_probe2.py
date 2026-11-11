"""Baselines next to decode_probe.py: mlx_lm's own generate loop, a pipelined
loop (async_eval of token t+1 before waiting on t), and the same under
mx.compile."""
import sys
import time
from functools import partial

import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.models.cache import make_prompt_cache

from decode_probe import prefill


def pipelined(step, model, n, prompt_len):
    cache = prefill(model, prompt_len)
    tok = mx.array([[42]])
    nxt = step(tok, cache)
    mx.async_eval(nxt)
    for _ in range(8):
        cur = nxt
        nxt = step(cur, cache)
        mx.async_eval(nxt)
        cur.item()
    t = time.perf_counter()
    for _ in range(n):
        cur = nxt
        nxt = step(cur, cache)
        mx.async_eval(nxt)
        cur.item()
    return (time.perf_counter() - t) / n


def main(path, n=128, prompt_len=256):
    model, tok = load(path)
    prompt = tok.encode("word " * prompt_len)[:prompt_len]
    last = None
    for r in stream_generate(model, tok, prompt, max_tokens=n):
        last = r
    for r in stream_generate(model, tok, prompt, max_tokens=n):
        last = r
    print(f"E  mlx_lm stream_generate   {1e3/last.generation_tps:7.2f} ms  {last.generation_tps:6.1f} tok/s  (prompt {last.prompt_tokens}, {last.generation_tokens} gen, peak {last.peak_memory:.2f} GB)")
    step = lambda t, c: mx.argmax(model(t.reshape(1, 1), cache=c)[:, -1], -1)
    tp = pipelined(step, model, n, prompt_len)
    print(f"P  pipelined async_eval     {tp*1e3:7.2f} ms  {1/tp:6.1f} tok/s")
    cache_holder = {}

    def stepc(t, c):
        return step(t, c)
    cstep = mx.compile(stepc)
    try:
        tc = pipelined(cstep, model, n, prompt_len)
        print(f"C  pipelined + mx.compile   {tc*1e3:7.2f} ms  {1/tc:6.1f} tok/s")
    except Exception as e:
        print("C  mx.compile failed:", type(e).__name__, str(e)[:200])


if __name__ == "__main__":
    main(sys.argv[1])
