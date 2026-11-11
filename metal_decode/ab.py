"""Interleaved A/B of decode modes: many short pipelined runs, modes in
rotating order, median and spread per mode. Use this, not single runs: the
desktop GPU is shared and one run swings 10 percent or more."""
import re
import collections
import sys

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

from fused_qwen3 import Fused, stock_logits, time_loop

SCRATCH = sys.argv[2] if len(sys.argv) > 2 else "/tmp/m.dot"
NOCOPY = {"Reshape", "Transpose", "Broadcast", "Slice", "Squeeze", "ExpandDims", "AsStrided"}


def main(path, rounds=12, n=64, prompt_len=256):
    model, _ = load(path)
    f = Fused(model)
    filler = mx.array([[1000 + i for i in range(prompt_len)]])
    sc = make_prompt_cache(model)
    mx.eval(model(filler, cache=sc))
    t = mx.array([42], dtype=mx.int32)
    for mode in ("stock", "merge", "fused", "skeleton"):
        sc = make_prompt_cache(model)
        mx.eval(model(filler, cache=sc))
        out = stock_logits(model, t, sc) if mode == "stock" else f.step(t, sc, mode)
        extra = [x for cc in sc for x in (cc.keys, cc.values)]
        mx.export_to_dot(SCRATCH, out, *extra)
        mx.eval(out, *extra)
        cnt = collections.Counter(re.findall(r'label ="([A-Za-z_]+)', open(SCRATCH).read()))
        print(f"{mode:9s} kernels {sum(v for k, v in cnt.items() if k not in NOCOPY)}")
    sc = None

    def mk(mode):
        s = make_prompt_cache(model)
        mx.eval(model(filler, cache=s))
        if mode == "stock":
            return lambda tk, i: mx.argmax(stock_logits(model, tk, s)).reshape(1).astype(mx.int32)
        return lambda tk, i: mx.argmax(f.step(tk, s, mode)).reshape(1).astype(mx.int32)

    modes = ["stock", "merge", "fused", "skeleton"]
    res = {m: [] for m in modes}
    for r in range(rounds):
        order = modes[r % 4:] + modes[:r % 4]
        for m in order:
            res[m].append(time_loop(mk(m), n) * 1e3)
    base = sorted(res["stock"])[rounds // 2]
    print(f"\n{'mode':9s}{'median ms':>10s}{'tok/s':>8s}{'vs stock':>9s}{'p25':>8s}{'p75':>8s}")
    for m in modes:
        v = sorted(res[m])
        med = v[rounds // 2]
        print(f"{m:9s}{med:10.3f}{1e3/med:8.1f}{base/med:9.3f}{v[rounds//4]:8.3f}{v[3*rounds//4]:8.3f}")
    # paired ratio per round, robust to drift
    for m in modes[1:]:
        pr = sorted(a / b for a, b in zip(res["stock"], res[m]))
        print(f"paired stock/{m}: median {pr[rounds//2]:.3f}  range {pr[0]:.3f}-{pr[-1]:.3f}")


if __name__ == "__main__":
    main(sys.argv[1])
