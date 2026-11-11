"""Time of one target forward over L new tokens (the verify step of
speculative decoding) at a 256-token context, for L = 1 to 16. The cache is
trimmed back after each call so every call sees the same context. L values
are interleaved across rounds and the median is reported."""
import sys
import time

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

LS = (1, 2, 3, 4, 5, 6, 7, 8)


def main(paths, ctx=256, rounds=15):
    import spec_patch
    patched = "--patch" in paths
    paths = [p for p in paths if p != "--patch"]
    if patched:
        spec_patch.patch()
    for path in paths:
        model, _ = load(path)
        c = make_prompt_cache(model)
        mx.eval(model(mx.array([[1000 + i for i in range(ctx)]]), cache=c))
        ts = {L: [] for L in LS}
        for r in range(rounds + 3):
            for L in LS[r % len(LS):] + LS[:r % len(LS)]:
                x = mx.array([[42 + i for i in range(L)]])
                a = time.perf_counter()
                mx.eval(model(x, cache=c))
                dt = time.perf_counter() - a
                for cc in c:
                    cc.trim(L)
                if r >= 3:
                    ts[L].append(dt)
        med = {L: sorted(v)[len(v) // 2] * 1e3 for L, v in ts.items()}
        name = path.rstrip("/").split("--")[-1].split("/")[0]
        name += " (qmv_h2)" if patched else " (mlx)"
        print(f"{name}: " + "  ".join(f"L{L} {med[L]:.2f}" for L in LS) + " ms")
        print(" " * (len(name) + 2) + "  ".join(f"x{med[L]/med[1]:.2f}" for L in LS))
        del model, c
        mx.clear_cache()


if __name__ == "__main__":
    main(sys.argv[1:])
