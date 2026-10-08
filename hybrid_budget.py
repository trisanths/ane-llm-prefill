"""Item 1: hybrid at fp16 ANE prefill + int4 MLX decode, with the host copy freed."""
import argparse
import gc
import json
import time
import warnings

import numpy as np
import aneforge as af

warnings.filterwarnings("ignore")
import diskguard  # noqa: E402
import memguard  # noqa: E402
from qdecoder import QuantDecoder, free_host_weights  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--chunk-mb", type=float, default=150)
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--group-size", type=int, default=32)
    ap.add_argument("--prompt-len", type=int, default=512)
    ap.add_argument("--bucket", type=int, default=512)
    ap.add_argument("--gen", type=int, default=64)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--free-host", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    diskguard.guard(limit_gb=5.0, min_free_gb=8.0)
    stages = {"start": memguard.stamp()}

    m = af.load_llm(a.model, compress=None)
    if a.chunk_mb:
        m._chunk_bytes = int(a.chunk_mb * 1e6)
        m._pre = None
    stages["after_load"] = memguard.stamp()

    rng = np.random.default_rng(0)
    ids = rng.integers(1000, 140000, size=a.prompt_len).tolist()

    logits, kv = m._prefill_seed(ids, pad_to=a.bucket)      # compiles + bakes
    stages["after_ane_compile"] = memguard.stamp()

    dec = QuantDecoder(m.cfg, m.w, max_len=a.bucket + a.gen + 8,
                       bits=a.bits, group_size=a.group_size, quantize=a.bits > 0)
    stages["after_mlx_build"] = memguard.stamp()

    freed = 0.0
    if a.free_host:
        freed = free_host_weights(m)
        gc.collect()
        stages["after_free_host"] = memguard.stamp()

    ane_t = []
    for _ in range(a.repeats):
        t = time.perf_counter()
        m._prefill_seed(ids, pad_to=a.bucket)
        ane_t.append(time.perf_counter() - t)
    first = int(np.asarray(logits).reshape(-1).argmax())

    dec.seed(kv, a.prompt_len)
    dec.generate(first, 8)                                   # warm Metal kernels
    dec.seed(kv, a.prompt_len)
    dec.generate(first, 8)

    dec_t = []
    for _ in range(3):
        dec.seed(kv, a.prompt_len)
        t = time.perf_counter()
        dec.generate(first, a.gen)
        dec_t.append(time.perf_counter() - t)
    stages["after_runs"] = memguard.stamp()

    r = {"bench": "hybrid_budget", "model": a.model, "bits": a.bits,
         "group_size": a.group_size, "free_host": a.free_host,
         "freed_gb": round(freed, 2), "chunk_mb": a.chunk_mb,
         "n_segments": len(m._layer_chunks()),
         "prompt_len": a.prompt_len, "gen": a.gen,
         "ane_prefill_s": float(np.median(ane_t)),
         "ane_prefill_tok_s": a.prompt_len / float(np.median(ane_t)),
         "decode_tok_s": a.gen / float(np.median(dec_t)),
         "stages": stages}
    diskguard.stamp(r)
    memguard.stamp(r)
    print(json.dumps(r), flush=True)
    if a.out:
        with open(a.out, "a") as f:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
