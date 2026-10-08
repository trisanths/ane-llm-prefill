"""Do the ANE and the GPU actually run for free alongside each other?

The first version of this test let a worker thread issue ANE prefills in a tight
loop while the GPU decoded, and reported 99.8 percent decode retention on
Qwen3-0.6B. That number was measured over a 0.66 s window in which the ANE
completed 8 prefills. Repeating it on Qwen3-1.7B over a longer window collapsed
to between 1 and 26 percent retention, with the worker completing hundreds of
prefills. The engines do contend; the short window hid it.

This version controls the ANE's duty cycle instead of saturating it, so decode
throughput can be read as a function of how hard the ANE is being driven. Duty 0
is decode alone; duty 1.0 is the saturating case the first test accidentally
measured.
"""
import argparse
import json
import threading
import time
import warnings

import numpy as np
import aneforge as af

warnings.filterwarnings("ignore")
import diskguard  # noqa: E402
from hybrid import MLXDecoder  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--compress", default="")
    ap.add_argument("--chunk-mb", type=float, default=150)
    ap.add_argument("--prompt-len", type=int, default=512)
    ap.add_argument("--bucket", type=int, default=512)
    ap.add_argument("--gen", type=int, default=64)
    ap.add_argument("--duties", type=float, nargs="+", default=[0.0, 0.25, 0.5, 1.0])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    diskguard.guard(limit_gb=5.0, min_free_gb=8.0)
    m = af.load_llm(a.model, compress=(a.compress or None))
    if a.chunk_mb:
        m._chunk_bytes = int(a.chunk_mb * 1e6)
        m._pre = None
    dec = MLXDecoder(m.cfg, m.w, max_len=a.bucket + a.gen + 8)

    rng = np.random.default_rng(0)
    idsA = rng.integers(1000, 140000, size=a.prompt_len).tolist()
    idsB = rng.integers(1000, 140000, size=a.prompt_len).tolist()

    logits, kv = m._prefill_seed(idsA, pad_to=a.bucket)
    first = int(np.asarray(logits).reshape(-1).argmax())

    # one clean prefill timing, to size the duty sleep
    t0 = time.perf_counter()
    m._prefill_seed(idsB, pad_to=a.bucket)
    prefill_s = time.perf_counter() - t0

    # Warm MLX before any timed run. The first generate() compiles Metal kernels,
    # which previously made the duty=0 baseline look SLOWER than the contended
    # runs and inverted the whole result.
    dec.seed(kv, a.prompt_len)
    dec.generate(first, min(a.gen, 16))
    dec.seed(kv, a.prompt_len)
    dec.generate(first, min(a.gen, 16))

    for duty in a.duties:
        dec.seed(kv, a.prompt_len)
        stop = threading.Event()
        count = {"n": 0}

        def worker(d=duty):
            # run prefill, then idle so the ANE is busy a fraction d of the time
            idle = prefill_s * (1.0 - d) / max(d, 1e-9)
            while not stop.is_set():
                m._prefill_seed(idsB, pad_to=a.bucket)
                count["n"] += 1
                if idle > 0:
                    stop.wait(idle)

        th = None
        if duty > 0:
            th = threading.Thread(target=worker, daemon=True)
            th.start()
            time.sleep(0.05)
        t0 = time.perf_counter()
        dec.generate(first, a.gen)
        el = time.perf_counter() - t0
        stop.set()
        if th:
            th.join(timeout=30)

        r = {"bench": "contend", "model": a.model, "compress": a.compress or "fp16",
             "prompt_len": a.prompt_len, "gen": a.gen, "ane_duty": duty,
             "decode_tok_s": a.gen / el, "decode_s": el,
             "ane_prefills": count["n"],
             "ane_prefill_s": prefill_s,
             "prompt_tok_prefilled": count["n"] * a.prompt_len,
             "prefill_tok_s_effective": count["n"] * a.prompt_len / el}
        diskguard.stamp(r)
        print(json.dumps(r), flush=True)
        if a.out:
            with open(a.out, "a") as f:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
