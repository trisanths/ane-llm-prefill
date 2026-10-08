"""Item 2: layer-segmented prefill, and the cost of chaining segments.

ANEForge already groups layers into programs sized by `_chunk_bytes`, chaining
the hidden state from one program to the next through the host. The default of
1600 MB puts all 28 layers of Qwen3-1.7B in a single program, which does not
compile here. Lowering it forces more, smaller programs.

This measures what that costs: each extra segment adds a dispatch plus a host
round-trip of the hidden state, so TTFT should rise with segment count while the
peak program size falls.
"""
import argparse
import json
import time
import warnings

import numpy as np
import aneforge as af

warnings.filterwarnings("ignore")
import diskguard  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--compress", default=None)
    ap.add_argument("--chunk-mb", type=float, nargs="+", default=[800, 400, 200, 100])
    ap.add_argument("--prompt-len", type=int, default=512)
    ap.add_argument("--bucket", type=int, default=512)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--cache-cap-gb", type=float, default=5.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    rng = np.random.default_rng(0)
    ids = rng.integers(1000, 140000, size=a.prompt_len).tolist()

    for mb in a.chunk_mb:
        diskguard.guard(limit_gb=a.cache_cap_gb, min_free_gb=8.0)
        rec = {"bench": "segment", "model": a.model, "compress": a.compress,
               "chunk_mb": mb, "prompt_len": a.prompt_len, "bucket": a.bucket}
        try:
            m = af.load_llm(a.model, compress=a.compress)
            m._chunk_bytes = int(mb * 1e6)
            m._pre = None
            groups = [len(g) for g in m._layer_chunks()]
            rec["groups"] = groups
            rec["n_segments"] = len(groups)

            t0 = time.perf_counter()
            logits, kv = m._prefill_seed(ids, pad_to=a.bucket)
            rec["ttft_cold_s"] = time.perf_counter() - t0

            ts = []
            for _ in range(a.repeats):
                t0 = time.perf_counter()
                m._prefill_seed(ids, pad_to=a.bucket)
                ts.append(time.perf_counter() - t0)
            rec["ttft_warm_s"] = float(np.median(ts))
            rec["ttft_sd_s"] = float(np.std(ts))
            rec["prefill_tok_s"] = a.prompt_len / rec["ttft_warm_s"]
            rec["kv_MB"] = sum(k.nbytes + v.nbytes for k, v in kv) / 1e6
            rec["top1"] = int(np.asarray(logits).reshape(-1).argmax())
            m.release()
            del m
        except Exception as e:  # noqa: BLE001
            rec["error"] = f"{type(e).__name__}: {e}"[:300]
        diskguard.stamp(rec)
        print(json.dumps(rec), flush=True)
        if a.out:
            with open(a.out, "a") as f:
                f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
