"""Item 4 part 2: GPU-only baseline, and ANE prefill concurrent with GPU decode."""
import argparse
import json
import threading
import time
import warnings

import numpy as np
import aneforge as af

warnings.filterwarnings("ignore")
import mlx.core as mx  # noqa: E402

from hybrid import MLXDecoder  # noqa: E402


def mlx_prefill(dec, ids):
    """GPU-only prefill: run the whole prompt through MLX, filling the KV cache.

    Same math as the decode step but over S positions at once, so it is the fair
    baseline for time-to-first-token without the ANE.
    """
    S = len(ids)
    cfg = dec.cfg
    H, KV, dh, n_rep, eps = dec.H, dec.KV, dec.dh, dec.n_rep, dec.eps
    x = dec.embed[mx.array(ids)]                      # [S, dim]
    cos = dec.cos[:S][None]
    sin = dec.sin[:S][None]
    mask = mx.array(np.triu(np.full((S, S), -np.inf, np.float32), k=1).astype(np.float16))
    half = dh // 2
    cache = []
    for L in dec.layers:
        h = mx.fast.rms_norm(x, L["attn_norm"], eps)
        q = (h @ L["wq"].T).reshape(S, H, dh).transpose(1, 0, 2)
        k = (h @ L["wk"].T).reshape(S, KV, dh).transpose(1, 0, 2)
        v = (h @ L["wv"].T).reshape(S, KV, dh).transpose(1, 0, 2)
        q = mx.fast.rms_norm(q, L["q_norm"], eps)
        k = mx.fast.rms_norm(k, L["k_norm"], eps)
        q = q * cos + mx.concatenate([-q[..., half:], q[..., :half]], axis=-1) * sin
        k = k * cos + mx.concatenate([-k[..., half:], k[..., :half]], axis=-1) * sin
        cache.append([k, v])
        ctx = mx.fast.scaled_dot_product_attention(
            q[None], mx.repeat(k, n_rep, axis=0)[None], mx.repeat(v, n_rep, axis=0)[None],
            scale=1.0 / np.sqrt(dh), mask=mask)[0]
        x = x + ctx.transpose(1, 0, 2).reshape(S, -1) @ L["wo"].T
        h2 = mx.fast.rms_norm(x, L["mlp_norm"], eps)
        g = h2 @ L["wgate"].T
        x = x + (g * mx.sigmoid(g) * (h2 @ L["wup"].T)) @ L["wdown"].T
    logits = mx.fast.rms_norm(x[-1:], dec.final_norm, eps) @ dec.lm_head.T
    mx.eval(logits, [t for kv in cache for t in kv])
    return logits, cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--compress", default="int8")
    ap.add_argument("--prompt-len", type=int, default=512)
    ap.add_argument("--bucket", type=int, default=512)
    ap.add_argument("--gen", type=int, default=48)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--chunk-mb", type=float, default=0, help="0 = ANEForge default")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    import diskguard
    diskguard.guard(limit_gb=5.0, min_free_gb=8.0)
    m = af.load_llm(a.model, compress=a.compress)
    if a.chunk_mb:
        m._chunk_bytes = int(a.chunk_mb * 1e6)
        m._pre = None
    dec = MLXDecoder(m.cfg, m.w, max_len=a.bucket + a.gen + 8)
    rng = np.random.default_rng(0)
    ids = rng.integers(1000, 140000, size=a.prompt_len).tolist()

    # warm both paths
    m._prefill_seed(ids, pad_to=a.bucket)
    mlx_prefill(dec, ids)

    ane_t = []
    for _ in range(a.repeats):
        t = time.perf_counter()
        logits, kv = m._prefill_seed(ids, pad_to=a.bucket)
        ane_t.append(time.perf_counter() - t)
    gpu_t = []
    for _ in range(a.repeats):
        t = time.perf_counter()
        lg2, _ = mlx_prefill(dec, ids)
        gpu_t.append(time.perf_counter() - t)
    ane_ttft, gpu_ttft = float(np.median(ane_t)), float(np.median(gpu_t))

    # decode alone
    first = int(np.asarray(logits).reshape(-1).argmax())
    dec.seed(kv, a.prompt_len)
    t = time.perf_counter()
    dec.generate(first, a.gen)
    solo_s = time.perf_counter() - t

    # decode while the ANE prefills a second request
    stop = threading.Event()
    counter = {"n": 0}

    def ane_worker():
        idsB = rng.integers(1000, 140000, size=a.prompt_len).tolist()
        while not stop.is_set():
            m._prefill_seed(idsB, pad_to=a.bucket)
            counter["n"] += 1

    dec.seed(kv, a.prompt_len)
    th = threading.Thread(target=ane_worker, daemon=True)
    th.start()
    time.sleep(0.2)
    t = time.perf_counter()
    dec.generate(first, a.gen)
    conc_s = time.perf_counter() - t
    stop.set()
    th.join(timeout=5)

    r = {"bench": "hybrid_concurrent", "model": a.model, "compress": a.compress,
         "chunk_mb": a.chunk_mb, "n_segments": len(m._layer_chunks()),
         "prompt_len": a.prompt_len, "bucket": a.bucket, "gen": a.gen,
         "ane_prefill_s": ane_ttft, "gpu_prefill_s": gpu_ttft,
         "ane_speedup_on_prefill": gpu_ttft / ane_ttft,
         "ane_prefill_tok_s": a.prompt_len / ane_ttft,
         "gpu_prefill_tok_s": a.prompt_len / gpu_ttft,
         "decode_solo_tok_s": a.gen / solo_s,
         "decode_concurrent_tok_s": a.gen / conc_s,
         "decode_retention": solo_s / conc_s,
         "ane_prefills_completed_during_decode": counter["n"]}
    diskguard.stamp(r)
    print(json.dumps(r), flush=True)
    if a.out:
        with open(a.out, "a") as f:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
