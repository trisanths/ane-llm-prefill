"""Item 4: ANE prefill -> KV bridge -> MLX GPU decode.

ANEForge prefills a bucketed prompt on the Neural Engine and hands back per-layer
K and V as host numpy. Those seed an MLX decoder that runs the autoregressive
loop on the GPU. Prefill is compute-bound and the ANE is 2.5x the GPU there;
decode is bandwidth-bound and the GPU has 156 GB/s against the ANE's 58, so each
phase runs where it is strongest.

Bucketing matters: each prompt length is a separate compiled program and compiles
cost seconds, so prompts are right-padded to a fixed bucket (causal attention
ignores the trailing pads) and one compile serves every prompt in that bucket.
"""
import argparse
import json
import time
import warnings

import numpy as np
import aneforge as af

warnings.filterwarnings("ignore")

import mlx.core as mx  # noqa: E402


class MLXDecoder:
    """Qwen3-style decode step on the GPU, seeded from an ANE prefill."""

    def __init__(self, cfg, w, max_len=2048):
        self.cfg = cfg
        self.H, self.KV, self.dh = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
        self.n_rep = self.H // self.KV
        self.eps = cfg.norm_eps
        self.layers = []
        for l in w["layers"]:
            self.layers.append({k: mx.array(v) for k, v in l.items()})
        self.embed = mx.array(w["embed"])
        self.final_norm = mx.array(w["final_norm"])
        self.lm_head = mx.array(w["lm_head"])
        cos, sin = af.rope_tables(max_len, self.dh, base=cfg.rope_base)
        self.cos, self.sin = mx.array(np.asarray(cos)), mx.array(np.asarray(sin))
        mx.eval([v for l in self.layers for v in l.values()],
                self.embed, self.final_norm, self.lm_head, self.cos, self.sin)
        self.cache = None
        self.pos = 0

    def seed(self, kv_by_layer, seq):
        """Adopt the ANE's per-layer K/V. kv[li] is (K, V) shaped [KV, seq, dh]."""
        self.cache = [[mx.array(np.ascontiguousarray(k)), mx.array(np.ascontiguousarray(v))]
                      for k, v in kv_by_layer]
        mx.eval([t for kv in self.cache for t in kv])
        self.pos = seq

    def _rope1(self, t, p):
        """Rotate a single position. t is [heads, 1, dh]."""
        c, s = self.cos[p][None, None, :], self.sin[p][None, None, :]
        half = self.dh // 2
        rot = mx.concatenate([-t[..., half:], t[..., :half]], axis=-1)
        return t * c + rot * s

    def step(self, token_id):
        x = self.embed[token_id][None, :]                      # [1, dim]
        for li, L in enumerate(self.layers):
            h = mx.fast.rms_norm(x, L["attn_norm"], self.eps)
            q = (h @ L["wq"].T).reshape(1, self.H, self.dh).transpose(1, 0, 2)
            k = (h @ L["wk"].T).reshape(1, self.KV, self.dh).transpose(1, 0, 2)
            v = (h @ L["wv"].T).reshape(1, self.KV, self.dh).transpose(1, 0, 2)
            q = mx.fast.rms_norm(q, L["q_norm"], self.eps)
            k = mx.fast.rms_norm(k, L["k_norm"], self.eps)
            q, k = self._rope1(q, self.pos), self._rope1(k, self.pos)
            ck = mx.concatenate([self.cache[li][0], k], axis=1)
            cv = mx.concatenate([self.cache[li][1], v], axis=1)
            self.cache[li][0], self.cache[li][1] = ck, cv
            ke = mx.repeat(ck, self.n_rep, axis=0)
            ve = mx.repeat(cv, self.n_rep, axis=0)
            ctx = mx.fast.scaled_dot_product_attention(
                q[None], ke[None], ve[None], scale=1.0 / np.sqrt(self.dh))[0]
            x = x + ctx.transpose(1, 0, 2).reshape(1, -1) @ L["wo"].T
            h2 = mx.fast.rms_norm(x, L["mlp_norm"], self.eps)
            g = h2 @ L["wgate"].T
            x = x + (g * mx.sigmoid(g) * (h2 @ L["wup"].T)) @ L["wdown"].T
        self.pos += 1
        return mx.fast.rms_norm(x, self.final_norm, self.eps) @ self.lm_head.T

    def generate(self, first_token, n):
        out, tok = [], first_token
        for _ in range(n):
            logits = self.step(tok)
            mx.eval(logits)
            tok = int(mx.argmax(logits, axis=-1).item())
            out.append(tok)
        return out


def bucket_for(n, buckets):
    for b in buckets:
        if n <= b:
            return b
    return buckets[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--compress", default="int8")
    ap.add_argument("--prompt-len", type=int, nargs="+", default=[128, 512])
    ap.add_argument("--buckets", type=int, nargs="+", default=[512, 1024])
    ap.add_argument("--gen", type=int, default=24)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    t0 = time.perf_counter()
    m = af.load_llm(a.model, compress=a.compress)
    load_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    dec = MLXDecoder(m.cfg, m.w, max_len=max(a.buckets) + a.gen + 8)
    dec_build_s = time.perf_counter() - t0

    rng = np.random.default_rng(0)
    results = []
    for P in a.prompt_len:
        ids = rng.integers(1000, 140000, size=P).tolist()
        B = bucket_for(P, a.buckets)

        # cold: first call for this bucket pays the compile
        t0 = time.perf_counter()
        logits, kv = m._prefill_seed(ids, pad_to=B)
        ttft_cold = time.perf_counter() - t0

        warm = []
        for _ in range(a.repeats):
            t0 = time.perf_counter()
            logits, kv = m._prefill_seed(ids, pad_to=B)
            warm.append(time.perf_counter() - t0)
        ttft_warm = float(np.median(warm))

        t0 = time.perf_counter()
        dec.seed(kv, P)
        bridge_s = time.perf_counter() - t0

        first = int(np.asarray(logits).reshape(-1).argmax())
        t0 = time.perf_counter()
        toks = dec.generate(first, a.gen)
        gen_s = time.perf_counter() - t0

        kv_bytes = sum(k.nbytes + v.nbytes for k, v in kv)
        r = {"bench": "hybrid", "model": a.model, "compress": a.compress,
             "prompt_len": P, "bucket": B, "gen": a.gen,
             "ttft_cold_s": ttft_cold, "ttft_warm_s": ttft_warm,
             "bridge_s": bridge_s, "kv_MB": kv_bytes / 1e6,
             "decode_s": gen_s, "decode_tok_s": a.gen / gen_s,
             "prefill_tok_s": P / ttft_warm,
             "load_s": round(load_s, 1), "decoder_build_s": round(dec_build_s, 1),
             "sample": tok.decode(toks[:12])}
        results.append(r)
        print(json.dumps(r), flush=True)
        if a.out:
            with open(a.out, "a") as f:
                f.write(json.dumps(r) + "\n")
    return results


if __name__ == "__main__":
    main()
