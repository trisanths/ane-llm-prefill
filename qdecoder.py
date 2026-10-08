"""MLX decoder with 4-bit quantized weights, for the hybrid weight budget.

The residency finding says a hybrid holds weights twice, once baked per engine.
That is unavoidable, but the two copies need not be the same size. Prefill stays
fp16 on the ANE, where accuracy matters most and where int8 is currently broken
at 1.7B. Decode runs from 4-bit weights on the GPU, where decode is
bandwidth-bound and quantization helps throughput as well as footprint.

ANEForge's host numpy copy is freed after the prefill program is compiled: baked
weights survive it, verified in item 2.
"""
import gc

import mlx.core as mx
import numpy as np
import aneforge as af

# matrices worth quantizing; norms are tiny and stay fp16
BIG = ("wq", "wk", "wv", "wo", "wgate", "wup", "wdown")
SMALL = ("attn_norm", "mlp_norm", "q_norm", "k_norm")


class QuantDecoder:
    def __init__(self, cfg, w, max_len=2048, bits=4, group_size=32, quantize=True):
        self.cfg, self.bits, self.gs, self.q = cfg, bits, group_size, quantize
        self.H, self.KV, self.dh = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
        self.n_rep, self.eps = self.H // self.KV, cfg.norm_eps
        self.layers = []
        for L in w["layers"]:
            d = {k: mx.array(L[k]) for k in SMALL if k in L}
            for k in BIG:
                a = mx.array(L[k])
                if quantize:
                    wq, s, b = mx.quantize(a, group_size=group_size, bits=bits)
                    mx.eval(wq, s, b)
                    d[k] = (wq, s, b)
                    del a
                else:
                    d[k] = a
            self.layers.append(d)
            mx.eval([t for v in d.values() for t in (v if isinstance(v, tuple) else (v,))])
        self.embed = mx.array(w["embed"])          # lookup table, stays fp16
        self.final_norm = mx.array(w["final_norm"])
        head = w["lm_head"]
        same = head is w["embed"]
        if quantize and not same:
            hq, hs, hb = mx.quantize(mx.array(head), group_size=group_size, bits=bits)
            mx.eval(hq, hs, hb)
            self.lm_head = (hq, hs, hb)
        elif quantize and same:
            hq, hs, hb = mx.quantize(self.embed, group_size=group_size, bits=bits)
            mx.eval(hq, hs, hb)
            self.lm_head = (hq, hs, hb)
        else:
            self.lm_head = mx.array(head)
        cos, sin = af.rope_tables(max_len, self.dh, base=cfg.rope_base)
        self.cos, self.sin = mx.array(np.asarray(cos)), mx.array(np.asarray(sin))
        mx.eval(self.embed, self.final_norm, self.cos, self.sin)
        self.cache, self.pos = None, 0

    def mm(self, x, W):
        if isinstance(W, tuple):
            wq, s, b = W
            return mx.quantized_matmul(x, wq, s, b, transpose=True,
                                       group_size=self.gs, bits=self.bits)
        return x @ W.T

    def seed(self, kv_by_layer, seq):
        self.cache = [[mx.array(np.ascontiguousarray(k)), mx.array(np.ascontiguousarray(v))]
                      for k, v in kv_by_layer]
        mx.eval([t for kv in self.cache for t in kv])
        self.pos = seq

    def _rope1(self, t, p):
        c, s = self.cos[p][None, None, :], self.sin[p][None, None, :]
        h = self.dh // 2
        return t * c + mx.concatenate([-t[..., h:], t[..., :h]], axis=-1) * s

    def step(self, token_id):
        x = self.embed[token_id][None, :]
        for li, L in enumerate(self.layers):
            h = mx.fast.rms_norm(x, L["attn_norm"], self.eps)
            q = self.mm(h, L["wq"]).reshape(1, self.H, self.dh).transpose(1, 0, 2)
            k = self.mm(h, L["wk"]).reshape(1, self.KV, self.dh).transpose(1, 0, 2)
            v = self.mm(h, L["wv"]).reshape(1, self.KV, self.dh).transpose(1, 0, 2)
            q = mx.fast.rms_norm(q, L["q_norm"], self.eps)
            k = mx.fast.rms_norm(k, L["k_norm"], self.eps)
            q, k = self._rope1(q, self.pos), self._rope1(k, self.pos)
            ck = mx.concatenate([self.cache[li][0], k], axis=1)
            cv = mx.concatenate([self.cache[li][1], v], axis=1)
            self.cache[li][0], self.cache[li][1] = ck, cv
            ctx = mx.fast.scaled_dot_product_attention(
                q[None], mx.repeat(ck, self.n_rep, axis=0)[None],
                mx.repeat(cv, self.n_rep, axis=0)[None], scale=1.0 / np.sqrt(self.dh))[0]
            x = x + self.mm(ctx.transpose(1, 0, 2).reshape(1, -1), L["wo"])
            h2 = mx.fast.rms_norm(x, L["mlp_norm"], self.eps)
            g = self.mm(h2, L["wgate"])
            x = x + self.mm(g * mx.sigmoid(g) * self.mm(h2, L["wup"]), L["wdown"])
        hn = mx.fast.rms_norm(x, self.final_norm, self.eps)
        return self.mm(hn, self.lm_head)

    def generate(self, first, n):
        out, tok = [], first
        for _ in range(n):
            lg = self.step(tok)
            mx.eval(lg)
            tok = int(mx.argmax(lg, axis=-1).item())
            out.append(tok)
        return out


def free_host_weights(m):
    """Drop ANEForge's host numpy weights. Baked programs keep working."""
    n = 0
    try:
        for L in m.w["layers"]:
            for k in list(L.keys()):
                if isinstance(L[k], np.ndarray):
                    n += L[k].nbytes
                    L[k] = None
        for k in ("embed", "lm_head", "final_norm"):
            if k in m.w and isinstance(m.w[k], np.ndarray):
                n += m.w[k].nbytes
                m.w[k] = None
    except Exception:
        pass
    gc.collect()
    return n / 1e9
