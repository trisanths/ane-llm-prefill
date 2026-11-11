"""Batch-1 decode for Qwen3 (MLX affine 4-bit) with fewer kernels per token.

Modes, each a full decode step over the weights of a loaded mlx_lm model:
  merge     q/k/v as one matmul and gate/up as one matmul; MLX ops otherwise.
  fused     merge, plus two custom Metal kernels: residual add + RMSNorm, and
            q/k RMSNorm + RoPE for all heads in one launch.
  skeleton  merged matmuls and attention only: no norms, RoPE or residual
            adds. Wrong output. Timing only, to bound what fusion can win.

Run as a script to time every mode against stock mlx_lm in one process and to
check that greedy tokens match.
"""
import sys
import time

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.activations import swiglu
from mlx_lm.models.cache import make_prompt_cache

GS, BITS = 64, 4

ADD_NORM_SRC = """
uint t = thread_position_in_threadgroup.x;
uint lane = thread_index_in_simdgroup;
uint sg = simdgroup_index_in_threadgroup;
threadgroup float part[32];
float ss = 0.0f;
T hv[PER];
for (uint j = 0; j < PER; ++j) {
  uint i = t + j * NT;
  T v = x[i] + r[i];
  hv[j] = v;
  h[i] = v;
  float f = float(v);
  ss += f * f;
}
ss = simd_sum(ss);
if (lane == 0) part[sg] = ss;
threadgroup_barrier(mem_flags::mem_threadgroup);
if (sg == 0) {
  float p = lane < NT / 32 ? part[lane] : 0.0f;
  p = simd_sum(p);
  if (lane == 0) part[0] = p;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
float inv = metal::precise::rsqrt(part[0] / float(D) + EPS);
for (uint j = 0; j < PER; ++j) {
  uint i = t + j * NT;
  n[i] = w[i] * T(float(hv[j]) * inv);
}
"""

QK_ROPE_SRC = """
uint hd = threadgroup_position_in_grid.x;
uint i = thread_position_in_threadgroup.x;
uint lane = thread_index_in_simdgroup;
uint sg = simdgroup_index_in_threadgroup;
threadgroup float part[2];
const device T* src = qkv + hd * HD;
const device T* w = hd < NQ ? qw : kw;
float a = float(src[i]);
float b = float(src[i + HALF]);
float ss = simd_sum(a * a + b * b);
if (lane == 0) part[sg] = ss;
threadgroup_barrier(mem_flags::mem_threadgroup);
float inv = metal::precise::rsqrt((part[0] + part[1]) / float(HD) + EPS);
float na = float(w[i] * T(a * inv));
float nb = float(w[i + HALF] * T(b * inv));
float theta = float(pos[0]) * metal::exp2(-float(i) * LOG2BASE / float(HALF));
float c = metal::precise::cos(theta);
float s = metal::precise::sin(theta);
T oa = T(na * c - nb * s);
T ob = T(na * s + nb * c);
if (hd < NQ) {
  q[hd * HD + i] = oa;
  q[hd * HD + i + HALF] = ob;
} else {
  uint kh = hd - NQ;
  k[kh * HD + i] = oa;
  k[kh * HD + i + HALF] = ob;
}
"""


class Fused:
    def __init__(self, model):
        a = model.args
        self.D, self.H, self.KV, self.HD = a.hidden_size, a.num_attention_heads, a.num_key_value_heads, a.head_dim
        self.I = a.intermediate_size
        self.eps = a.rms_norm_eps
        self.scale = self.HD ** -0.5
        self.base = a.rope_theta
        self.emb = model.model.embed_tokens
        self.final_w = model.model.norm.weight
        self.dtype = self.final_w.dtype
        self.layers = []
        for L in model.model.layers:
            at, mlp = L.self_attn, L.mlp
            cat = lambda *ms: tuple(mx.concatenate([getattr(m, f) for m in ms], 0) for f in ("weight", "scales", "biases"))
            self.layers.append(dict(
                qkv=cat(at.q_proj, at.k_proj, at.v_proj),
                o=(at.o_proj.weight, at.o_proj.scales, at.o_proj.biases),
                gu=cat(mlp.gate_proj, mlp.up_proj),
                down=(mlp.down_proj.weight, mlp.down_proj.scales, mlp.down_proj.biases),
                in_w=L.input_layernorm.weight, post_w=L.post_attention_layernorm.weight,
                qw=at.q_norm.weight, kw=at.k_norm.weight,
            ))
        mx.eval([list(l.values()) for l in self.layers])
        nt = 256
        self.add_norm_k = mx.fast.metal_kernel(
            name="add_rmsnorm", input_names=["x", "r", "w"], output_names=["h", "n"],
            source=ADD_NORM_SRC.replace("EPS", f"{self.eps:.10e}f"))
        self.add_norm_cfg = dict(template=[("T", self.dtype), ("D", self.D), ("NT", nt), ("PER", self.D // nt)],
                                 grid=(nt, 1, 1), threadgroup=(nt, 1, 1),
                                 output_shapes=[(self.D,), (self.D,)], output_dtypes=[self.dtype, self.dtype])
        import math
        self.qk_k = mx.fast.metal_kernel(
            name="qk_norm_rope", input_names=["qkv", "qw", "kw", "pos"], output_names=["q", "k"],
            source=QK_ROPE_SRC.replace("EPS", f"{self.eps:.10e}f").replace("LOG2BASE", f"{math.log2(self.base):.10f}f"))
        self.qk_cfg = dict(template=[("T", self.dtype), ("HD", self.HD), ("HALF", self.HD // 2), ("NQ", self.H)],
                           grid=((self.H + self.KV) * (self.HD // 2), 1, 1), threadgroup=(self.HD // 2, 1, 1),
                           output_shapes=[(self.H * self.HD,), (self.KV * self.HD,)],
                           output_dtypes=[self.dtype, self.dtype])

    def qmv(self, x, wsb):
        w, s, b = wsb
        return mx.quantized_matmul(x, w, s, b, transpose=True, group_size=GS, bits=BITS)

    def add_norm(self, x, r, w):
        h, n = self.add_norm_k(inputs=[x, r, w], **self.add_norm_cfg)
        return h, n

    def step(self, tok, cache, mode):
        """One decode token. cache is a list of mlx_lm KVCache objects, so KV
        handling is identical to stock and only the kernels differ."""
        D, H, KV, HD = self.D, self.H, self.KV, self.HD
        pos = cache[0].offset
        x = self.emb(tok.reshape(1)).reshape(D)
        posa = mx.array([pos], dtype=mx.int32)
        r = None
        for li, (L, kvc) in enumerate(zip(self.layers, cache)):
            if mode == "skeleton":
                n = x if r is None else r
            elif r is None:
                n = mx.fast.rms_norm(x, L["in_w"], self.eps)
            elif mode == "fused":
                x, n = self.add_norm(x, r, L["in_w"])
            else:
                x = x + r
                n = mx.fast.rms_norm(x, L["in_w"], self.eps)
            qkv = self.qmv(n, L["qkv"])
            if mode == "fused":
                q, k = self.qk_k(inputs=[qkv, L["qw"], L["kw"], posa], **self.qk_cfg)
                q = q.reshape(1, H, 1, HD)
                k = k.reshape(1, KV, 1, HD)
            elif mode == "skeleton":
                q = qkv[: H * HD].reshape(1, H, 1, HD)
                k = qkv[H * HD:(H + KV) * HD].reshape(1, KV, 1, HD)
            else:
                q = mx.fast.rms_norm(qkv[: H * HD].reshape(1, H, 1, HD), L["qw"], self.eps)
                k = mx.fast.rms_norm(qkv[H * HD:(H + KV) * HD].reshape(1, KV, 1, HD), L["kw"], self.eps)
                q = mx.fast.rope(q, HD, traditional=False, base=self.base, scale=1.0, offset=pos)
                k = mx.fast.rope(k, HD, traditional=False, base=self.base, scale=1.0, offset=pos)
            v = qkv[(H + KV) * HD:].reshape(1, KV, 1, HD)
            keys, values = kvc.update_and_fetch(k, v)
            o = mx.fast.scaled_dot_product_attention(q, keys, values, scale=self.scale)
            r = self.qmv(o.reshape(H * HD), L["o"])
            if mode == "skeleton":
                n = r
            elif mode == "fused":
                x, n = self.add_norm(x, r, L["post_w"])
            else:
                x = x + r
                n = mx.fast.rms_norm(x, L["post_w"], self.eps)
            gu = self.qmv(n, L["gu"])
            if mode == "skeleton":
                a = gu[: self.I]
            else:
                a = swiglu(gu[: self.I], gu[self.I:])
            r = self.qmv(a, L["down"])
        if mode == "fused":
            _, n = self.add_norm(x, r, self.final_w)
        elif mode == "skeleton":
            n = r
        else:
            n = mx.fast.rms_norm(x + r, self.final_w, self.eps)
        logits = self.emb.as_linear(n.reshape(1, 1, D)).reshape(-1)
        return logits


def stock_logits(model, tok, cache):
    return model(tok.reshape(1, 1), cache=cache)[0, -1]


def time_loop(stepfn, n):
    tok = mx.array([42], dtype=mx.int32)
    nxt = stepfn(tok, 0)
    mx.async_eval(nxt)
    for i in range(1, 9):
        cur = nxt
        nxt = stepfn(cur, i)
        mx.async_eval(nxt)
        cur.item()
    t = time.perf_counter()
    for i in range(9, 9 + n):
        cur = nxt
        nxt = stepfn(cur, i)
        mx.async_eval(nxt)
        cur.item()
    return (time.perf_counter() - t) / n


def main(path, n=128, prompt_len=256, reps=3):
    model, tok = load(path)
    f = Fused(model)
    prompt = tok.encode("The history of the printing press began when")
    ptoks = mx.array([prompt])

    def stock_prefill():
        c = make_prompt_cache(model)
        lg = model(ptoks, cache=c)[0, -1]
        mx.eval(lg)
        return c, lg

    # correctness: greedy tokens from the same prefill
    ref_c, lg = stock_prefill()
    first = mx.argmax(lg).reshape(1)
    seqs, firstdiff = {}, {}
    for mode in ("stock", "merge", "fused"):
        if mode == "stock":
            c, _ = stock_prefill()
        else:
            c, _ = stock_prefill()
        t = first
        out = []
        for i in range(64):
            if mode == "stock":
                logits = stock_logits(model, t, c)
            else:
                logits = f.step(t, c, mode)
            if i == 0:
                firstdiff[mode] = logits
            t = mx.argmax(logits).reshape(1).astype(mx.int32)
            out.append(t.item())
        seqs[mode] = out
    for mode in ("merge", "fused"):
        same = sum(a == b for a, b in zip(seqs["stock"], seqs[mode]))
        firstmatch = next((i for i, (a, b) in enumerate(zip(seqs["stock"], seqs[mode])) if a != b), 64)
        d = mx.abs(firstdiff[mode].astype(mx.float32) - firstdiff["stock"].astype(mx.float32)).max().item()
        print(f"{mode:8s} greedy match {same}/64, first divergence at {firstmatch}, step-0 max |dlogit| {d:.4f}")
    print("stock text :", repr(tok.decode(seqs["stock"][:24])))
    print("fused text :", repr(tok.decode(seqs["fused"][:24])))

    # timing: pipelined loops, modes interleaved, positions start after a 256-token prefill
    filler = mx.array([[1000 + i for i in range(prompt_len)]])

    def mk(mode):
        if mode == "stock":
            c = make_prompt_cache(model)
            mx.eval(model(filler, cache=c))
            return lambda t, i: mx.argmax(stock_logits(model, t, c)).reshape(1).astype(mx.int32)
        c = make_prompt_cache(model)
        mx.eval(model(filler, cache=c))
        return lambda t, i: mx.argmax(f.step(t, c, mode)).reshape(1).astype(mx.int32)

    modes = ("stock", "merge", "fused", "skeleton")
    res = {m: [] for m in modes}
    for _ in range(reps):
        for m in modes:
            res[m].append(time_loop(mk(m), n))
    print(f"\n{'mode':10s}{'ms/token':>10s}{'tok/s':>9s}   (median of {reps}, all runs ms)")
    for m in modes:
        v = sorted(res[m])
        med = v[len(v) // 2]
        print(f"{m:10s}{med*1e3:10.3f}{1/med:9.1f}   " + " ".join(f"{x*1e3:.3f}" for x in res[m]))


if __name__ == "__main__":
    main(sys.argv[1])
