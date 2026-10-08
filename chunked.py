"""Item 1: chunked prefill -- one program, any prompt length.

Bucketed prefill compiles a separate program per prompt length. Each program
bakes the whole model, so buckets are expensive in compile time (20 s each) and
in disk (GBs each), and a prompt that half-fills its bucket wastes half the work.

Chunked prefill instead compiles ONE program that consumes C tokens at a time
against a fixed-capacity KV buffer:

  inputs : hidden [C, d], past K/V per layer [KV, L, dh], cos/sin [C, dh],
           mask [C, L+C]
  outputs: hidden [C, d], this chunk's K/V per layer

The host keeps the KV buffer and calls the program ceil(P/C) times. Position
dependence lives entirely in the cos/sin rows and the mask, both runtime inputs,
so one compile serves every prompt length up to L.
"""
import argparse
import json
import time
import warnings

import numpy as np
import aneforge as af

warnings.filterwarnings("ignore")
NEG = np.float16(-1e4)


def build_chunk_program(cfg, w, C, L, compress=None, layers=None):
    """One program: C new tokens attending to L cached positions plus themselves."""
    d, H, KV, dh = cfg.dim, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
    n_rep, eps = H // KV, cfg.norm_eps
    idx = range(cfg.n_layers) if layers is None else layers

    x = af.input([C, d])
    cos = af.input([C, dh])
    sin = af.input([C, dh])
    mask = af.input([H, C, L + C])
    past = [(af.input([KV, L, dh]), af.input([KV, L, dh])) for _ in idx]

    outs, h = [], x
    for n, li in enumerate(idx):
        W = w["layers"][li]
        hn = h.rms_norm(W["attn_norm"], eps=eps)
        q = hn.linear(W["wq"]).reshape([C, H, dh]).transpose([1, 0, 2])
        k = hn.linear(W["wk"]).reshape([C, KV, dh]).transpose([1, 0, 2])
        v = hn.linear(W["wv"]).reshape([C, KV, dh]).transpose([1, 0, 2])
        # rms_norm is 2D only, so flatten the head axis for the per-head q/k norm
        q = q.reshape([H * C, dh]).rms_norm(W["q_norm"], eps=eps).reshape([H, C, dh])
        k = k.reshape([KV * C, dh]).rms_norm(W["k_norm"], eps=eps).reshape([KV, C, dh])
        q, k = af.rope(q, cos, sin), af.rope(k, cos, sin)
        outs.append((k, v))                                   # emit BEFORE expansion
        pk, pv = past[n]
        fk = af.concat([pk, k], axis=1)                       # [KV, L+C, dh]
        fv = af.concat([pv, v], axis=1)
        ek = af.concat([fk.slice_by_size([i, 0, 0], [1, L + C, dh])
                        for i in range(KV) for _ in range(n_rep)], axis=0)
        ev = af.concat([fv.slice_by_size([i, 0, 0], [1, L + C, dh])
                        for i in range(KV) for _ in range(n_rep)], axis=0)
        s = af.einsum("hid,hjd->hij", q, ek) * np.float16(1.0 / np.sqrt(dh))
        p = (s + mask).softmax(axis=-1)
        ctx = af.einsum("hij,hjd->hid", p, ev)
        h = h + ctx.transpose([1, 0, 2]).reshape([C, H * dh]).linear(W["wo"])
        h2 = h.rms_norm(W["mlp_norm"], eps=eps)
        g = h2.linear(W["wgate"]).silu()
        h = h + (g * h2.linear(W["wup"])).linear(W["wdown"])
    # the batched prefiller applies final_norm at the last chunk; do the same here
    if layers is None or (list(idx) and list(idx)[-1] == cfg.n_layers - 1):
        h = h.rms_norm(w["final_norm"], eps=eps)
    flat = [h] + [t for kv in outs for t in kv]
    from aneforge._compile import compile_multi
    prog = compile_multi(flat, compress=compress)
    return prog, len(list(idx))


def chunk_mask(C, L, offset, filled, H):
    """[H, C, L+C]: query i (global offset+i) sees filled past slots and causal within chunk."""
    m = np.full((C, L + C), NEG, np.float16)
    m[:, :filled] = 0                                  # all real cached positions
    for i in range(C):
        m[i, L:L + i + 1] = 0                          # causal inside this chunk
    return np.broadcast_to(m, (H, C, L + C)).astype(np.float16).copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--compress", default=None)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--cap", type=int, default=1024)
    ap.add_argument("--prompt-len", type=int, nargs="+", default=[64, 128, 256, 512, 1024])
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    import diskguard, memguard
    diskguard.guard(limit_gb=5.0, min_free_gb=8.0)
    m = af.load_llm(a.model, compress=a.compress)
    cfg, C, L = m.cfg, a.chunk, a.cap
    H, KV, dh = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim

    t0 = time.perf_counter()
    prog, nl = build_chunk_program(cfg, m.w, C, L, a.compress)
    compile_s = time.perf_counter() - t0
    cos_t, sin_t = af.rope_tables(L + C, dh, base=cfg.rope_base)
    cos_t, sin_t = np.asarray(cos_t), np.asarray(sin_t)

    def run_prompt(ids):
        P = len(ids)
        emb = m.w["embed"][np.asarray(ids)].astype(np.float16)
        pk = [np.zeros((KV, L, dh), np.float16) for _ in range(cfg.n_layers)]
        pv = [np.zeros((KV, L, dh), np.float16) for _ in range(cfg.n_layers)]
        filled = 0
        h = None
        for off in range(0, P, C):
            n = min(C, P - off)
            xin = np.zeros((C, cfg.dim), np.float16)
            xin[:n] = emb[off:off + n]
            args = [xin, cos_t[off:off + C], sin_t[off:off + C],
                    chunk_mask(C, L, off, filled, H)]
            for li in range(cfg.n_layers):
                args += [pk[li], pv[li]]
            out = prog(*args)
            op = [name for _, name in prog.output_ports]   # ordered: hidden, then k,v per layer
            h = np.asarray(out[op[0]])
            for li in range(cfg.n_layers):
                k = np.asarray(out[op[1 + 2 * li]])[:, :n, :]
                v = np.asarray(out[op[2 + 2 * li]])[:, :n, :]
                pk[li][:, filled:filled + n, :] = k
                pv[li][:, filled:filled + n, :] = v
            filled += n
        return h[(P - 1) % C if P % C else C - 1]

    for P in a.prompt_len:
        if P > L:
            continue
        rng = np.random.default_rng(0)
        ids = rng.integers(1000, 140000, size=P).tolist()
        run_prompt(ids)
        ts = []
        for _ in range(a.repeats):
            t0 = time.perf_counter()
            run_prompt(ids)
            ts.append(time.perf_counter() - t0)
        r = {"bench": "chunked", "model": a.model, "chunk": C, "cap": L,
             "prompt_len": P, "n_chunks": (P + C - 1) // C,
             "compile_s": round(compile_s, 1),
             "ttft_s": float(np.median(ts)), "ttft_sd_s": float(np.std(ts)),
             "prefill_tok_s": P / float(np.median(ts))}
        diskguard.stamp(r); memguard.stamp(r)
        print(json.dumps(r), flush=True)
        if a.out:
            with open(a.out, "a") as f:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
