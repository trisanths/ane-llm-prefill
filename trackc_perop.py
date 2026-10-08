"""Track C.1: per-op timing of the attention subgraph on the ANE, M=512/1024.

Times each stage as its own compiled program by diffing cumulative programs:
QKV proj, RoPE, GQA expand, QK^T, mask+softmax, PV, transposes. Then a
bytes-moved model to see if bandwidth predicts the ~15 ms full-attention time.
"""
import time, json, warnings
import numpy as np, aneforge as af
warnings.filterwarnings("ignore")

H, KV, dh = 40, 8, 128
def med(prog, sample, n=9):
    args = [sample] * len(prog._inputs)
    for _ in range(3): prog(*args)
    return float(np.median([(lambda: (t:=time.perf_counter(), prog(*args), time.perf_counter()-t)[-1])() for _ in range(n)]))

def run(M):
    rng = np.random.default_rng(0)
    scale = np.float16(1/np.sqrt(dh))
    q = af.input([H, M, dh]); k = af.input([H, M, dh]); v = af.input([H, M, dh])
    mask = np.triu(np.full((M,M), -1e4, np.float16), 1)
    qi,ki,vi = (rng.standard_normal((H,M,dh)).astype(np.float16) for _ in range(3))

    # stage endpoints, each compiled and timed
    stages = {}
    s = af.einsum("hid,hjd->hij", q, k) * scale
    stages["qkT"] = af.compile(s)
    p = (s + mask).softmax(axis=-1)
    stages["qkT+softmax"] = af.compile(p)
    ctx = af.einsum("hij,hjd->hid", p, v)
    stages["full_attn_core"] = af.compile(ctx)

    times = {name: med(prog, qi) for name, prog in stages.items()}
    # differences give per-stage cost
    out = {"M": M,
           "qkT_ms": round(times["qkT"]*1e3,3),
           "softmax_ms": round((times["qkT+softmax"]-times["qkT"])*1e3,3),
           "pv_ms": round((times["full_attn_core"]-times["qkT+softmax"])*1e3,3),
           "full_core_ms": round(times["full_attn_core"]*1e3,3)}
    # bytes-moved model for the core: scores SxS x H fp16, read+write
    score_bytes = 2 * H*M*M  # write scores
    sm_bytes = 2*2*H*M*M     # read+write softmax
    pv_bytes = 2*(H*M*M + H*M*dh)
    total_gb = (score_bytes + sm_bytes + pv_bytes)/1e9
    out["core_bytes_GB"] = round(total_gb,3)
    out["implied_bw_GBps"] = round(total_gb/(times["full_attn_core"])/1e9*1e9,1) if False else round(total_gb/times["full_attn_core"],1)
    for p in stages.values(): p.release()
    return out

for M in (512, 1024):
    r = run(M); print(json.dumps(r), flush=True)
    open("results/trackc.jsonl","a").write(json.dumps(r)+"\n")
