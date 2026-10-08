"""Track C.3: MLX SDPA on the GPU at the same attention shapes, for the split call."""
import time, json
import numpy as np, mlx.core as mx

H, KV, dh = 40, 8, 128
def med(f, n=15):
    for _ in range(4): f()
    return float(np.median([(lambda: (t:=time.perf_counter(), f(), time.perf_counter()-t)[-1])() for _ in range(n)]))

for M in (512, 1024, 2048):
    q = mx.random.normal((1,H,M,dh)).astype(mx.float16)
    k = mx.random.normal((1,H,M,dh)).astype(mx.float16)
    v = mx.random.normal((1,H,M,dh)).astype(mx.float16)
    mask = mx.array(np.triu(np.full((M,M), -np.inf, np.float32), 1).astype(np.float16))
    mx.eval(q,k,v,mask)
    scale = 1/np.sqrt(dh)
    def sdpa(): mx.eval(mx.fast.scaled_dot_product_attention(q,k,v,scale=scale,mask=mask))
    t = med(sdpa)
    flops = 4.0*M*M*H*dh
    print(json.dumps({"M":M, "mlx_sdpa_ms":round(t*1e3,3), "tflops":round(flops/t/1e12,2)}), flush=True)
    open("results/trackc.jsonl","a").write(json.dumps({"bench":"mlx_sdpa","M":M,"ms":t*1e3})+"\n")
