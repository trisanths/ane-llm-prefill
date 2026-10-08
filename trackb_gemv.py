"""Track B.3: isolated 2-bit GEMV vs pure weight streaming, and mx.compile gain."""
import time, json
import numpy as np
import mlx.core as mx

def med(f, n=30):
    for _ in range(5): f()
    return float(np.median([(lambda: (t:=time.perf_counter(), f(), time.perf_counter()-t)[-1])() for _ in range(n)]))

# One Bonsai-sized MLP down-proj: [4096, 12288] 2-bit, gemv M=1
D, FF, gs, bits = 4096, 12288, 128, 2
W = mx.random.normal((D, FF)).astype(mx.float16)
wq, s, b = mx.quantize(W, group_size=gs, bits=bits); mx.eval(wq, s, b)
x = mx.random.normal((1, FF)).astype(mx.float16); mx.eval(x)

qbytes = wq.nbytes + s.nbytes + b.nbytes
def gemv(): mx.eval(mx.quantized_matmul(x, wq, s, b, transpose=True, group_size=gs, bits=bits))
t = med(gemv)
print(json.dumps({"op":"gemv_2bit", "MB":round(qbytes/1e6,1), "us":round(t*1e6,1),
                  "GBps":round(qbytes/t/1e9,1), "frac_of_159":round(qbytes/t/1e9/159.2,3)}), flush=True)

# pure streaming of the same bytes (sum over the packed uint32)
def stream(): mx.eval(mx.sum(wq.view(mx.uint32).astype(mx.float32)))
ts = med(stream)
print(json.dumps({"op":"pure_stream_same_bytes", "us":round(ts*1e6,1), "GBps":round(qbytes/ts/1e9,1)}), flush=True)

# mx.compile the gemv
cg = mx.compile(lambda xx: mx.quantized_matmul(xx, wq, s, b, transpose=True, group_size=gs, bits=bits))
def cgemv(): mx.eval(cg(x))
tc = med(cgemv)
print(json.dumps({"op":"gemv_2bit_compiled", "us":round(tc*1e6,1), "GBps":round(qbytes/tc/1e9,1),
                  "speedup":round(t/tc,3)}), flush=True)
