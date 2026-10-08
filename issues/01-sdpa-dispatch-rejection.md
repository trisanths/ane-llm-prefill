# af.sdpa: program compiles, then the driver rejects the bundle at dispatch

**aneforge** 0.4.0 · macOS 27.0 (26A5425a) · Apple M2 Pro (family 3) ·
ANECompiler.framework 10.26.6 · ANEServices.framework 10.19

## Summary

`af.sdpa` inside a transformer block compiles successfully, and then fails on the
first `execute()` with a driver-level rejection. A decomposed
matmul/softmax/matmul attention over the same tensors works.

Because `af.sdpa` is a graph cut whose segment is compiled lazily on first run,
`af.compile()` returning successfully is not evidence the program will run. That
made this considerably harder to bisect than it should have been.

## Error

For M of 128, 256 and 384:

```
RuntimeError: sdpa_invoker failed (rc=4):
stdout: {"status":"compile_failed","compile_ms":5.023,
         "error":"Error Domain=com.apple.appleneuralengine Code=10
                  \"verifyBundleAtPath: invalid model\"
                  UserInfo={NSLocalizedDescription=verifyBundleAtPath: invalid model}"}
stderr: compile failed: Error Domain=com.apple.appleneuralengine Code=10
        "verifyBundleAtPath: invalid model"
```

For M of 512 and above, aneforge refuses earlier, which seems correct and is not
what this report is about:

```
NotImplementedError: af.sdpa: causal attention at min(q,k)seq=512 (>= 512) or
seq=512 (> 2048) is outside the reliable native regime and the causal
decomposition ...
```

## Reproduction

Shapes are Qwen3-14B-like: d=5120, 40 query heads, 8 KV heads, head_dim=128,
SwiGLU ff=17408, causal, fp16.

```python
import numpy as np, aneforge as af

M, d, H, H_kv, dh = 256, 5120, 40, 8, 128
rng = np.random.default_rng(0)
w = lambda n, k: (rng.standard_normal((n, k)) / np.sqrt(k)).astype(np.float16)
Wq, Wk, Wv = w(H * dh, d), w(H_kv * dh, d), w(H_kv * dh, d)
cos, sin = af.rope_tables(M, dh)

x = af.input([M, d])
h = x.rms_norm(np.ones(d, np.float16), eps=1e-6)
q = h.linear(Wq).reshape([M, H, dh]).transpose([1, 0, 2])
k = h.linear(Wk).reshape([M, H_kv, dh]).transpose([1, 0, 2])
v = h.linear(Wv).reshape([M, H_kv, dh]).transpose([1, 0, 2])
q, k = af.rope(q, cos, sin), af.rope(k, cos, sin)
# GQA expansion by slice+concat (repeat_interleave is slow on this path)
k = af.concat([k.slice_by_size([i, 0, 0], [1, M, dh]) for i in range(H_kv) for _ in range(H // H_kv)], axis=0)
v = af.concat([v.slice_by_size([i, 0, 0], [1, M, dh]) for i in range(H_kv) for _ in range(H // H_kv)], axis=0)

ctx = af.sdpa(q.reshape([1, H, M, dh]), k.reshape([1, H, M, dh]), v.reshape([1, H, M, dh]),
              scale=1.0 / np.sqrt(dh), is_causal=True)

prog = af.compile(ctx.reshape([H, M, dh]))   # succeeds
prog(np.random.default_rng(1).standard_normal((M, d)).astype(np.float16))  # raises
```

## What was ruled out

- **Not the standalone op.** `af.sdpa` on three plain `af.input` tensors compiles
  and runs at H of 8 and 40, causal and non-causal, at M=256 and M=512.
- **Not any single upstream feature.** Projections, tiled projections, RoPE and
  the GQA expansion each feed `af.sdpa` successfully in isolation.
- **Not the downstream graph.** Output projection, residual, RMSNorm and the full
  SwiGLU MLP all compile on top of it.
- **Not the compile cache.** Reproduces with `~/Models/.aneforge-cache` removed.
- **Deterministic.** Identical failure across repeated fresh processes.

The failure appears only when the assembled block is actually executed.

## Impact

Attention has to use the decomposed form on this hardware. That is workable: a
full block reaches 7.75 TFLOPS with decomposed attention, and key-axis blocking
raises the attention core from 1.14 to 1.77 TFLOPS at M=2048. So this is not
blocking, but the native fused path is unreachable for this shape family.

## Suggestion

If the reliable-regime guard that fires at seq >= 512 has a counterpart condition
that applies below 512 for GQA-expanded inputs, extending the guard would turn a
driver-level rejection at dispatch into an actionable Python error at compile
time.
