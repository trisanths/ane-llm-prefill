# Track C: ANE attention, per-op and blocked, vs GPU SDPA

M2 Pro, macOS 27.0, ANEForge. Qwen3-14B-ish attention shape: 40 query heads,
8 KV heads, head_dim 128, causal. Medians of 7-15 runs.

## C.1 Per-op timing and a bytes-moved model

Full attention core (QK^T, mask+softmax, PV), timed by differencing cumulative
compiled programs:

| M | QK^T | softmax | full core | core bytes | implied BW |
|---|---|---|---|---|---|
| 512 | 3.31 ms | 1.71 ms | 4.78 ms | 0.089 GB | 18.6 GB/s |
| 1024 | 9.85 ms | 8.55 ms | 15.81 ms | 0.346 GB | 21.9 GB/s |

The full-core time reproduces the earlier ~15 ms at M=1024. QK^T and softmax
dominate; the PV difference comes out negative because separately-compiled
programs do not fuse identically, so treat the split as QK^T+softmax ~= the whole
cost and PV as small.

**The bytes-moved model explains the slowness.** The core moves 0.346 GB at
M=1024 (the H x M x M score matrix, 84 MB, written by QK^T then read by softmax,
plus PV). At the 159 GB/s ceiling that is 2.2 ms. It takes 15.8 ms, so attention
runs at ~22 GB/s, about 7x off the bandwidth roofline. The ANE does not have a
fused-attention path here (af.sdpa is broken, per Exp earlier), so the score
matrix is materialized to and from memory, and that traffic is the cost.

## C.2 Blocked causal attention

Walking the key axis in blocks keeps the score matrix at H x M x B:

| M | full | best blocked | block | speedup | peak score |
|---|---|---|---|---|---|
| 512 | 4.78 ms | 3.94 ms | 512 | 1.21x | 21 -> 21 MB |
| 1024 | 15.74 ms | 13.07 ms | 512 | 1.20x | 84 -> 42 MB |
| 2048 | 75.24 ms | 48.81 ms | 256 | 1.54x | 335 -> 42 MB |

The win grows with M, and peak memory drops up to 8x. Best block lands the score
tensor near 40 MB. Cosine 1.00000 throughout. This matches the bytes-moved model:
blocking cuts the score traffic, so it helps most where the score matrix is
largest.

## C.3 GPU SDPA at identical shapes, and a walk-back

MLX flash SDPA on the GPU:

| M | GPU SDPA | TFLOPS | ANE blocked | GPU speedup |
|---|---|---|---|---|
| 512 | 2.13 ms | 2.52 | 3.94 ms | 1.85x |
| 1024 | 5.46 ms | 3.94 | 13.07 ms | 2.39x |
| 2048 | 21.06 ms | 4.08 | 48.81 ms | 2.32x |

**This revises the earlier "GPU is no faster at attention" conclusion.** That
came from the per-layer split harness, where the GPU path measured 14.9 ms at
M=1024, roughly equal to ANE blocked. The difference is that the split number
included host->MLX transfer, numpy conversion, and the GQA head expansion around
the kernel. The isolated flash-SDPA kernel is 5.46 ms, 2.4x faster than ANE
blocked, because the GPU has a hardware-tuned fused-attention kernel and the ANE
materializes the score matrix.

The reconciliation for the split decision:

- The GPU *kernel* is 2.4x faster at attention. That is real.
- But moving attention to the GPU mid-block costs the QKV transfer out and the
  context back. The earlier split measured that overhead at ~2 ms round trip,
  which does not erase a ~7.6 ms kernel saving at M=1024.
- So the honest statement is: **if the KV is already on the GPU, attention should
  run there.** In the hybrid design decode already lives on the GPU with the KV
  resident, so GPU attention is free to reach. It is prefill, where everything is
  on the ANE, that faces the transfer cost.

The net: for prefill-only on the ANE, use blocked attention (1.2-1.5x, 8x less
memory). For any path where the KV is already GPU-resident, the GPU's flash SDPA
is 2.4x faster and should be used.
