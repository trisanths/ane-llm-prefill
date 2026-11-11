# Batch-1 Metal decode: where the time goes (2026-11-11)

Model: mlx-community/Qwen3-1.7B-4bit, revision 3b1b1768f8 (affine 4-bit,
group 64, bf16 scales and biases, tied embedding). M2 Pro, 16 GB, MLX 0.32.2,
mlx-lm 0.31.3. Prompt 256 tokens, 128 decode tokens, greedy.

Each decode token reads 968 MB of weights, of which the lm_head is 175 MB.

## Decode speed against the stream floor

| run | ms/token | tok/s | GB/s of weights |
|---|---|---|---|
| S: one quantized matmul over the same 968 MB | 6.19 | 161.5 | 156.3 |
| E: mlx_lm stream_generate | 7.60 | 131.6 | 127.3 |
| P: own loop, async_eval of t+1 before waiting on t | 7.78 | 128.5 | 124.4 |
| B: 8 independent streams per eval | 7.64 | 130.9 | 126.7 |
| A: own loop, blocking eval per token | 17.91 | 55.8 | 54.0 |

Scripts: `decode_probe.py` (A, B, S, D), `decode_probe2.py` (E, P).

- mlx_lm reaches 81 percent of the stream floor. The gap is 1.4 ms per token,
  which caps the gain from launch and fusion work at about 1.23x on this model.
- The gap is on the GPU, not the CPU. B and P hide all CPU work and land on
  the same 7.6 to 7.8 ms as mlx_lm.
- A blocking eval per token costs 2.3x. The GPU idles while Python builds the
  next graph. mlx_lm already pipelines, so this is a trap for custom loops,
  not a gain over mlx_lm.
- A dependent tiny kernel costs 6.85 us in a long chain, and an empty eval
  costs 388 us. Both were measured with the GPU mostly idle, so they include
  clock ramp-up and are an upper bound.

Correction: NOTES.md Track B said MLX decode lost a third of its speed to
about 250 launches per token. That run was Bonsai 2-bit under heavy swap. On
a clean run with this model, the loss is 19 percent.

## Kernels per token

From `mx.export_to_dot` of one decode token (992 graph nodes):

| primitive | count | per layer |
|---|---|---|
| QuantizedMatmul | 197 | 7, plus the lm_head |
| RMSNorm | 113 | 4 (input, post-attention, q-norm, k-norm), plus the final norm |
| RoPE | 56 | 2 |
| Add (residual) | 56 | 2 |
| SliceUpdate (KV write) | 56 | 2 |
| ScaledDotProductAttention | 28 | 1 |
| fused SwiGLU | 28 | 1 |

Reshape, Transpose and Broadcast nodes do not copy. Concatenate and Full
appear because this token crossed the 256-step KV allocation boundary; most
tokens do not have them.

That is about 535 dispatches per token. Of these, 197 stream weights and about
340 are small kernels (norms, RoPE, residual adds, KV writes). If the 1.4 ms
gap belongs to the small kernels, each costs about 4 us. Fusing them into the
neighbouring matmuls is the lever. This attribution is an estimate; a Metal
System Trace with per-dispatch timing would confirm it.

## Lossless compression bound

From `entropy.py`, order-0 entropy over all 1720.5M quantized weights:

| part | stored bits/weight | entropy bits/weight |
|---|---|---|
| 4-bit codes, single symbols | 4.000 | 3.700 |
| 4-bit codes, adjacent pairs | 4.000 | 3.699 |
| scales + biases, 16-bit symbols | 0.500 | 0.263 |
| scales + biases, coded as high and low bytes | 0.500 | 0.273 |
| total | 4.500 | 3.962 |

- The entropy bound is 12.0 percent fewer bytes. At the stream floor that is
  1.14x tok/s, and the model on disk and in RAM shrinks by the same amount.
- Adjacent codes are not correlated (pairs give no gain), so the code gain
  needs a per-symbol entropy coder. That is hard to decode at stream rate on
  a GPU. Treat the 7.5 percent from codes as an upper bound.
- Scales and biases carry most of the easy gain. Byte-wise coding gets to
  0.273 bits, which is 5 percent of all bytes. A fixed-width format in the
  style of F12 (exponent window plus escapes) would get close to that and
  decodes cheaply.
- Per-tensor code entropy runs from 3.630 to 3.725. The v_proj tensors are
  the most compressible.

## Combined headroom on this model

| lever | gain over mlx_lm |
|---|---|
| close the 1.4 ms GPU gap by fusing small kernels | up to 1.23x |
| lossless scales and biases (fixed width) | about 1.05x |
| lossless codes (entropy coder) | up to 1.08x more |
| all three | about 1.40x at best |

Speculative decoding is outside this table and can multiply on top of it.
