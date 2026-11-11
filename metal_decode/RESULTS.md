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

## Fusing the small kernels (2026-11-11, later the same day)

`fused_qwen3.py` builds the decode step from the loaded weights in four modes.
`ab.py` runs them interleaved, 12 rounds of 64 tokens each, with the order
rotated per round. All modes use mlx_lm's own KVCache, so KV handling is
identical and only the kernels differ.

| mode | kernels per token | median ms | vs stock (paired median) |
|---|---|---|---|
| stock mlx_lm | 650 | 8.34 | 1.000 |
| merge: q/k/v as one matmul, gate/up as one | 566 | 9.41 | 0.933 |
| fused: merge plus add+RMSNorm and qk-norm+RoPE kernels | 426 | 8.41 | 0.969 |
| skeleton: matmuls and attention only, wrong output | 313 | 7.99 | 1.069 |

The kernel counts include a KV-buffer resize, because the 256-token prompt
sits on the 256-step allocation boundary; most tokens have about 110 fewer.

- **Fusion is worth at most about 7 percent.** The skeleton deletes every
  norm, RoPE, residual add and SwiGLU, and gains only 1.07x. The earlier
  "up to 1.23x" bound assumed that the small kernels cause the whole 1.4 ms
  gap. They do not.
- **The fused path ties stock.** It cut 224 kernels and did not get faster.
  Its greedy tokens match stock for 27 of 64 tokens (step-0 max logit
  difference 0.31), and the text is the same for the first 24 tokens. The
  difference is bf16 rounding order in the norm and RoPE, not a bug that
  changes the meaning, but it is not bit-exact.
- **Merging matmuls helps alone and not in the full step.** Alone, merged
  q/k/v runs 28 layers in 1.47 ms against 2.06 ms for three matmuls. In the
  full step, merge reads slower than stock. The spread across rounds is
  0.65 to 1.23, so the full-step effect is inside the noise.
- **The rest of the gap is kernel boundaries and attention.** The 113 weight
  matmuls alone, run as a dependent chain, take 7.17 ms. Run independently,
  they take 6.56 ms, which is the stream floor (`qmv_chain.py`). That is
  about 5 us per dependency boundary. The o_proj matmul (2.36 MB) streams at
  93 GB/s and q/k/v (4.72 MB) at 125 GB/s; large ones reach 148 to 164 GB/s.
  Small matrices pay their start and tail.
- **Machine noise.** The desktop shares the GPU with WindowServer, a VM and a
  browser. Single runs of identical code move 10 to 30 percent. Use paired
  interleaved rounds and medians only.

### A trap: in-place KV writes and pipelining

A decode loop that keeps its own preallocated KV buffer and writes it with
`buf[..., pos:pos+1] = k` can lose MLX's in-place donation when the next
token's graph is queued before the previous token finishes. MLX then copies
the whole buffer on every write. With a 4096-token buffer this cost 0.83 ms
per token against a 512-token buffer; a microtest with 56 buffers moved from
0.75 to 3.2 ms when only the capacity changed. mlx_lm's KVCache did not show
the copy at 16K context (23.3 ms against a 19.0 ms floor). Keep KV in mlx_lm's
cache class, or verify donation, before trusting a custom loop.

### Context length, stock mlx_lm

| context | KV read per token | ms/token | tok/s | floor at 150 GB/s |
|---|---|---|---|---|
| 256 | 29 MB | 9.64 | 103.7 | 6.65 ms |
| 4096 | 470 MB | 13.33 | 75.0 | 9.58 ms |
| 16384 | 1879 MB | 23.26 | 43.0 | 18.98 ms |

At 16K context, the KV cache is twice the weight bytes. Decode speed at long
context is a KV-bytes problem, which makes KV quantization the lever for
priority 3. This table was measured during heavy machine load (the 256-token
row is 1.2x slower than the clean run above).

## Headroom on this model, revised

| lever | gain over mlx_lm |
|---|---|
| fuse small kernels (measured upper bound) | up to 1.07x |
| remove matmul dependency barriers (one persistent kernel) | up to 1.09x more |
| lossless scales and biases (fixed width) | about 1.05x |
| lossless codes (entropy coder) | up to 1.08x more |

Kernel work on batch-1 decode for this model is worth about 1.2x at best, in
total, and the persistent-kernel part needs grid-wide sync, which Metal does
not guarantee. Speculative decoding is the lever that can pass the stream
floor. KV quantization is the lever for long context.

## Speculative decoding (2026-11-11, evening)

Target mlx-community/Qwen3-8B-4bit (545dc4251c), draft
mlx-community/Qwen3-0.6B-4bit (73e3e38d98), mlx_lm's own speculative loop,
greedy, chat template with thinking off, 192 new tokens per prompt.

### Verify cost: MLX re-reads the weights for every row

MLX's `qmv` launches one set of threadgroups per input row, so a forward over
M tokens reads the weights M times until M reaches the `qmm` path at 6, which
pads M to 32. Checking k draft tokens therefore costs nearly k decode steps.
That cost sets whether speculative decoding can win at all.

`qmv_h2.py` is a small-M kernel for affine 4-bit, group 64. It reads each
weight word once for all M rows. It builds two fp16 weights per pair of
integer ops (OR the nibbles into the 1024.0 bit pattern, subtract 1024
exactly) and accumulates in half2 only within one 8-value word, then in fp32.
Error against an fp32 dequantized reference is 2.4e-3 (MLX: 7e-3 at M <= 4),
and stays 2.4e-3 with activations near +-300. `spec_patch.py` routes
2 <= M <= 8 through it; M = 5 and 7 are padded to 6 and 8.

Verify time over L new tokens at 256 context, as a multiple of L = 1
(`verify_cost.py`, medians of 15 interleaved rounds):

| L | 8B, MLX | 8B, qmv_h2 | 1.7B, MLX | 1.7B, qmv_h2 |
|---|---|---|---|---|
| 2 | 1.58 | 1.15 | 1.48 | 1.33 |
| 3 | 2.25 | 1.39 | 2.11 | 1.63 |
| 4 | 3.03 | 1.74 | 2.68 | 2.00 |
| 6 | 5.46 | 2.37 | 4.71 | 2.61 |
| 8 | 5.85 | 3.14 | 5.04 | 3.29 |

The kernel is still ALU-bound, not bandwidth-bound: the 4-bit unpack plus
one FMA per weight per row saturates the M2 Pro's ALUs before the memory
bus. Splitting rows across simd groups, half-precision x, wider tiles and
more rows per simd group were all swept (`tune_qmv.py`, `tune_h2.py`); none
reached L = 4 for less than 1.7x. A simdgroup_matrix path with an 8-row M
tile is the next step for this kernel.

### End to end

Speedup over the 8B target alone (33.5 tok/s mean). "acc" is the share of
output tokens that came from the draft.

| prompt | best stock mlx_lm | best with qmv_h2 | acc at that k |
|---|---|---|---|
| code | 0.94x (k=2) | 1.35x (k=3) | 0.69 |
| prose | 0.61x (k=2) | 0.81x (k=3) | 0.49 |
| explain | 0.69x (k=2) | 0.85x (k=2) | 0.51 |
| list | 0.75x (k=2) | 0.95x (k=2) | 0.51 |
| mean | 0.75x (k=2) | 0.97x (k=2) | 0.52 |

- **Stock speculative decoding is a net loss on this Mac** at every k for
  every prompt type. Most users who turn on `--draft-model` here get slower
  output.
- **With the verify kernel it is break-even on average** and a 1.35x win on
  code, where the 0.6B draft agrees with the target more often.
- **A cost model fits the numbers.** Per round: k draft steps (about 6.5 ms
  each) plus one verify of k+1 tokens, divided by the tokens kept. At k=2 on
  prose, (2 x 6.5 + 50) ms / 2.1 tokens is about 30 ms per token, which is
  0.97x of the 33 ms target step.

### What would make it win

| lever | effect in the cost model |
|---|---|
| verify kernel at about 1.05x for L <= 4 (simdgroup_matrix) | k=3 code to about 1.8x, mean above 1.2x |
| draft cost near its 2.3 ms floor (fused tiny model, or the draft on the ANE in parallel) | removes about 20 percent of each round |
| zero-cost drafts (prompt lookup or n-gram) for code and edits | removes the draft step entirely where text repeats |
| a better-matched draft (Qwen3-1.7B, or a trained draft head) | raises acceptance, at more draft cost |
