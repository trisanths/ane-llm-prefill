# Track B: MLX 2-bit decode roofline (Ternary Bonsai 8B)

M2 Pro, 16 GB. macOS 27.0. Measured under memory pressure (swap ~8.5 GB, other
apps resident), so absolute tok/s is a floor, not a best case.

## B.1 Bandwidth ceiling

Achievable GPU read bandwidth, MLX `sum` over fp16:

| buffer | GB/s |
|---|---|
| 256 MB | 151.5 |
| 1 GB | 158.1 |
| 2.3 GB | 159.2 |

So 159 GB/s is the streaming ceiling on this machine, close to the 200 GB/s spec.

## B.2 Decode against the roofline

Bonsai 8B is 2.3 GB of 2-bit weights, streamed once per token.

| quantity | value |
|---|---|
| weight bytes | 2.30 GB |
| bandwidth ceiling | 69.1 tok/s |
| measured decode | 45.1 tok/s |
| fraction of roofline | 65% |
| implied bandwidth | 104 GB/s |
| sampling + Python loop | negligible (below noise) |

Forward-only and full-decode timings are within noise of each other, so the
argmax and the Python token loop cost nothing measurable. The 35% gap to the
roofline is entirely in the GEMV kernels not saturating bandwidth at M=1.

## B.3 Where the gap is

One Bonsai-shaped 2-bit GEMV, [4096, 12288], M=1, 14.2 MB of packed weight:

| op | GB/s | fraction of 159 |
|---|---|---|
| isolated 2-bit GEMV | 50.4 | 32% |
| pure stream, same bytes | 38.0 | 24% |
| GEMV after mx.compile | 51.0 | 32% (1.01x) |

Two things this settles:

- **MLX's kernel is not the problem.** The quantized GEMV moves 14 MB *faster*
  than a naive sum over the same bytes. It is well tuned.
- **The gap is launch latency at M=1.** A single 14 MB op cannot saturate a
  159 GB/s bus; it is dominated by kernel-launch and dispatch overhead. In the
  full model, 36 layers of ops pipeline and reach 104 GB/s, but per-token there
  are ~250 tiny launches (36 layers x 7 matrices) and the queue never stays full.

`mx.compile` on the isolated op gives nothing (1.01x), because the op is already
one kernel; there is nothing to fuse. It would only help by fusing the per-layer
launch chain, which is a full-step compile, not an op-level one.

## Verdict on a custom Metal GEMV

The criterion was: build one only if MLX is under 80% of bandwidth. The full
model is at 65%, which is under the bar, but the isolated-kernel evidence says a
custom GEMV would not help: MLX already beats naive streaming, and the loss is
launch latency, not kernel efficiency. The right lever is fewer, larger launches
(batched decode, or a fused multi-layer step), not a faster GEMV.

The one caveat is memory pressure: with swap at 8.5 GB these numbers are a floor.
On an unloaded machine the implied bandwidth would likely rise toward the ceiling
and the roofline gap would shrink, which would only strengthen the conclusion
that a custom kernel is not worth it.
