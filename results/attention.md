# Attention: key-axis blocking, and the GPU split

Both experiments measured against the step 2b baseline of 1.38 TFLOPS for the
decomposed full-attention kernel. Shapes are 40 heads of 128, causal. Medians
over 7 to 9 runs. Raw rows in `attn.jsonl` and `split.jsonl`.

## 1a. Key-axis blocking

Walking the key axis in blocks of B keeps the widest intermediate at [H, M, B]
instead of [H, M, M].

| M | baseline (full) | best blocked | B | gain | peak intermediate |
|---|---|---|---|---|---|
| 512 | 1.22 | 1.40 | 512 | 1.15x | 21.0 MB |
| 1024 | 1.36 | 1.66 | 512 | 1.22x | 41.9 MB |
| 2048 | 1.14 | 1.77 | 256 | 1.55x | 41.9 MB (from 335.5) |

The win grows with M, which is what matters: the baseline degrades as M rises
(1.36 down to 1.14) while the blocked version improves (1.66 up to 1.77). At
M=2048 blocking is worth 55 percent and cuts the largest tensor by 8x.

The optimum block is not the smallest. At M=1024, B=128 gives 1.39 against
B=512's 1.66. Op count grows as M/B and per-op overhead eats the gain, so the
best B is the one that lands the score tensor near 20 to 42 MB, which means
B=512 at M=1024 and B=256 at M=2048.

This does not follow the step 2a reduction-width law. If throughput tracked the
PV contraction width, smaller B would keep winning; it does not. Attention is not
limited by the same thing the MLP is.

Accuracy is unaffected: cosine against fp32 stays at 0.999998 or better at every
block size, with maximum relative error 0.0013.

### Surprise: the textbook online-softmax form does not compile

The flash formulation carries a running maximum and rescales each block. It
fails to compile on this ANE compiler, and the bisect is specific: every
individual step compiles, and the failure appears only where the final
normalization joins the accumulated output to the accumulated denominator.
Combining either accumulator with a shallow operand is fine. Combining the two
with each other is not, whether by divide, multiply or add. It is a
shared-deep-ancestor diamond that breaks, not any one operator.

Dropping the running maximum removes the depth and compiles at every block size.
The cost is numerical: scores are exponentiated without recentring, so anything
above about 11 overflows fp16. Measured here, scores span -5.4 to 5.1 and the
largest exponential is 169 against a limit of 65504, with no infinities in the
output. That is comfortable for RMSNorm'd activations and is checked per
configuration, but this kernel is not safe for unbounded scores and should not be
shipped without either a bound or a working online form.

## 1b. MLP on ANE, attention on GPU

| stage | M=512 | M=1024 |
|---|---|---|
| ANE front, norm + QKV | 5.10 ms | 9.38 ms |
| host to MLX | 0.27 ms | 0.53 ms |
| GPU attention | 4.16 ms | 14.92 ms |
| MLX to host | 0.65 ms | 0.95 ms |
| ANE back, output proj + MLP | 35.75 ms | 60.96 ms |
| **split total** | **45.93 ms** | **86.73 ms** |
| monolithic ANE block | 42.97 ms | 82.23 ms |
| ANE blocked attention alone | 4.06 ms | 12.95 ms |

The split loses, by 5 percent at both sizes. Two reasons, and the second is the
interesting one.

**Transfer is cheap.** Moving q, k, v out and the context back costs 1.48 ms at
M=1024, 2 percent of the split. Unified memory delivers on its promise here.
This is good news well beyond this experiment: it means the 2c KV bridge will
not be dominated by transfer.

**The GPU is not better at attention.** 14.92 ms on the GPU against 12.95 ms for
the blocked ANE kernel on the same work. Attention runs at roughly 1.4 TFLOPS on
both engines, so there is nothing to gain by moving it, and the move costs a
transfer plus the loss of fusion.

So attention stays on the ANE. The recommendation from step 2b stands but the
reason has changed: it is not that the GPU is unavailable, it is that the GPU is
no faster.

## Where this leaves attention

Blocked attention at M=1024 is 1.66 TFLOPS against the MLP's 9. It remains about
5x less efficient per FLOP and is still the ceiling on long prompts. Blocking
bought 22 to 55 percent and cut peak memory 8x, which is worth having, but it
does not change the shape of the problem.

Attention is only 3.7 percent of block FLOPs at M=1024, so at that size it is not
worth more effort. The term is quadratic though, so the crossover matters: at
M=2048 attention is 7.3 percent of FLOPs, and the blocked kernel is what keeps
the intermediate from reaching 335 MB. For prefill buckets of 512 or 1024 tokens,
which is what 2c will use, blocking is sufficient and attention is not the
bottleneck.
