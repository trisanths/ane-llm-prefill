# Step 2b: one fused transformer block on the ANE

Shapes are Qwen3-14B-like: d=5120, 40 query heads, 8 KV heads, head_dim=128,
SwiGLU with ff=17408, pre-norm, causal. Same machine and pins as step 2a.
Medians over 9 timed runs. Kernel in `../block2b.py`,each constraint commented in
place. Raw rows in `block_bench.jsonl`, plots in `block_bench.png`.

## Result against the target

The MLP target was 6 TFLOPS. It runs at 9.04.

| piece | M=1024 time | TFLOPS | share of block FLOPs | share of block time |
|---|---|---|---|---|
| MLP alone | 60.6 ms | 9.04 | 89 % | 67 % |
| attention core alone | 15.6 ms | 1.38 | 3.7 % | 17 % |
| full block, fp16 | 90.1 ms | 7.75 | 100 % | 100 % |
| full block, int8 | 73.1 ms | 9.55 | 100 % | 100 % |

Attention is the weak part, running 6.5 times slower per FLOP than the MLP. It
is cheap enough in FLOP terms that the block still clears the target, but at
longer prompts its quadratic term will dominate, so it is the thing to attack
next.

## Against the GPU

Identical weights, identical math, one program each.

| M | ANE | GPU (MLX) | ANE advantage |
|---|---|---|---|
| 256 | 22.5 ms | 36.0 ms | 1.60x |
| 512 | 47.2 ms | 72.3 ms | 1.53x |
| 1024 | 90.6 ms | 143.9 ms | 1.59x |

The advantage is smaller than the 2.4x that pure chained GEMM showed in step 2a,
because attention drags the average down.

## Accuracy

Checked against both an fp32 numpy reference and an fp16 MLX implementation of
the same block. The MLX comparison is the fair one for argmax, since fp32 numpy
has no rounding of its own.

| comparison | cosine | argmax agreement | max relative error |
|---|---|---|---|
| ANE vs MLX fp16 | 0.999981 | 0.986 | 0.0042 |
| ANE vs fp32 | 0.999987 | 0.987 | 0.0042 |
| MLX fp16 vs fp32 | 0.999995 | 0.995 | 0.0007 |

Cosine clears the 0.999 bar with four digits to spare. Argmax agreement is 98.6
percent rather than 100, and that gap is worth explaining rather than waving at.

At M=512 there are 8 disagreeing rows out of 512. At every one of them the
relative gap between the true top-1 and the value the ANE picked has median
1.0e-3, while the typical top-1 to top-2 gap across all rows is 5.4e-2, fifty
times larger. Every single mismatch is a near-tie that fp16 rounding could flip
either way. Note also that this argmax is over a residual hidden state, not over
logits, because there is no lm_head in a single block. It is a weaker signal
than it looks and the cosine is the number to trust.

int8 holds up here too, at cosine 0.999957, unlike the W8A8 disaster in step 1b.
The difference is that this path quantizes weights only and leaves activations
in fp16.

## Constraints hit

Each of these is commented at the point of use in `block2b.py`.

### Family 3 caps every tensor dimension at 16384

ff=17408 exceeds it, so a SwiGLU intermediate of shape [M, 17408] is rejected
outright with a clear error. The fix is also the fast path: chunk the MLP along
ff, and for each chunk compute gate, up, their product, and that chunk's
down-projection contribution, then sum the contributions. The wide intermediate
never exists, and each down-projection now contracts over 2176 rather than
17408, which the step 2a tiling law wants anyway.

### Native fused attention is unusable for this block

`af.sdpa` fails at every sequence length tried.

| M | outcome |
|---|---|
| 128, 256, 384 | compiles, then the driver rejects the bundle at dispatch: `verifyBundleAtPath: invalid model` |
| 512 and up | ANEForge refuses upfront, calling causal attention at seq >= 512 outside the reliable native regime |

The dispatch-time failure took some pinning down, because `af.sdpa` is a graph
cut: `af.compile` succeeds and the segment is only compiled when the program
first runs. Bisecting by compiling alone showed everything passing. The failure
only appears on execution.

This is the question step 2b existed to answer, and the answer is good. Attention
does not fall off the ANE. The decomposed matmul, softmax, matmul path runs
entirely on the engine, in one program, at 7.75 TFLOPS for the whole block. Only
Apple's native fused-attention layer is broken for this shape. Nothing here
argues for going under the driver.

### The step 2a tiling law holds on a real block

| kt | block TFLOPS |
|---|---|
| 1280 | 8.66 |
| 2560 | 7.69 |
| 5120 | 5.06 |

Same 1.7x spread as the synthetic GEMM sweep, from nothing but how the
contraction is split. d=5120 and ff=17408 are not multiples of 2048, so the
tiler handles ragged last tiles; 5120 splits as 2x2560 and 17408 as 8x2176,
both landing near the plateau without a remainder.

### GQA expansion avoids repeat_interleave

K and V are expanded to 40 heads by slicing each KV head and concatenating,
which lowers to slice and concat rather than the slow repeat path the ExecuTorch
notes warn about.

### Native RMSNorm beats the LayerNorm trick

The ANEMLL `concat([x, -x])` trick exists because Core ML has fused LayerNorm and
no RMSNorm. ANEForge exposes `rms_norm` natively, and it is faster: 90.1 ms
against 94.6 ms for the trick, a 5 percent penalty for the workaround. Do not
port that trick here.

### A warned numerical hazard that did not reproduce

Compiling any last-axis slice at a non-zero offset raises an ANEForge warning
that pre-A16 hardware saturates sliced values above 4094 to infinity through a
Q.4 crop-DMA path. Since reduction-axis tiling is built entirely on last-axis
slices, this would be a correctness landmine for real models, whose residual
streams do carry large outliers.

It did not reproduce. Feeding a tiled linear inputs with maximum magnitude 20192,
five times the stated threshold, gave a maximum relative error of 0.0005 against
an untiled reference and produced no infinities. Either the saturation applies to
a path this shape does not take, or the threshold is conservative. Recording it
as warned but unreproduced, and worth retesting on real model activations before
trusting tiling at scale.

## Costs

Compile time for the fp16 block runs 7 to 12 seconds, growing with tile count:
107 ops at kt=5120, 187 at kt=2560, 339 at kt=1280. The fastest configuration is
also the slowest to build. Cacheable and one-time per machine.

## What this sets up for 2c

The prefill engine works and beats the GPU by about 1.6x per block at full fp16
accuracy. Two things to carry forward. Attention at 1.38 TFLOPS is the ceiling on
long prompts and deserves a tiled or flash-style rewrite before chunk sizes grow.
And int8 weights with fp16 activations give another 1.23x on the block at no
meaningful accuracy cost, so it is worth enabling once the hybrid loop is correct.

---

# Additions: per-layer drift, W8A16, and a correction

Run after the sections above, at kt=2048 as agreed. Naming corrected throughout:
what earlier sections called "int8" on the ANEForge path is weight-only int8 with
fp16 activations, so it is **W8A16**. ANEForge's quantizer is weight-only; there
is no activation quantization in it. The W8A8 result that blew up in step 1b came
from Core ML, which is a different thing entirely, and the two should never have
shared a label.

## Per-layer cosine along the residual stream

Eight blocks with independent weights, chained, residual captured at every depth
and compared against an fp32 numpy reference of the same chain. M=512.

| depth | fp16 kt=1024 | fp16 kt=2048 | W8A16 kt=1024 | W8A16 kt=2048 |
|---|---|---|---|---|
| 1 | 0.999991 | 0.999991 | 0.999964 | 0.999961 |
| 2 | 0.999982 | 0.999982 | 0.999926 | 0.999921 |
| 3 | 0.999973 | 0.999973 | 0.999887 | 0.999879 |
| 4 | 0.999964 | 0.999964 | 0.999848 | 0.999838 |
| 5 | 0.999956 | - | 0.999809 | - |
| 6 | 0.999949 | - | 0.999771 | - |
| 7 | 0.999942 | - | 0.999733 | - |
| 8 | 0.999935 | - | 0.999696 | - |

Two things fall out.

**Tile width does not affect accuracy.** The fp16 columns for kt=1024 and kt=2048
are identical to six decimal places at every matching depth. Reduction tiling
changes the order of summation and nothing else that matters. It buys throughput
for free, which is the cleanest result in this project.

**Drift is linear in depth, and W8A16 drifts about 4.5 times faster.** fp16 loses
roughly 8e-6 of cosine per layer, W8A16 roughly 3.8e-5. Extrapolating the W8A16
slope, a 28-layer model lands near 0.99894 and a 40-layer model near 0.99849,
so W8A16 crosses the 0.999 bar somewhere around 26 to 31 layers of depth. fp16 at
40 layers is still 0.99968.

That is a real constraint on where W8A16 is safe, and it is not visible from a
single block. A one-block cosine of 0.99996 looks like plenty of headroom and is
not.

Measured directly against fp16 on identical inputs, rather than against fp32:

| depth | W8A16 vs fp16, kt=1024 | kt=2048 |
|---|---|---|
| 1 | 0.999972 | 0.999969 |
| 4 | 0.999879 | 0.999869 |
| 8 | 0.999751 | - |

Same slope, which confirms the drift is the quantization and not fp16 rounding.

## W8A16 as a measured delta on the block

Full block, kt=2048, fp16 against W8A16 on identical inputs.

| M | fp16 TFLOPS | W8A16 TOPS | delta | fp16 cosine | W8A16 cosine |
|---|---|---|---|---|---|
| 256 | 8.00 | 9.63 | 1.20x | 0.999989 | 0.999957 |
| 512 | 8.05 | 9.15 | 1.14x | 0.999988 | 0.999959 |
| 1024 | 8.49 | 9.12 | 1.07x | 0.999987 | 0.999958 |

The delta shrinks as M grows, from 20 percent down to 7 percent.

## Correction: the int8 ratio does not survive tiling

This is the part worth pausing on before anything goes upstream.

Untiled, the W8A16 advantage holds across M and is large, including at
compute-bound M where a dequantize-only implementation should show nothing:

| M | 128 | 512 | 1024 | 2048 | 4096 |
|---|---|---|---|---|---|
| ratio, 8 chained layers | 1.87 | 1.68 | 1.68 | 1.62 | 1.62 |

Tiled, it mostly evaporates:

| kt | fp16 | W8A16 | ratio |
|---|---|---|---|
| 4096 (untiled) | 6.89 | 10.54 | 1.53x |
| 2048 | 11.07 | 12.83 | 1.16x |
| 1024 | 12.19 | 13.59 | 1.12x |

If W8A16 were running the multiplies at a native 2x integer rate, that rate would
be orthogonal to how the contraction is split, and the ratio would hold at every
kt. It does not. Instead, tiling and W8A16 deliver almost the same speedup, and
applying both gives barely more than either alone. That is the signature of two
techniques relieving one shared bottleneck, which on this evidence is the cost of
a wide reduction rather than the multiply rate.

So the earlier framing of "int8 is native, not dequantized, contradicting the
docs" is too strong and should not be reported upstream as written. ANEForge's
statement that int8 weights dequantize on-device is compatible with everything
measured here: dequantizing from int8 halves the bytes pulled into the array per
contraction step, which speeds up exactly the wide reductions that tiling also
fixes, and helps nothing once the reduction is already narrow.

What would actually settle it is a kernel where the reduction is narrow and the
weight volume is large, so the two effects separate. That is not measured yet.
Until it is, the defensible claim is narrower: W8A16 gives 1.6x on untiled wide
reductions and 1.1x on tiled ones, and the mechanism is unresolved.

## Operational: a failed compile poisons the cache

Chasing an apparent "kt=2048 does not compile" result cost real time and the
conclusion was wrong. kt=2048 compiles fine.

What happens is that a compile failing under memory pressure leaves an entry in
`~/Models/.aneforge-cache` keyed by the graph, and every later attempt at the
same graph then fails identically, including from a fresh process. It looks
exactly like a deterministic structural limit. Clearing the cache directory makes
the same code compile first try.

The underlying pressure is real and worth planning around: each block at these
shapes bakes about 600 MB of weights into its program, mostly the three
17408-wide MLP matrices, and the numpy weights are held alongside. An 8-layer
fp16 stack needs roughly 10 GB between the two and does not fit in 16 GB. The
W8A16 version, at half the weight bytes, does fit. Build stacks in separate
processes, or free weights between layers.

Two rules for later steps. Treat any repeated compile failure as a suspect cache
before believing it is a hardware limit. And expect the cache to grow fast: it
reached 20 GB twice in one session and filled the volume to 99 percent.
