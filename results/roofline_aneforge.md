# Step 2a: roofline through ANEForge

M2 Pro, 16-core GPU, 16 GB. macOS 27.0 build 26A5425a, ANECompiler 10.26.6,
ANEServices 10.19, aneforge 0.4.0, MLX 0.32.2. All figures are medians over 7 to
9 timed runs after warmup. Raw rows in `roofline_aneforge.jsonl`, plots in
`roofline_aneforge.png`.

The GPU reference throughout is MLX at 5.0 fp16 TFLOPS, measured in step 1.

## Headline

Reduction-axis tiling is worth more than quantization, and the two compose.

Eight chained 4096x4096 linears at M=1024, which is the shape a transformer MLP
stack presents:

| reduction tile kt | fp16 | int8 |
|---|---|---|
| 4096 (untiled) | 6.89 | 10.54 |
| 2048 | 11.07 | 12.83 |
| 1024 | 12.19 | 13.59 |

Tiling alone takes fp16 from 6.89 to 12.19 TFLOPS, a factor of 1.77, at cosine
0.999995 against an fp32 reference. That is 2.4 times the GPU with no numerical
compromise at all. Adding int8 on top buys a further 1.11x and costs an order of
magnitude in accuracy, cosine 0.99975.

## Is int8 native or dequantized?

Native, or close enough that it does not matter. ANEForge's docs say int8 and
int4 weights "both dequantize on-device," which predicts no gain at
compute-bound M. The measurement disagrees.

Eight chained layers, untiled, int8 over fp16:

| M | fp16 | int8 | ratio |
|---|---|---|---|
| 128 | 7.27 | 13.59 | 1.87 |
| 512 | 6.13 | 10.28 | 1.68 |
| 1024 | 6.82 | 11.48 | 1.68 |
| 2048 | 7.29 | 11.84 | 1.62 |
| 4096 | 7.49 | 12.13 | 1.62 |

At M=4096 the weight stream is under 3 GB/s, nowhere near the 58 GB/s the engine
sustains, so nothing here is bandwidth-bound. A pure dequantize-to-fp16 path
would sit at 1.0. Core ML agrees independently: its W8A8 path reaches 12.72 TOPS
against 6.81 fp16 at the same shape, 1.87x. Two unrelated dispatch paths giving
the same ratio rules out an artifact of either one.

int4 is not a further win. At one layer it peaks at 8.40 TOPS against int8's
8.21 and carries 25 to 70 percent run-to-run variance, plus a 57 second compile.
Treat int4 as a memory-footprint option, not a throughput one.

## The tiling result, and a correction

The starting hypothesis was a 32 MB SRAM cliff, since a 4096-square fp16 weight
is exactly 32 MB. That is not what the hardware does.

My first attempt at this test was wrong and I am recording it because the error
is instructive. Chunking only the output axis leaves each weight tile at
K by chunk, which for K=16384 is 134 MB however narrow the chunk. Unsurprisingly
it changed nothing: 1.92 TOPS at 4096-wide chunks against 1.94 monolithic. That
result says nothing about residency. The real test splits the reduction axis too,
so `tile2d.py` splits both and sums the partial products.

With both axes split, a 16384x16384 GEMM at M=1024:

| kt | nt | tile MB | TFLOPS |
|---|---|---|---|
| 16384 | 16384 | 537 | 1.94 |
| 8192 | 8192 | 134 | 3.64 |
| 4096 | 4096 | 33.6 | 6.45 |
| 2048 | 2048 | 8.4 | 9.86 |
| 1024 | 1024 | 2.1 | 9.01 |
| 512 | 512 | 0.5 | 8.83 |

Five times the throughput from restructuring the same arithmetic. There is no
cliff at 32 MB. The curve is smooth and peaks around kt=2048.

### What the constraint actually is

Two tiles of identical size settle it:

| kt | nt | tile MB | TFLOPS |
|---|---|---|---|
| 1024 | 4096 | 8.4 | 9.01 |
| 4096 | 1024 | 8.4 | 6.47 |

Same bytes, same tile count, 39 percent apart. Throughput tracks the reduction
axis width and is nearly indifferent to the output axis. A 2048x4096 tile at
16.8 MB reaches 9.71, beating every square tile except 2048.

So the rule for step 2b is to keep kt at or below 2048 and let nt be whatever is
convenient. The earlier "efficiency collapses past 4096-wide weights" finding
from step 1 was really the reduction axis growing past 2048, not a weight-size
limit. Core ML happened to leave 1.77x on the table because it never split K.

This also holds at K=4096, where there is no size pressure at all. One layer,
M=1024: fp16 goes from 5.39 untiled to 8.13 at kt=2048. The gain is not about
fitting anything, it is about the shape of the reduction.

## Dispatch floor

A trivial one-op program, 40 runs:

| program | median | min |
|---|---|---|
| 1x1 linear | 194 us | 113 us |
| 8x8 linear | 129 us | 104 us |
| 32x32 linear | 335 us | 120 us |

The floor is about 105 us at best, against the 70 to 100 us the ANEForge paper
predicts. Medians run higher and noisier than minima because the fixed cost is a
firmware round trip that does not pipeline. Budget 150 us per dispatch when
planning chunk sizes, not 70.

## Costs

Tiling is not free at compile time. The 16384 GEMM at kt=1024 builds 753 ops in
25 seconds against 26 seconds for the single-op monolithic version, so build time
is roughly flat there. But the op count grows as K/kt times N/nt, and the 512
tile case reached 3041 ops. Compilation is one-time and cacheable, so this is a
first-run cost per machine rather than a per-run cost.

One compile failed outright, a column-chunked 8192 case at chunk 2048:
`ane_e5rt_program_compile failed (mask=0x4)`. The 2D-tiled equivalents all
compiled, so this looks like a limit on that specific concat shape rather than
anything fundamental. Logged, not yet chased.

## Accuracy

Everything checks against an fp32 numpy reference.

| configuration | cosine | max relative error |
|---|---|---|
| fp16, 8 layers, kt=1024 | 0.999995 | 0.0024 |
| int8, 8 layers, kt=1024 | 0.999749 | 0.0225 |

Both clear the cosine > 0.999 bar. The int8 max relative error near 2 percent is
worth watching once it sits under a softmax.

## What this changes for step 2b

The MLP should be built from kt <= 2048 reduction tiles regardless of precision.
fp16 at 12.19 TFLOPS already beats the 6 TFLOPS target by a wide margin, so the
int8 decision can be made on accuracy grounds rather than speed. Given that W8A8
blew up on a real block in step 1b, starting 2b in fp16 with tiling is both
faster to get right and only 11 percent slower.
