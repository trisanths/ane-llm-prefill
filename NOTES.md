# ANE prefill engine — working notes

Running log of design decisions, version pins and surprises. Newest section last.

## Machine and version pins

ANEForge is version-gated, so these are the pins that matter for reproducing anything here.

| Component | Version |
|---|---|
| Chip | Apple M2 Pro, 16-core GPU, 16 GB unified |
| macOS | 27.0, build 26A5425a |
| ANECompiler.framework | 10.26.6 |
| ANEServices.framework | 10.19 |
| aneforge | 0.4.0 (pip) |
| coremltools | 9.0 |
| MLX | 0.32.2 |
| Python | 3.11 (uv venv at .venv) |

`af.device_status("linear")` returns `native`, so the linear op is running on the
engine rather than a bridge path.

## Step 1 (done): Core ML roofline

Chained fp16 linears, weights 4096x4096, placement confirmed per op through
MLComputePlan rather than powermetrics, because powermetrics needs a sudo
password nobody can type from an agent session.

- ANE peaks at 7.2 fp16 TFLOPS with 16 chained layers, 6.9 with 8.
- MLX on the GPU peaks at 5.0. This is the 16-core M2 Pro, and Apple GPUs run
  fp16 matmul near the fp32 rate, so 5.0 is close to its real ceiling. The
  13 TFLOPS figure from the original plan never applied to this machine.
- Weight streaming: ANE 58 GB/s, GPU 156 GB/s. Decode stays on the GPU.
- Throughput falls off past 4096-wide weights: 3.7 TFLOPS at 4096 square,
  3.05 at 8192, 1.87 at 16384. Chaining 4 layers of 8192-square only reached
  3.63, so the falloff is not per-call overhead.

### Surprise 1: the ANE is the faster GEMM engine, not the smaller one

The whole original plan assumed the ANE was a second, smaller accelerator and
that the interesting number would be concurrency. On large fused GEMM it simply
beats the GPU by about 1.4x at fp16. That reordered the entire project.

### Surprise 2: Core ML will not put a single M=1 linear on the ANE

A one-layer model at M=1 was placed on the CPU. Chain 8 of them and all 8 go to
the ANE. Placement depends on total program weight, not op type. This is
exactly the "placement roulette" that motivates dispatching through ANEForge.

## Step 1b (done): int8 through Core ML

Before switching substrate I ran the int8 question through Core ML, because the
answer gates which model sizes are worth targeting.

8 chained 4096x4096 layers, medians over 10 iterations:

| M | fp16 | int8 weights, fp16 activations | int8 weights + int8 activations |
|---|---|---|---|
| 128 | 6.39 | 11.34 | 7.12 |
| 1024 | 6.29 | 10.19 | 11.55 |
| 4096 | 6.81 | 10.89 | 12.72 |

### Surprise 3: int8 is not dequant-only, contradicting the ANEForge docs

ANEForge's docs/llm.md says int8 and int4 weights "both dequantize on-device."
If that were the whole story, throughput at compute-bound M would be flat
against fp16 and only the bandwidth-bound small-M end would improve. Instead
W8A8 reaches 12.72 TOPS against 6.81 TFLOPS at M=4096, where weight bandwidth
is only 3.1 GB/s and nothing is bandwidth-bound. 1.87x at the compute-bound end
is the signature of a native int8 rate, not of dequantization.

Reproduced through ANEForge at M=512, 2 layers: fp16 4.50, int8 8.33 TOPS, so
1.85x on a completely different dispatch path. Two substrates agreeing rules
out a Core ML artifact.

### Surprise 4: W8A8 destroys accuracy on a real transformer block

Chained plain linears quantize fine (max relative error about 1.6 percent). The
same W8A8 recipe on a full attention-plus-MLP block gives 45 percent max
relative error. Plain int8 weights with fp16 activations stay accurate. The
likely culprit is calibrating activation scales on Gaussian noise rather than
real activations, which mis-scales the attention logits and the SwiGLU product.
This needs real calibration data before any W8A8 number is trustworthy at the
model level. Weight-only int8 remains safe and still gives about 1.6x.

## Step 2 decisions

Asahi dual-boot dropped. ANEForge reaches the engine directly on macOS, which
removes the only reason to go under Core ML.

Substrate switched to ANEForge 0.4.0 for all step 2 work. Core ML stays as a
cross-check oracle, not as the execution path.

Dispatch floor measured at 104 to 130 microseconds for a trivial one-op program,
against the roughly 70 to 100 microseconds the paper predicts. Variance is high
because the floor is dominated by a firmware round trip that does not pipeline.

## Open blocker: power counters

`powermetrics --samplers ane_power` still needs a sudo password, which an agent
session cannot supply. `power_capture.sh` is ready to run. Until someone runs it
by hand, every ANE power claim in this repo is unmeasured, and placement
evidence comes from the compute plan API and from ANEForge dispatching directly.

## Reference reading, for step 2b

Techniques worth applying when the fused block gets written.

**RMSNorm via LayerNorm** (ANEMLL, via the CoreML-LLM survey). The engine has
fused LayerNorm but no RMSNorm. Concatenating `[x, -x]` along the last axis
forces the mean to zero, so LayerNorm over the doubled width computes exactly
RMSNorm; slice the first half back out and scale. ANEForge exposes `rms_norm`
directly, so check whether it already lowers to this or to a slower decomposition.

**Prefill bypass** (AFM/SwiftKV). Late layers that only consume KV and never
produce it can be skipped for every prompt token except the last. The survey
measures the skipped chunks at about 47 percent of per-token prefill time, and
reports 870 ms down to 460 ms on an 8K prompt. This is orthogonal to raw
throughput and is probably the single largest TTFT win available.

**Ping-pong buffers** (ANEMLL). The engine runs asynchronously, so reusing one
buffer for hidden state between chunks races. Alternate two buffers, or index a
16-deep ring by token counter.

**Avoid repeat_interleave** for GQA head expansion; use a broadcast matmul.

**ane-infer** reports 3.6 TFLOPS from fusing an 8-op FFN into one program,
against 1.1 TFLOPS per single-op dispatch. Worth noting that the fp16 chained
linears here already reach 6.9 through Core ML and the ANEForge path holds a
similar rate, so the fusion win they describe is mostly about escaping per-op
dispatch rather than a hardware ceiling.

**Orion** (arXiv 2603.06728, Ramchand Kumaresan) catalogs 20 constraints on MIL
IR programs, memory layout, compilation limits and numerical behavior, 14 of
them previously undocumented. Read the full text before debugging any fallback.

## Step 2a (done): ANEForge roofline

Full writeup in results/roofline_aneforge.md. Three things changed my model of
the hardware.

### Surprise 5: my first tiling test was wrong, and the right one overturned the premise

Chunking only the output axis leaves each weight tile at K by chunk width, which
for K=16384 is 134 MB no matter how narrow the chunk. It measured nothing and I
nearly reported it as evidence against the 32 MB hypothesis. Splitting both axes
gives 1.94 TOPS monolithic against 9.86 at 2048-square tiles, a factor of five on
identical arithmetic.

### Surprise 6: the constraint is the reduction axis, not tile bytes

A 1024x4096 tile and a 4096x1024 tile are both 8.4 MB. They run at 9.01 and 6.47
TOPS. Throughput tracks kt and barely notices nt. So there is no SRAM residency
cliff, there is a preferred contraction width, and it sits at about 2048. The
step 1 finding that "efficiency collapses past 4096-wide weights" was really K
growing past 2048 the whole time. Core ML never split K, which is where its
missing 1.77x went.

### Surprise 7: tiling beats quantization, and they compose

Eight chained 4096-square layers at M=1024: fp16 untiled 6.89, fp16 at kt=1024
12.19, int8 at kt=1024 13.59 TOPS. Tiling buys 1.77x at cosine 0.999995. int8 on
top buys a further 1.11x and costs a decimal place of accuracy. That inverts the
plan: fp16 plus tiling is the safe default and int8 is now an optional 11 percent.

### Decision: start 2b in fp16 with kt <= 2048

Rationale is that 12.19 TFLOPS already clears the 6 TFLOPS target for the MLP,
W8A8 destroyed a real block in step 1b, and debugging a fused block is easier
without quantization error in the loop. int8 can be layered on afterwards as a
measured delta.

### Dispatch floor

105 microseconds best case, 130 typical, against the 70 to 100 the paper predicts.
Budget 150 microseconds per dispatch when sizing prefill chunks.

## Step 2b (done): one fused transformer block

Full writeup in results/block_bench.md. Qwen3-14B shapes, d=5120, 40 query heads
over 8 KV heads, ff=17408.

MLP runs at 9.04 TFLOPS against a 6 TFLOPS target. Full block 7.75 fp16, 9.55
int8. The ANE beats the GPU by 1.53x to 1.60x across M=256 to 1024 on the same
block. Cosine against MLX fp16 is 0.999981.

### Surprise 8: attention does not fall off the engine, but the native op is broken

This was the question the step existed to answer. The decomposed matmul, softmax,
matmul path runs entirely on the ANE in one program. What fails is Apple's own
fused-attention layer: af.sdpa compiles and then the driver rejects the bundle at
dispatch with "verifyBundleAtPath: invalid model" for M of 128, 256 and 384, and
ANEForge refuses upfront at 512 and above.

That cost me a bisect, because af.sdpa is a graph cut. af.compile succeeds and
the segment is compiled lazily on first run, so compiling alone showed every
stage passing and only execution revealed the failure. Lesson: when a graph has
a cut in it, compile is not a test.

Since the decomposed path works, nothing here argues for going under the driver.

### Surprise 9: attention is 6.5x less efficient per FLOP than the MLP

Attention core alone is 1.38 TFLOPS against the MLP's 9.04. It is only 3.7
percent of block FLOPs but 17 percent of block time. At M=1024 that is tolerable,
but the term is quadratic, so it sets the ceiling on prompt chunk size in 2c.
Attention is now the thing worth optimizing, not the MLP.

### Surprise 10: the ANEMLL LayerNorm trick is a pessimization here

concat([x,-x]) plus LayerNorm exists because Core ML has no fused RMSNorm.
ANEForge has one natively, and it is 5 percent faster than the trick, 90.1 ms
against 94.6. Ported workarounds need re-measuring on a new substrate.

### Family 3 dimension cap, and why the fix was free

Every tensor dimension is capped at 16384 and ff=17408 exceeds it, so the SwiGLU
intermediate is rejected. Chunking the MLP along ff removes the wide intermediate
and simultaneously drops each down-projection's contraction from 17408 to 2176,
which the 2a tiling law wanted anyway. The constraint pushed the code toward the
faster shape.

### Warned but unreproduced: Q.4 crop-DMA saturation

ANEForge warns that on pre-A16 hardware a last-axis slice at a non-zero offset
saturates values above 4094 to infinity. All reduction-axis tiling is built on
exactly those slices, so this would be a landmine. Feeding magnitudes of 20192,
five times the threshold, produced 0.0005 relative error and no infinities.
Recorded as unreproduced; retest against real model activations before trusting
tiling at scale.

### Open question for 2c

int8 weights with fp16 activations give 1.23x on the block at cosine 0.999957,
which is nothing like the W8A8 failure in step 1b. Enable it after the hybrid
loop is correct, not before.

## Power, measured at last

First loaded ANE power capture, 300 ms sampling across a 7.2 s window while the
8-layer tiled fp16 chain ran at 12.24 TFLOPS.

| rail | idle | loaded median | loaded peak |
|---|---|---|---|
| ANE | 0 mW | 8490 mW | 8904 mW |
| GPU | 35 mW | 35 mW | 328 mW |
| CPU | - | 12821 mW | 19573 mW |

That works out to 1.44 TFLOPS per watt on the ANE rail.

Two things this settles. The GPU rail never left idle, which is independent
confirmation that the work ran on the Neural Engine and not silently on the GPU;
until now that claim rested on the compute plan API and on ANEForge's dispatch
path. And the 13 samples reading 0 mW are the compile phase, where the engine is
genuinely idle while the CPU rail sits near 13 W doing the compilation.

The CPU number is worth remembering: compiling a tiled program is expensive, and
during 2a's larger builds the machine is burning more power on compilation than
the ANE ever draws running the result.

Superseded below: the GPU-side capture was attempted and the GPU rail proved unreliable. See "Power comparison".

Still missing is a GPU-side capture of the same arithmetic, needed before any
performance-per-watt comparison between the two engines.

## Step 2c: blocked on model architecture, pipeline validated on a substitute

### Surprise 11: ANEForge cannot load Qwen3.5 at all

`aneforge.llm.PREFILL_MIXERS` contains exactly one entry, `attention`, and
`ModelType` tops out at `QWEN` (the dense Qwen2/Qwen3 family). Qwen3.5-2B is a
hybrid: `full_attention_interval` of 4 gives 18 `linear_attention` layers, which
are Gated DeltaNet, interleaved with 6 real attention layers. There is no mixer
for the linear-attention layers, so `af.load_llm` rejects it rather than
mis-loading it, which is the documented and correct behavior.

Qwen3.5-2B is also multimodal (`Qwen3_5ForConditionalGeneration`, with a nested
`text_config` and an image token), so even the config shape differs from what the
loader expects.

Writing a Gated DeltaNet prefill mixer is possible but it is a project, not a
step: the recurrence is sequential, which is the opposite of what this engine is
good at, and it would have to be validated against the reference implementation
before any hybrid number meant anything.

### Validated instead: the dense path works end to end

Qwen3-0.6B (dense, 28 layers, 16 heads over 8 KV heads, head_dim 128) loads and
runs. "The capital of France is" gives " Paris" at logit 17.59, with the next
candidate at 14.35, so the ANE prefill path is numerically sound on a real model
and not just on synthetic weights.

### The KV bridge exists and is reachable

`LlamaPrefill._prefill_seed(token_ids, pad_to=...)` returns
`(next_logits, kv_by_layer)`, where `kv_by_layer[li]` is a tuple of K and V as
fp16 numpy in decode-cache layout, already trimmed to the real (unpadded)
positions. `_seed_cache` is the reverse, writing those buffers back into the
resident decode cache.

That is exactly the handoff 2c needs. The ANE produces per-layer K and V as host
numpy, and MLX can adopt them directly. No reverse engineering required, though
both are private methods so the API may move between ANEForge versions.

`pad_to` matters for the hybrid design: it right-pads a prompt to a fixed bucket
so one compiled prefiller is reused across prompts of different length. Given
that compiles cost seconds, bucketing is mandatory, not an optimization.

### Disk

Today's sweeps left 18.3 GB of ANEForge compile cache and took the volume to
99 percent full. Cleared; it regenerates on demand. Worth watching, because a
per-length prefill bucket is a separate compiled program and they accumulate fast.

## Additions to 2b: per-layer drift, W8A16, and a walked-back claim

### Naming fixed

ANEForge's quantizer is weight-only, so `compress="int8"` is W8A16: int8 weights,
fp16 activations. Earlier sections called it "int8", which collided with the Core
ML W8A8 path that failed in step 1b. Different mechanisms, now named apart.

### Surprise 12: tiling is free in accuracy terms

fp16 per-layer cosine at kt=1024 and kt=2048 is identical to six decimals at every
depth. Reduction tiling changes summation order and nothing else that matters. A
1.77x speedup at zero numerical cost is the cleanest thing in this project.

### Surprise 13: single-block cosine hides depth risk

W8A16 drifts about 4.5x faster than fp16, roughly 3.8e-5 of cosine per layer
against 8e-6. One block reads 0.99996 and looks like ample headroom. Extrapolated,
W8A16 crosses the 0.999 bar around 26 to 31 layers. A per-block accuracy check is
not a model-level accuracy check, and this only showed up because the per-layer
measurement was asked for.

### Surprise 14: I overclaimed on native int8, and the tiled data walks it back

Untiled, W8A16 gives 1.62x to 1.87x across M including compute-bound M, which is
what made "native, not dequantized" look right. Tiled, it gives 1.12x at kt=1024
and 1.16x at kt=2048, and on the real block only 1.07x to 1.20x.

A genuine 2x integer multiply rate would be orthogonal to how the contraction is
split. It is not. Tiling and W8A16 buy nearly the same speedup and do not stack,
which says they relieve one shared bottleneck: the cost of a wide reduction.
Dequantizing int8 weights halves the bytes pulled per contraction step, which is
entirely consistent with ANEForge's documentation.

Do not report the contradiction upstream as it stood. The separating experiment
is a kernel with a narrow reduction and large weight volume, which has not been
run.

### Surprise 15: a failed compile poisons the cache

A compile that fails under memory pressure leaves a cache entry keyed by the
graph, and the same graph then fails from any fresh process. It is
indistinguishable from a deterministic structural limit, and it sent me chasing a
non-existent "kt=2048 is unsupported" constraint. Clearing
~/Models/.aneforge-cache fixes it immediately.

The memory pressure behind it is real: each block bakes about 600 MB, mostly the
three 17408-wide MLP matrices, and an 8-layer fp16 stack needs roughly 10 GB
alongside the numpy weights. It does not fit in 16 GB; the W8A16 version does.
Compile stacks in separate processes.

Rule: suspect the cache before believing a hardware limit.

## Power comparison: resolved, ANE is 6x more efficient

Full writeup in results/power.md.

| | ANE | GPU (MLX) |
|---|---|---|
| throughput | 12.16 TFLOPS | 4.94 TFLOPS |
| loaded median | 8.70 W | 21.76 W |
| efficiency | 1.40 TFLOPS/W | 0.23 TFLOPS/W |

2.5x the work at 2.5x less power, so about 6x the efficiency per joule. On a
laptop that doubles the prefill argument, because sustained GPU prefill is what
heats the machine.

### Surprise 16, and its correction

The first attempt caught only 5 GPU samples above 3 W across 285 s against an
8.9 s benchmark, including a 678 mW reading wedged between 24 W and 46 W
samples. I concluded the rail was dropping samples on this beta build and
refused to publish a ratio from it.

Driving the GPU continuously for 60 s shows the rail is fine: 379 of 500 samples
above 3 W, 99 percent inside the sustained window, median 21.8 W. So the "broken
rail" reading was wrong.

What is still unexplained is why 8.9 s of benchmark registered as 1.5 s of load.
Startup, weight generation, teardown and clock ramp do not fully cover the gap.
The durable lesson is procedural: measure power against a sustained load, and
read a sparse loaded-sample count as a reason to distrust the capture rather
than the hardware.

Refusing to publish the 7x figure from 5 samples was right; the sustained
measurement lands at 6.2x, close to it, but for defensible reasons now.

### Asymmetry worth closing

The GPU number has 379 loaded samples. The ANE number has 12, tightly clustered
between 8.16 and 8.92 W with a 0 mW idle floor and independent corroboration from
the combined rail, but still 12. A sustained ANE capture would tighten it.

## Item 1: attention

Full writeup in results/attention.md.

Key-axis blocking works and the win grows with M: 1.15x at M=512, 1.22x at
M=1024, 1.55x at M=2048, where it also cuts the largest intermediate from 335 MB
to 42 MB. Best block size is the one that lands the score tensor near 20 to 42
MB, so B=512 at M=1024 and B=256 at M=2048. Smaller is not better; op overhead
eats it below that.

### Surprise 17: the online-softmax form does not compile, for a specific reason

Flash-style running max plus per-block rescale fails on this ANE compiler. Every
individual step compiles. The failure appears only where the final normalization
joins the accumulated output to the accumulated denominator. Either accumulator
combined with a shallow operand is fine; the two combined with each other fail,
by divide, multiply or add alike. It is a shared-deep-ancestor diamond, not an
operator.

Dropping the running max compiles at every block size, at the cost of
exponentiating uncentred scores. Measured scores span -5.4 to 5.1 here so the
largest exponential is 169 against an fp16 limit of 65504, but this kernel is not
safe for unbounded scores.

### Surprise 18: attention on the GPU is no faster than on the ANE

The per-layer split loses by 5 percent. Transfer is not why: moving q, k, v out
and context back costs 1.48 ms at M=1024, 2 percent of the total. The GPU simply
takes 14.92 ms against the blocked ANE kernel's 12.95 ms. Both engines run
attention near 1.4 TFLOPS.

So attention stays on the ANE, but for a new reason: not because the GPU is
unavailable, because it is no better.

The cheap transfer is the more useful result. It means the 2c KV bridge will not
be transfer-bound.

## Item 2: weight residency

### No zero-copy sharing with MLX, and weights are held once per engine

ANEForge bakes weights into the compiled program. They survive deleting the host
numpy array, so at steady state a weight exists exactly once per ANE program, not
twice. There is no API to point a program at an externally allocated buffer.

`Model.input_view()` returns a writable numpy view onto the program's own input
buffer, which removes one copy on the activation path: 5.27 ms against 5.99 ms
for passing numpy at M=1024, about 12 percent. It is ANEForge's buffer, though,
not a shared one, so this is copy avoidance rather than aliasing.

MLX exposes `__dlpack__` but not `__array_interface__`, and its arrays live in
Metal allocations. Nothing at the Python level lets the two engines name the same
IOSurface. So a hybrid that runs prefill on the ANE and decode in MLX holds the
weights twice, once baked per engine.

Copies themselves are cheap and not worth engineering around: 8.4 MB moves in
0.166 ms, about 50 GB/s, and MLX to numpy costs the same 0.167 ms.

Runtime weights are possible. `af.einsum("mk,nk->mn", x, w)` with both as inputs
compiles and runs, which would allow feeding weights from outside. It costs
1.7x: 2.99 TFLOPS against 5.13 baked. Not worth it to save a copy, and it still
would not alias MLX memory.

### int8 storage with fp16 compute already exists

This is what ANEForge's `compress="int8"` does, and the disk numbers confirm it
is real storage compression, not a compute-time trick. A 4096-square fp16 weight
produces a 33.6 MB program, exactly its fp16 size. The same weight at
`compress="int8"` produces 16.8 MB, exactly half. Activations stay fp16
throughout, which is why this is W8A16.

So no work is needed here. int4 is available the same way and would quarter it,
though step 2a found int4 slower than int8 and highly variable.

### Largest dense model per RAM tier

Assumes 4 GB for OS and runtime, 1 GB of KV cache, weights dominating. Hybrid
means an ANE prefill copy plus an MLX decode copy.

| RAM | usable | ANE-only int8 | ANE-only fp16 | hybrid int8+int8 | hybrid fp16+int8 |
|---|---|---|---|---|---|
| 16 GB | 12 GB | 8B (9.0) | 4B (9.0) | 4B (9.0) | 1.7B (6.1) |
| 32 GB | 28 GB | 14B (15.0) | 8B (17.0) | 8B (17.0) | 8B (25.0) |
| 48 GB | 44 GB | 32B (33.0) | 14B (29.0) | 14B (29.0) | 14B (43.0) |
| 64 GB | 60 GB | 32B (33.0) | 14B (29.0) | 14B (29.0) | 14B (43.0) |

Weights alone, in GB: 0.6B is 1.2/0.6/0.3 at fp16/int8/int4; 1.7B is 3.4/1.7/0.8;
4B is 8/4/2; 8B is 16/8/4; 14B is 28/14/7; 32B is 64/32/16.

Two things this decides. The hybrid design costs a full model tier, because
holding weights twice on a 16 GB machine drops the ceiling from 8B to 4B. And
above 48 GB nothing changes without a 70B-class dense model in the list, since
32B int8 already fits at 48.

The practical reading for this machine, at 16 GB: 4B int8 hybrid, or 8B int8 if
decode also runs on the ANE. That matches the 2c recommendation of Qwen3-4B.

## Item 3: reduction law separated

Full writeup in results/reduction_law.md.

Holding weight volume and FLOPs fixed (K times N constant at 33.6 MB) and varying
K by 16x swings fp16 throughput 4.4x, from 7.64 TFLOPS at K=2048 down to 1.73 at
K=16384. That settles it: the reduction width is the bottleneck, not the weight
bytes resident.

The W8A16 gain is conditional on K, ranging 1.01x to 1.49x and not monotonic. At
K=16384 it buys nothing at all. A native 2x integer rate would be flat across
contraction shapes. It is not, so the walk-back stands and the "contradicts the
docs" claim should not be filed.

What is worth reporting upstream is the reduction-width law itself, which is
undocumented and worth 1.5 to 1.8x.

Unexplained: why K=16384 collapses for both precisions alike.

## Item 4: end-to-end hybrid

Full writeup in results/hybrid.md. Qwen3-0.6B dense, int8, 512-token bucket.

### Surprise 19: prefill on the ANE is free during GPU decode

Decode alone 72.58 tok/s; decode while the ANE prefills a second request 72.45
tok/s. 99.8 percent retention. Eight full 512-token prefills completed during a
48-token decode, so about 4096 prompt tokens were processed at no measurable cost
to interactive latency.

This is the architectural payoff and it is larger than the prefill speedup
itself. The engines contend for almost nothing.

### Prefill speedup is only 1.19x at this model size

ANE 0.110 s against MLX 0.131 s for 512 tokens. Far below the 2.5x on raw chained
GEMM, for two structural reasons: Qwen3-0.6B has a hidden size of 1024 so every
reduction is already narrow and item 3's advantage has nothing to bite on, and the
ANE pays for the full bucket while MLX prefills the real length.

Bucket granularity is the biggest TTFT lever for short prompts: a 128-token
prompt in a 512 bucket gets 1186 tok/s effective, against 4855 for a prompt that
fills its bucket.

### The bridge costs 4 to 6 ms

For 15 to 59 MB of KV. Consistent with item 2's 50 GB/s copy rate and with item
1b's prediction. Not transfer-bound.

### Surprise 20: the size ceiling was disk, misreported as a compiler error

Qwen3-1.7B prefill programs fail with `mask=0x4`, which reads as a compiler
limit. One attempt surfaced the truth: `[Errno 28] No space left on device`. Each
bucket is a separate program holding the full model's baked weights, so a few
attempts consumed 9.5 GB of cache on a volume already at 97 percent, and the
cache-poisoning bug then made the failures permanent.

After clearing the cache with 21 GB free, 1.7B still failed at the same graph
hash while consuming 7 GB. So there are two compounding limits and this machine
cannot separate them. 0.6B works reliably in both precisions.

## Item 5: issues drafted

issues/01-sdpa-dispatch-rejection.md and issues/02-failed-compile-poisons-cache.md.
Both written against aneforge 0.4.0 with versions pinned, minimal reproductions,
and what was ruled out. Not filed: this environment has no authenticated GitHub
access, so they need a human to post.

## Item 6: still needs a password

`./power_ane.sh` is ready and dry-run at 12.18 TFLOPS sustained. It is the last
open measurement, matching the GPU's 379 loaded samples against the ANE's current 12.

## Round 4: chunked prefill, segmentation, and a headline walked back

### Surprise 21: the concurrency headline was measured wrong, 87 percent not 99.8

Three faults compounded. The window was 0.66 s, too short for contention to
appear. The worker saturated the ANE with no pause, which at 1.7B drove decode
from 35 tok/s to 0.42. And MLX warmup made the uncontended baseline run slower
than the contended one, at one point producing a nonsensical 128 percent
retention.

Interleaving idle and saturated conditions to cancel drift gives paired ratios of
0.880, 0.865, 0.857. Retention is 86 to 87 percent: saturating the ANE costs
about 13 percent of GPU decode. Still a good trade for 26000 prompt tokens, but
not free.

Lesson worth keeping: a throughput ratio measured in one short window, with the
baseline run first, is worth nothing. Interleave and warm.

### Surprise 22: finer layer segmentation is faster

Expected chaining overhead to punish it. Qwen3-1.7B fp16 at 512 tokens: 10
segments of 3 layers gives 0.605 s, 28 segments of 1 layer gives 0.334 s. 1.8x
the wrong way round. Compile time also drops, 85 s to 50 s. Set the budget to
about one layer's weights and stop tuning.

This is what makes 1.7B work at all: the default 1600 MB budget puts all 28
layers in one program, which does not compile.

### Surprise 23: int8 is numerically broken on Qwen3-1.7B

Mean full-model logit cosine against fp16 is 0.019, essentially uncorrelated,
top-1 agreement 1 in 5. "The capital of France is" gives ' Italy'.

Not the predicted depth drift, which would have been about 0.9989 at 28 layers.
Not a segmentation interaction: 0.6B has the same 28 layers and stays correct
under int8 at every granularity from 28 layers per program down to 1. Specific to
the 1.7B weights, cause not yet isolated, so not in the drafted issues.

### Chunked prefill works and is correct

One program, runtime cos/sin and mask, any length up to capacity. Cosine
0.999995 against the bucketed path. Two bugs on the way: rms_norm is 2D only so
per-head q/k norm needs a reshape, and I forgot final_norm before the LM head,
which showed up as cosine 0.23.

Wins where buckets waste work (128-token prompt: 55 ms against 110 ms, 2x) and
loses where the bucket is full (512: 156 ms against 112 ms). Capacity matters as
much as chunk size, because every chunk attends over the full capacity whether or
not it is filled.

The decisive win is compile and disk: one compile of about 20 s serving all
lengths, against one per bucket length. That is the difference between a bounded
cache and one that grows with the prompt-length histogram.

### The memory boundary, for the bigger-Mac decision

1.7B prefill alone on the ANE: works, 303 ms, 1687 tok/s, correct, RSS 5.07 GB,
free RAM 0.07 GB at peak.

1.7B hybrid: does not fit. Weights live in ANEForge host arrays, in 28 baked
segment programs, and in MLX at once. Swap reached 16.05 GB of 17.4 GB and
timings became meaningless.

The blocker is RAM alone. Not compile, not disk, not the ANE.

### Disk

No second volume exists, so the Hugging Face cache cannot be moved off. Home is
379 GB of the user's own data. Added diskguard.py: caps the compile cache by
evicting oldest entries and stamps free space and cache size into every result
row. Also added memguard.py for RSS and swap, after swap silently invalidated a
whole round of measurements.

## Weight sharing: the ANE runtime copies, so two resident copies are unavoidable

Compiled a 0.97 GB baked program and inspected the live process. 29 Malloc Large
regions totalling 928 MB, dirty and resident, one per baked layer. Mapped-file
total is 34.9 MB with 416 KB resident, which is libraries. IOSurface is about
1 MB. `footprint` attributes exactly 928 MB to `neural_peak`.

So the blob is process-private anonymous memory, not a file mapping. There is no
file to mmap twice, and the bytesNoCopy plan has nothing to point at. Physical
footprint grows by the full weight size and stays there after the host arrays are
freed and after repeated runs.

The host copy IS released at compile, which is worth having, but it is replaced
one for one. Writeup in results/weight_sharing.md.

## Ternary Bonsai 8B: what ANEForge can and cannot take

`compress="int4"` is a single 16-entry codebook per weight tensor, not group-wise
fp16 scales. `compress="blockwise"` is group-wise but stores int8 data, with
fp16 scales per [OUT, nblocks] and block_size clamped to a divisor of IN.
Precedence inside the emitter is sparse, then int4-LUT, then per-channel int8,
then fp16, and a mode that misses its error tolerance silently falls back.

There is no API to hand ANEForge packed codes. `weight()` takes an fp16 numpy
array and quantizes it itself, so Bonsai must be dequantized on the host. Doing
that per layer bounds the transient cost to one layer, 386 MB at 8B shapes.

Measured on synthetic ternary tensors: with one global scale, int4 engages at
0.500 bytes/weight and reconstructs exactly. With g128 scales, which is Bonsai's
format, it falls back to 1.001 bytes/weight. Blockwise is 1.063 either way at
cosine 1.000000.

On real Bonsai layer-0 weights, int4 does partially engage at 0.750 bytes/weight
with cosine 0.9989 to 0.9995 per matrix, and blockwise gives 1.063 at cosine
1.000000.

### 8B budget on this box

Layer params are 192.9M, so 36 layers is 6.94B, plus a 621M embed and a 621M
lm_head (not tied).

| ANE mode | bytes/weight | 36 layers | plus fp16 embed for host lookup |
|---|---|---|---|
| int4 | 0.750 | 5.21 GB | 6.45 GB |
| blockwise | 1.063 | 7.38 GB | 8.62 GB |

MLX 2-bit decode measures 2.28 GB resident. So the hybrid is 8.7 GB at int4 and
10.9 GB at blockwise, before Python, MLX runtime and KV.

This machine does not have that. During these runs free RAM sat between 0.09 and
0.3 GB with swap already at 5.3 GB, because other applications hold most of the
16 GB. The practical headroom for this work is 3 to 4 GB, not 12.

### Bonsai MLX decode, measured

Loads in 78 s, 2.3 GB of weights, 2.28 GB resident. Prefill 278.6 tok/s at 512
tokens, decode 44.9 tok/s. Generates correctly: "The capital of France is" gives
"Paris. The capital of Germany is Berlin."

## Direct backend, Experiment 0: what the compiled bundle is, and what can be patched

Full writeup in results/direct_backend_exp0.md.

### Surprise 24: there is no microcode on disk, and the bundle is never reloaded

ANECompiler emits, for a 1-layer 2048x4096 fp16 GEMM, a 2352-byte FlatBuffer
manifest (H14S.e5) and a 129-byte model.anehash. That is the whole "compiled
bundle". The weights stay in ANEForge's own weights.bin, referenced from the MIL
by path and byte offset. No file anywhere holds ANE microcode or descriptors:
the manifest says on-device-compilation = true, ANECompiler.framework is mapped
into our own process, aned stays at 7 MB, and the whole-disk search for .hwx or
any large e5rt artifact found only BNNS CPU programs belonging to other apps.

Every load is a fresh in-process compile from model.mil. Corrupting the .e5 root
offset, truncating it, deleting it, zeroing or deleting the anehash, or deleting
the entire e5bundlecache directory all produced bit-identical output in 0.2 s
and minted a fresh bundle dir alongside the damaged one. The on-disk bundle is
write-only bookkeeping.

### What model.anehash is

Two SHA-256-sized hex strings joined by an underscore. The first is identical
across runs with different weights; the second changes with weights.bin. So it is
program-hash_weights-hash, and the e5rt cache key includes the weights, which is
why each weight patch minted a new bundle dir.

### Patch surface on this build

- weights.bin: honoured bit for bit, no validation. A low-byte flip moved one
  output column by 0.40; setting the fp16 exponent byte to 0x7C produced Inf and
  0x7E produced NaN in that column. All ran to completion, nothing rejected,
  nothing hung. Orion's claim confirmed.
- model.mil: validated at compile. Shape mismatch (2048 to 2047), a transpose
  flag flip that breaks the contraction, and a misaligned blob offset (64 to 66)
  are all rejected in 0.14 s as ane_e5rt_program_compile failed, err=11, before
  anything reaches the engine.
- H14S.e5 and model.anehash: no effect, because they are never read back.
- microcode / descriptors: no on-disk target exists. Patching them means hooking
  ANECompiler's output buffer in memory, which is Experiment 1 territory.

### Recovery

Nothing hung. recover.sh gives submit-with-timeout plus a health probe; verified
that a 5 s timeout kills a hung child and the engine re-probes healthy. The
escalation past process kill is sudo launchctl kickstart -k system/com.apple.aned,
then reboot, documented from the launchd layout rather than from a recovery that
was needed.

Cross-version was not testable with one machine. Structurally, the e5rt cache is
keyed under the OS build (26A5425a) so bundles are build-scoped by construction,
and since they are never reloaded the question is moot on this build anyway.

## Tracks B and C (2026-09-26): decode roofline and attention

### Track B: MLX 2-bit decode is launch-latency bound, not kernel bound
Achievable GPU read BW is 159 GB/s. Bonsai 8B (2.3 GB) decode roofline is 69
tok/s; measured 45 tok/s = 65% of roofline, implied 104 GB/s. Sampling and the
Python token loop are below noise. An isolated 14 MB 2-bit GEMV hits only 50
GB/s but still beats a naive stream of the same bytes (38 GB/s), so MLX's kernel
is well tuned; the loss is ~250 tiny kernel launches per token at M=1. mx.compile
gives nothing on one op (1.01x). A custom Metal GEMV is NOT warranted: the lever
is fewer/larger launches, not a faster kernel. All under swap pressure so numbers
are a floor.

### Track C: attention is bandwidth-bound on the ANE, and the GPU kernel is 2.4x faster
Per-op: full attention core at M=1024 is 15.8 ms, moving 0.346 GB of score
matrix at ~22 GB/s = 7x off the BW roofline. The ANE has no fused attention
(af.sdpa broken), so the H x M x M scores are materialized to and from memory.
Blocked attention cuts that: 1.2x at M=1024, 1.54x at M=2048, peak memory down
8x, cosine 1.0.

WALK-BACK: earlier "GPU no faster at attention" was wrong. That included transfer
+ GQA expansion in the split path. Isolated MLX flash SDPA is 5.46 ms at M=1024
vs ANE blocked 13.07 ms = 2.4x faster, because the GPU has a hardware fused-
attention kernel. Corrected rule: prefill-only on ANE uses blocked attention;
any path where KV is already GPU-resident (i.e. decode) should run attention on
the GPU.

## Track A (2026-09-26): 8B residency and int4 vs blockwise

### Residency: ANE-wired memory is flat with depth
Bonsai-shaped 36-layer compile, 1 layer/prog, 1024-row tiles, int4. Through 18
layers: neural (ANE-wired) footprint FLAT at 182 MB while nominal weights passed
7.3 GB; process RSS 1.8 GB = 0.25x fp16-equiv, non-monotonic (arena reused). The
engine wires ~1 layer at a time, not all of them. Recipe works. Run stalled at
layer 18 because other apps held ~9 GB and free RAM hit 68 MB; killing it dropped
swap 3 GB. Blocker is ambient system RAM, not the ANE. Trigger for 32 GB: free
~4 GB locally or rent.

### int4 vs blockwise: blockwise for Bonsai (exact)
Per-layer cosine of down_proj vs fp16 reference: blockwise = 1.000000 at every
layer (stores g128 int8 + fp16 scales, matches Bonsai's structure exactly); int4
= 0.999 (single 16-entry codebook can't hold g128-scaled values, int8-quality
fallback). int4 is 0.75 B/w (~5.2 GB @ 8B) vs blockwise 1.063 (~7.4 GB). Use
blockwise for correctness; int4 needs the group-wise-LUT feature (issue 03) to be
viable on this format. Cosine run also stalled under swap (12.8 GB) but 3 samples
were unanimous.
