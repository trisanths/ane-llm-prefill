# Reducing the weight memory cost

Short version: **the "copied twice" framing was wrong, and the real cost was
worse and more fixable.** Compiling one large program costs about 4x the weight
size in resident memory. Compiling one layer per program costs 1.0x at the
margin. That is a 3.1x reduction available from a single structural change, and
quantization takes the per-program residency close to zero on top of it.

## What was actually happening

Three things I had wrong, corrected by measurement.

**The ANE daemon does not hold the weights.** `aned` stays at 7.0 MB across
compile, load and run. The memory is in our own process.

**The host numpy copy is genuinely released at compile.** Physical footprint at
0.97 GB of weights: 1080 MB with host arrays live, 928 MB after compile, and
deleting the host arrays changes nothing because they are already gone. So there
was never a host-plus-baked duplication at steady state.

**What I read as a second copy was mostly the compiler's working arena.** The
first large compile allocates a big pool that later compiles reuse. Measuring
marginal rather than total cost separates them:

| program shape | total resident / weight bytes | marginal per additional program |
|---|---|---|
| 24 layers in one program | 4.03x | - |
| 8 layers per program | 4.45x first, then 0.96x | 0.96x |
| 1 layer per program | 1.29x over 24 programs | 1.00x |

One layer per program, 24 programs, 805 MB of weights: 1063 MB resident, of which
296 MB is the one-time arena and the rest is exactly 33.6 MB per program, the
weight size. Compare 3267 MB for the same weights in one 24-layer program.

This also explains why finer segmentation measured *faster* earlier. It is not
only cheaper in memory, it avoids whatever the large-program path does.

## The arena, and how to bound it

The one-time cost is not fixed runtime initialisation. A tiny first compile costs
4 MB. The arena is sized by the largest single weight tensor, at roughly 130 to
160 bytes per element:

| largest tensor | elements | arena |
|---|---|---|
| 512 x 512 | 0.26 M | 55 MB |
| 1024 x 1024 | 1.05 M | 129 MB |
| 512 x 4096 | 2.10 M | 277 MB |
| 2048 x 2048 | 4.19 M | 603 MB |
| 4096 x 4096 | 16.8 M | 1821 MB |

It is reused for tensors of the same element count: after a 1024x4096 tensor,
both 2048x2048 and 4096x1024 added 0.8 MB. So the arena is paid once per distinct
size, not once per weight.

That makes it controllable. Splitting a weight's output axis so no baked tensor
exceeds a chosen element count caps the arena at that size, and every tile after
the first is nearly free. For Bonsai's 12288x4096 MLP matrices, which would
otherwise demand about 6.5 GB of arena, tiling to 1024x4096 caps it near 0.55 GB.

## Quantization nearly eliminates per-program residency

With `compress="int4"`, successive programs add 1 to 4 MB each rather than the
full weight size, at cosine 0.999993 against fp32. Verified with `ps` RSS
independently of `footprint`:

```
prog 0  cosine 0.999993  rss 2698 MB      <- arena
prog 1  cosine 0.999993  rss 2708 MB
prog 5  cosine 0.999993  rss 2690 MB
```

Six 4096-square int4 programs, 201 MB of fp16-equivalent weights, and RSS moves
by single-digit megabytes after the first. fp16 programs cost their full size.
So with int4 the arena dominates, which is exactly the quantity tiling bounds.

## What does not work

`prog.release()` does not return memory. Physical footprint is unchanged by it,
so programs cannot be streamed in and out to bound peak. Whatever is retained
stays for the process lifetime, and `malloc_zone_pressure_relief` does not
recover it either.

Sharing weights with MLX remains impossible. The baked bytes are process-private
anonymous memory, not a file mapping, and there is nothing to point a
`bytesNoCopy` Metal buffer at. That finding stands.

## The recipe

1. **One layer per program.** 3.1x less resident memory than one large program,
   and faster.
2. **int4 where accuracy allows.** Per-program residency drops from the full
   weight size to a few megabytes.
3. **Tile every weight so no baked tensor exceeds about 4 M elements.** This caps
   the compile arena, which becomes the dominant term once int4 is in use.
4. **Do not rely on release().** Size the process for the whole model.

## Honest limits

Applied to Bonsai-shaped layers (d=4096, ff=12288) at 1024-row tiles and int4,
four layers measured 1.42x their fp16-equivalent size. That is far better than
4x but not the near-zero the single-tensor experiments suggested, and the
per-layer deltas were noisy because the allocator returns and reuses memory
unpredictably between compiles. A reliable extrapolation to 36 layers needs more
layers measured than fit comfortably on this machine, and that measurement is
still running at the time of writing.

So the reduction is real and measured at small scale. Whether it brings 8B inside
16 GB is not yet established, and I am not going to claim it is.
