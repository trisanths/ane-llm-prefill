# Item 3: separating reduction width from weight volume

The question this settles: step 2a found throughput tracks the reduction tile
width, and step 2b's addendum found the W8A16 speedup collapses once the
reduction is tiled. Both were consistent with one shared bottleneck, but in every
shape measured so far, reduction width and weight bytes moved together. Nothing
proved which one the hardware cares about. I said upstream reporting should wait
for a separating experiment. This is it.

## 3a: hold weight volume and FLOPs fixed, vary K by 16x

K times N is held at 16.7M, so every row bakes exactly 33.6 MB of fp16 weights
and does exactly the same arithmetic. M=1024. Only the shape of the contraction
differs.

| K | N | fp16 | W8A16 | W8A16 gain |
|---|---|---|---|---|
| 1024 | 16384 | 5.09 | 6.55 | 1.29x |
| 2048 | 8192 | 7.64 | 8.39 | 1.10x |
| 4096 | 4096 | 5.72 | 8.54 | 1.49x |
| 8192 | 2048 | 3.26 | 4.44 | 1.36x |
| 16384 | 1024 | 1.73 | 1.75 | 1.01x |

K=512 could not run: N would be 32768, past the family 3 limit of 16384 per
dimension.

**Reduction width is the bottleneck, not weight volume.** fp16 throughput swings
4.4x, from 7.64 down to 1.73, while the weight bytes never change. That settles
the 2a law: it is genuinely about the contraction, and the earlier reading of a
"32 MB residency cliff" was wrong for the right reason.

**The int8 gain is not a flat multiply rate.** It ranges from 1.01x to 1.49x
across these rows and is not monotonic in K. A native 2x integer rate would be
independent of contraction shape and would show as a flat line at 2.0. It does
not. Most tellingly, at K=16384 int8 buys nothing at all (1.01x), which no
multiply-rate story explains.

## 3b: fixed volume, sweep M at narrow versus wide K

Same 33.6 MB in both columns, narrow means K=1024 with N=16384, wide means
K=4096 with N=4096.

| M | K=1024 fp16 | W8A16 | K=4096 fp16 | W8A16 | narrow/wide, fp16 |
|---|---|---|---|---|---|
| 128 | 3.15 | 4.54 | 4.93 | 7.67 | 0.64x |
| 256 | 6.47 | 7.30 | 4.18 | 6.20 | 1.55x |
| 512 | 6.92 | 7.13 | 5.12 | 7.48 | 1.35x |
| 1024 | 6.23 | 6.56 | 5.75 | 8.44 | 1.08x |
| 2048 | 6.96 | 7.07 | 5.82 | 8.45 | 1.20x |

The narrow reduction wins at every M from 256 up, by 8 to 55 percent. At M=128
it loses, because that size is dominated by per-dispatch cost rather than by the
contraction.

The W8A16 columns separate cleanly here. At narrow K it buys almost nothing once
past M=256: 6.23 to 6.56 at M=1024, about 5 percent. At wide K it buys 45 to 50
percent consistently. That is the mechanism showing itself.

## What can now be said upstream

Defensible, on this evidence:

- Throughput on this engine is governed by the reduction-axis width, not by the
  weight bytes resident. Holding bytes and FLOPs fixed and varying K alone swings
  throughput 4.4x, peaking near K=2048.
- The int8 speedup is conditional on K. It is worth roughly 1.5x at K=4096,
  about 1.05x at K=1024, and nothing at K=16384. It is not a native 2x integer
  multiply rate.
- Both findings are consistent with ANEForge's documentation that int8 weights
  dequantize on-device. Halving the bytes pulled per contraction step helps
  exactly where the contraction is the constraint, and stops helping when it is
  not, or when something else saturates entirely.

Not established, and worth saying so: why K=16384 collapses for both precisions
alike. Something other than weight bandwidth saturates there, and this experiment
does not identify it.

The earlier claim that measured int8 behavior "contradicts the docs" should not
be filed. What is worth reporting instead is the reduction-width law itself,
which is undocumented, actionable, and worth 1.5 to 1.8x on real workloads.

## Caveat on variance

At M=1024 and K=1024 the two experiments disagree on the int8 ratio, 1.29x in 3a
against 1.05x in 3b for the same shape. Run-to-run variation at this size is
real. The pattern across K holds in both, but individual ratios should be read as
plus or minus 0.2.
