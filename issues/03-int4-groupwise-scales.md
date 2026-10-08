# Feature request: group-wise (g128) scales for the int4 LUT path

**aneforge** 0.4.0 · macOS 27.0 (26A5425a) · Apple M2 Pro (family 3)

## Ask

Let `compress="int4"` carry per-group fp16 scales along the input axis, the way
`compress="blockwise"` already does for int8, instead of a single codebook per
weight tensor. A `group_size` parameter matching `blockwise`'s `block_size` would
be enough.

## Why

Modern on-device checkpoints ship pre-quantized with group-wise scales. The
motivating case is Ternary Bonsai 8B (`prism-ml/Ternary-Bonsai-8B-mlx-2bit`),
distributed as MLX 2-bit with `{"group_size": 128, "bits": 2}`. Its weights are
ternary codes with an fp16 scale and bias per 128 inputs.

That format is a near-perfect fit for a 4-bit LUT: three distinct code values per
group, far inside 16 entries. But because the codebook is per tensor and each
group carries its own scale, the dequantized tensor has thousands of distinct
values, so the LUT cannot represent it and the path silently falls back.

## Measured

Synthetic tensors with exactly this structure, 2048x4096 fp16:

| weight structure | distinct values | int4 result | bytes/weight |
|---|---|---|---|
| ternary, one global scale | 3 | engages, cosine 1.000000 | 0.500 |
| ternary, g128 scales | 7117 | falls back, cosine 0.999993 | 1.001 |
| ternary, g128 scales | 7117 | `blockwise`, cosine 1.000000 | 1.063 |

Real Bonsai layer-0 weights, dequantized from the 2-bit checkpoint:

| matrix | shape | distinct | int4 cosine | blockwise cosine |
|---|---|---|---|---|
| wq | 4096 x 4096 | 719 | 0.998952 | 1.000000 |
| wgate | 12288 x 4096 | 485 | 0.999548 | 1.000000 |
| wdown | 4096 x 12288 | 759 | 0.998868 | 1.000000 |

int4 does partially engage on real weights, at 0.750 bytes/weight, but at a
visible accuracy cost that compounds over 36 layers.

## What it would buy

With group-wise scales, a ternary checkpoint should reach roughly 0.5
bytes/weight at cosine 1.000000, against 1.063 for `blockwise` today. For an 8B
model that is 4.0 GB instead of 8.5 GB of baked weights, which is the difference
between fitting and not fitting on a 16 GB machine once MLX also holds a decode
copy.

The accuracy argument is as strong as the size one: `blockwise` reconstructs
these tensors exactly, so a group-wise int4 path should too, while halving the
footprint.

## A smaller, related ask

There is no way to hand ANEForge already-quantized data. `weight()` takes an
fp16 numpy array and quantizes it internally, so a 2-bit checkpoint has to be
dequantized on the host and re-quantized, which is both lossy in principle and
costly in peak memory. An entry point accepting packed codes plus scales, with
the group size declared, would avoid the round trip entirely.

## Note on the fallback being silent

When a mode misses its error tolerance it drops to the next one without telling
the caller. The size difference is the only signal, and it is easy to miss.
Emitting a warning naming the requested and actual modes would have saved time
here.
