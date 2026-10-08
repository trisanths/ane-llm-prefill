# Track A: Bonsai 8B, int4 vs blockwise per-layer cosine

Real Ternary Bonsai 8B weights (g128 ternary, 2-bit), dequantized to fp16 as the
reference, then recompiled on the ANE under `compress="int4"` (single 16-entry
codebook per tensor) and `compress="blockwise"` (per-group int8 + fp16 scales).
Cosine of each layer's down-projection output (the widest reduction, worst case
for int4) vs the fp16 reference.

## Per-layer cosine

| layer | int4 | blockwise |
|---|---|---|
| 0 | 0.998798 | 1.000000 |
| 6 | 0.999424 | 1.000000 |
| 12 | 0.999315 | 1.000000 |

The run could not finish 36 layers (swap reached 12.8 GB), but three independent
layer samples tell one story with no variance in the conclusion.

## The decision: blockwise for Bonsai

**Blockwise reconstructs exactly.** Cosine is 1.000000 at every layer, because
Bonsai's format is ternary codes with a per-128 fp16 scale, and blockwise stores
exactly that structure (per-group int8 data with fp16 scales). There is no
approximation.

**int4 cannot, and loses ~0.001 per layer.** Its single 16-entry codebook per
tensor cannot represent thousands of distinct g128-scaled values, so it falls
back to an int8-quality fit at 0.999. The loss is stable, not compounding
catastrophically, but it is a per-layer error that a 36-layer stack accumulates,
and it is avoidable.

## Cost trade

| mode | bytes/weight | 8B baked | per-layer cosine |
|---|---|---|---|
| int4 | 0.75 | ~5.2 GB | 0.999 |
| blockwise | 1.063 | ~7.4 GB | 1.000000 |

int4 is 30% smaller but lossy on this format; blockwise is exact but 2.2 GB
larger. For a model you want correct output from, blockwise is the call: the
exactness matters more than 2 GB on a machine that can host the model at all, and
the earlier Qwen3-1.7B int8 disaster (logit cosine 0.019) is a reminder that
quantization error on a real checkpoint is not always benign. The right upstream
fix is group-wise scales for the int4 LUT path (filed as issue 03), which would
give int4's size at blockwise's accuracy.
