# Item 2: layer-segmented prefill

ANEForge already groups layers into programs sized by `_chunk_bytes`, chaining
the hidden state between them through the host. The default of 1600 MB puts all
28 layers of Qwen3-1.7B into one program, which does not compile here. Lowering
it forces more, smaller programs.

## Qwen3-1.7B, fp16, 512-token prompt

| chunk budget | segments | layers each | cold | warm | tok/s |
|---|---|---|---|---|---|
| 800 MB | 4 | 7 | fail | - | - |
| 400 MB | 10 | 3 | 85.3 s | 0.605 s | 847 |
| 200 MB | 28 | 1 | 50.2 s | 0.394 s | 1301 |
| 100 MB | 28 | 1 | 50.8 s | 0.334 s | 1533 |
| 150 MB | 28 | 1 | - | 0.304 s | 1687 |

**Qwen3-1.7B prefill works, and segmentation is what makes it work.** Output is
correct: "The capital of France is" gives " Paris".

**Finer segmentation is faster, not slower.** This was the opposite of the
expectation. Chaining segments costs a dispatch plus a host round-trip of the
hidden state each, so 28 segments should lose to 10. It wins by 1.8x. Whatever
the per-segment overhead costs, the engine more than recovers it on smaller
programs. The practical rule is to set the budget to roughly one layer's weight
bytes and stop tuning.

Cold time is the compile, and it drops as segments get smaller, from 85 s at 10
segments to 50 s at 28. Smaller programs compile faster than the count grows.

## The memory boundary, precisely

This is the rent-a-bigger-Mac decision point, so here are the exact numbers.

Prefill alone at 1.7B fp16, 28 segments, on this 16 GB M2 Pro:

| | |
|---|---|
| resident set after load | 5.07 GB |
| free RAM at peak | 0.07 GB |
| swap used | 8.0 GB, rising about 1 GB during prefill |
| result | works, 303 ms, 1687 tok/s, correct output |

Adding the MLX decoder to that is what breaks. The hybrid needs weights in three
places at once: ANEForge's host numpy copy, the baked copies inside the 28
segment programs, and MLX's own copy. Measured during the attempted 1.7B hybrid
run, swap reached 16.05 GB of 17.4 GB and every timing became a swap artifact,
with decode throughput swinging between 0.4 and 37 tokens/s across identical
runs.

So: **1.7B prefill on the ANE works on 16 GB. The 1.7B hybrid does not.** The
blocker is RAM, not compile, not disk, and not the ANE. A machine with 32 GB or
more would hold all three copies with room to spare, and by the RAM table in
NOTES.md would reach 8B hybrid or 14B ANE-only.

## int8 is not usable at 1.7B

Qwen3-1.7B with `compress="int8"` compiles and runs but is numerically broken.

| prompt | fp16 | int8 |
|---|---|---|
| The capital of France is | ' Paris' | ' Italy' |
| The largest planet in our solar system is | ' the' | ' our' |
| The author of Romeo and Juliet is | ' William' | ' Romeo' |

Mean full-model logit cosine between the two is **0.019**, effectively
uncorrelated, with top-1 agreement of 1 in 5.

This is not the gradual depth drift predicted earlier. That model predicted about
0.9989 at 28 layers, not 0.019. And it is not a segmentation interaction:
Qwen3-0.6B, which also has 28 layers, produces correct output under int8 at every
segment granularity from 28 layers per program down to 1. The failure is specific
to the 1.7B weights.

Worth reporting upstream once it is narrowed further; it is not in the drafted
issues because the cause is not yet isolated to a shape or an op.
