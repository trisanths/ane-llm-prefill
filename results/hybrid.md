# Item 4: ANE prefill, KV bridge, MLX decode

Qwen3-0.6B dense, int8 weights on both engines, 512-token bucket. Medians over 5
runs. Raw rows in `hybrid.jsonl`, code in `../hybrid.py` and `../hybrid2.py`.

The model is smaller than the Qwen3-4B asked for; see "Why 0.6B" below.

## The result that matters: prefill is free

Running a second request's prefill on the ANE while the GPU decodes the first
costs the decode essentially nothing.

| | tokens/s |
|---|---|
| decode alone on the GPU | 72.58 |
| decode while the ANE prefills another request | 72.45 |

That is 99.8 percent retention. During the 0.66 s it took to decode 48 tokens,
the ANE completed 8 full 512-token prefills, so roughly 4096 prompt tokens were
processed for free alongside a decode that did not slow down.

This is the whole argument for the split architecture, and it holds because the
two engines contend for almost nothing. Decode is bandwidth-bound on the GPU;
prefill is compute-bound on a separate accelerator with its own power rail.

## Time to first token

| | ANE prefill | GPU prefill (MLX) |
|---|---|---|
| 512-token prompt | 0.110 s | 0.131 s |
| tokens/s | 4649 | 3913 |

The ANE is 1.19x faster, which is well short of the 2.5x it shows on raw chained
GEMM. Two reasons, both structural rather than disappointing. Qwen3-0.6B has a
hidden size of 1024, so every reduction is already narrow and the reduction-width
advantage from item 3 has little to bite on. And the ANE always pays for the full
512-token bucket even when the real prompt is shorter, while MLX prefills only
the real length.

The bucket effect is visible directly. A 128-token prompt padded into the 512
bucket gets 1186 tokens/s of effective prefill, against 4855 tokens/s for a
512-token prompt that fills its bucket. Bucket granularity is the single biggest
lever on TTFT for short prompts.

## Cold versus cached

| | 128-token prompt | 512-token prompt |
|---|---|---|
| cold, first use of the bucket | 21.8 s | 0.127 s |
| warm | 0.108 s | 0.105 s |

The 21.8 s is the one-time compile of the 512-token prefill program. The second
row is cheap because the bucket was already compiled by the first. This is the
"compile once per machine" property from step 1, and it is why bucketing is
mandatory rather than an optimization: every distinct prompt length would
otherwise pay 20 seconds.

## The bridge is free

Adopting the ANE's per-layer K and V into MLX costs 4 to 6 ms for 15 to 59 MB of
cache. That is consistent with the 50 GB/s host copy rate measured in item 2, and
it confirms the prediction from item 1b that the bridge would not be
transfer-bound. `LlamaPrefill._prefill_seed` returns K and V already trimmed to
the real positions, and MLX adopts them directly.

## Why 0.6B and not 4B

Qwen3-1.7B and larger will not compile their batched prefill program on this
machine, and the first diagnosis was wrong in an instructive way.

The failure reports as `ane_e5rt_program_compile failed (mask=0x4)`, which looks
like a compiler limit. Underneath, one attempt surfaced the real error:
`[Errno 28] No space left on device`. Each prompt-length bucket is a separate
compiled program holding the full model's baked weights, so a handful of
attempts at 1.7B consumed 9.5 GB of compile cache and filled a volume that
started at 97 percent. The failures then persisted through the cache-poisoning
mechanism from step 2b, where a failed compile leaves an entry that identical
graphs hit forever after.

After clearing the cache, with 21 GB free, 1.7B still failed at the same graph
hash, and the attempt itself consumed 7 GB. So there are two compounding limits,
disk headroom and something in the multi-output prefill program at that size, and
this machine cannot separate them cleanly.

Qwen3-0.6B compiles and runs reliably in both fp16 and int8, so the architecture
is validated end to end at that size. Moving to 4B needs roughly 8 GB for the
download plus 4 GB per int8 bucket program, which does not fit beside the 17 GB
of existing Hugging Face cache on this volume.

## Correctness

The decoder is a Qwen3 implementation in MLX driven by the same weights the ANE
uses, including per-head q and k RMSNorm, which Qwen3 has and Llama does not. It
produces coherent continuations from a real prompt and gibberish from random
token IDs, which is the expected behavior for both. Per-layer cosine for the
prefill path itself is covered in results/block_bench.md; this step adds no new
arithmetic, only the handoff.

## What this sets up

The concurrency result is strong enough to build on. A serving loop that keeps
the GPU decoding while the ANE prefills queued requests gets the prefill
throughput at no measured cost to interactive latency.

Two things to fix before that is real. Bucket granularity needs to be finer than
512 or short prompts waste most of their prefill. And the size ceiling needs
resolving, since a 0.6B model is below the useful range and the blocker is a
mixture of disk and an unexplained compile failure.

---

# Correction: the concurrency headline was measured wrong

The 99.8 percent decode retention above does not survive a correct measurement.
The real figure on Qwen3-0.6B is about 87 percent.

## What was wrong

Three compounding faults, each of which flattered the result.

**The window was too short.** The original test decoded 48 tokens in 0.66 s
while a worker thread issued ANE prefills. Eight prefills fitted in that window.
Contention needs longer to show.

**The worker saturated the ANE.** It looped with no pause, which is not a serving
pattern. On Qwen3-1.7B over a longer window the same code drove decode from 35
tokens/s to 0.42, a 98 percent collapse, with the worker completing 890 prefills.
That is the same code that reported 99.8 percent retention at 0.6B.

**MLX warmup contaminated the baseline.** The first `generate()` compiles Metal
kernels. In the duty-cycle rewrite, the uncontended baseline ran first and came
out *slower* than the contended runs, producing a nonsensical 128 percent
retention. Even after warming, throughput drifted upward across a run, from 63 to
80 tokens/s on identical work.

## The corrected measurement

Interleaving idle and saturated conditions, so drift affects both equally:

```
sequence: 0:62.6  1:65.3  0:80.4  1:70.8  0:80.2  1:69.3  0:80.7  1:69.2
```

The first pair is still warming. The three stable pairs give ratios of 0.880,
0.865 and 0.857.

| ANE state | decode tokens/s |
|---|---|
| idle | 80.3 |
| saturated | 69.2 |

**Retention is 86 to 87 percent.** Saturating the ANE costs about 13 percent of
GPU decode throughput.

## What it is worth anyway

During those runs the ANE completed 51 prefills of 512 tokens, roughly 26000
prompt tokens, at a cost of 13 percent of decode. That is still a strong trade,
and the architecture still holds. It is simply not free, and the earlier claim
that it was free came from a measurement that could not have detected the cost.

## Not measurable at 1.7B on this machine

The same test at Qwen3-1.7B cannot be run here. The hybrid needs weights in
ANEForge's host arrays, in the 28 baked segment programs, and in MLX
simultaneously, which drove swap to 16.05 GB of 17.4 GB. Decode throughput across
identical runs ranged from 0.42 to 37 tokens/s, which is swap noise rather than
contention. See results/segmentation.md for the exact memory figures.
