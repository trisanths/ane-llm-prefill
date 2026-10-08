# Item 1: chunked prefill against buckets

Qwen3-0.6B, fp16. One program consumes C tokens at a time against a
fixed-capacity KV buffer, with position dependence carried entirely in runtime
cos/sin rows and a runtime mask, so a single compile serves every prompt length
up to the capacity. Correctness verified at cosine 0.999995 against the bucketed
path, matching top-1. Free space logged in every row of `chunked.jsonl`.

## TTFT

| prompt | chunked C=128 cap=512 | chunked C=256 cap=512 | chunked C=256 cap=1024 | bucket |
|---|---|---|---|---|
| 64 | - | - | 0.107 | 0.110 (512) |
| 128 | **0.055** | - | 0.107 | 0.110 (512) |
| 256 | - | **0.079** | 0.108 | 0.113 (512) |
| 512 | 0.190 | 0.156 | 0.240 | **0.112** (512) |
| 1024 | - | - | 0.448 | **0.289** (1024) |

Chunking wins where the bucket wastes work and loses where the bucket is full.
A 128-token prompt takes 55 ms chunked against 110 ms bucketed, a clean 2x,
because the bucket pays for all 512 positions. A 512-token prompt that fills its
bucket takes 156 ms chunked against 112 ms, so chunking costs 1.4x there.

Capacity matters as much as chunk size. The same 256-token prompt takes 79 ms at
cap=512 and 108 ms at cap=1024, because every chunk attends over the full
capacity whether or not it is filled. The mask hides unfilled slots but the
arithmetic still happens. That is the central inefficiency of this design and it
is what makes long prompts lose.

## Compile and disk

This is where chunking wins decisively.

| | compiles | disk |
|---|---|---|
| chunked | 1, about 20 s, serves every length up to cap | one program |
| bucketed | 1 per distinct length, 21 to 23 s each | one program per length |

Buckets at 512 and 1024 cost two compiles and two full copies of the model's
baked weights. The chunked program costs one of each. For a server that sees
arbitrary prompt lengths, that is the difference between a bounded cache and one
that grows with the length histogram, which is what filled this volume twice.

## Recommendation

Use chunking, with capacity set to the real maximum context rather than a round
number, and chunk size near the median prompt length. The regime where buckets
win, prompts that exactly fill their bucket, is rare in practice and costs 1.4x,
while the regime where chunking wins, short prompts, is common and pays 2x.

The fixed-capacity attention waste is worth fixing before this ships. Sizing
attention to the filled length would need a program per filled value, defeating
the single-compile property, so the real fix is probably a small ladder of
capacities (512, 2048, 8192) rather than one.
