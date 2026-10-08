# A failed compile poisons the cache, and the failure then reproduces forever

**aneforge** 0.4.0 · macOS 27.0 (26A5425a) · Apple M2 Pro (family 3) · 16 GB

## Summary

When `af.compile` fails, an entry keyed by the graph is left in
`~/Models/.aneforge-cache`. Every later attempt at the same graph then fails
identically, including from a completely fresh process. The result is
indistinguishable from a deterministic hardware or compiler limit.

This cost real debugging time and produced a wrong conclusion that survived
several experiments: a tile width was recorded as unsupported when it works.

## Symptom

```
RuntimeError: ane_e5rt_program_compile failed (mil=~/Models/.aneforge-cache/<hash>/model.mil,
                                               mask=0x4)
compile(mask=0x4) err=11
```

Once seen for a given graph, it recurs on every subsequent attempt at that graph,
in new processes, until the cache directory is removed. After
`rm -rf ~/Models/.aneforge-cache`, the identical code compiles first try.

## Reproduction

The reliable trigger is memory pressure, because the initial failure has to come
from somewhere.

1. Build a transformer block with d=5120, ff=17408, M=512, which bakes roughly
   600 MB of weights per program, and compile several with distinct weights in
   one process. On a 16 GB machine the fourth or fifth fails.
2. Note the failing graph.
3. Start a fresh process and compile only that graph. It fails identically,
   although nothing else is resident.
4. `rm -rf ~/Models/.aneforge-cache`, run step 3 again. It succeeds.

Observed directly: an 8-layer fp16 stack at this shape fails, while the same
stack with `compress="int8"`, at half the weight bytes, succeeds.

## Why it is worth fixing

The wrong conclusion is very easy to reach. A reduction tile width of 2048 was
recorded as "does not compile on this hardware" across several experiments, on
the strength of it failing reproducibly from fresh processes. It compiles fine.
The first failure had come from an unrelated out-of-memory condition in a
different run.

A second instance of the same pattern: prefill programs for Qwen3-1.7B failed
with this error, and an unrelated attempt eventually surfaced the real cause,
`[Errno 28] No space left on device`. The cache had grown to 9.5 GB because each
prompt-length bucket is a separate program holding the full model's weights.

So the cache masks two distinct resource failures, memory and disk, behind a
message that reads like a compiler verdict.

## Suggested fixes, roughly in order of value

1. Do not write a cache entry when compilation fails, or write it with a failure
   marker that is not treated as a cache hit.
2. Surface the underlying errno. `[Errno 28]` reached the user only once by
   accident; `mask=0x4` never says "you are out of disk".
3. Consider a cache size cap or eviction. A single session reached 20 GB twice
   and filled the volume, because every distinct shape and every prompt bucket is
   a separate baked copy of the weights.
