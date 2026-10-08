# Weight sharing, step 1: the runtime keeps a private resident copy

**Stop point reached. The blob is not a mapped file, and physical footprint grows
by its full size. Steps 2 to 4 are not viable as described; the fallback applies.**

## Method

Compiled one ANEForge program with 29 chained 4096x4096 fp16 linears, 0.97 GB of
baked weights, then measured physical footprint at each stage and took
`vmmap -summary` and `footprint` on the live process. Full vmmap in
`vmmap_blob.txt`.

## Footprint through the lifecycle

| stage | physical footprint |
|---|---|
| before weights exist | 22 MB |
| host numpy weights built (0.97 GB) | 1080 MB |
| after `af.compile` | 928 MB |
| after deleting the host numpy arrays | 928 MB |
| after first run | 928 MB |
| after five more runs | 928 MB |

The host copy is released at compile, which is consistent with the earlier
finding that baked weights survive freeing the source arrays. What replaces it is
not free.

## Where the bytes live

From `vmmap -summary`:

| region type | size | dirty | count |
|---|---|---|---|
| Malloc Large | 928.0 MB | 925.8 MB | 29 |
| mapped file | 34.9 MB | 0 KB | 4 |
| IOSurface | 1088 KB | 1024 KB | 6 |

Twenty-nine Malloc Large regions, one per baked layer, totalling exactly the
weight size, and they are **dirty**. The mapped-file total is 34.9 MB with 416 KB
resident, which is shared libraries, not weights. IOSurface is about 1 MB, which
is the I/O buffers, not the weights.

`footprint` confirms it from the other side:

```
neural_peak:    928 MB
phys_footprint: 3908 MB
```

The tool attributes exactly 928 MB to the neural category.

## Conclusion

The baked weights are **process-private anonymous memory, dirty and resident**,
not a file mapping. There is no file to `mmap` a second time, so the step 2 plan
of wrapping the same pages in a Metal buffer via `bytesNoCopy` has nothing to
point at from outside the process.

In principle a `bytesNoCopy` buffer could alias an in-process malloc region, but
ANEForge exposes no pointer to it, the regions are per-layer rather than one
contiguous blob, and the layout is unknown. That is a much larger undertaking
than the "mmap the same file" path the plan assumed, and it would still be
fighting a runtime that has already decided to own the memory.

So: **the hybrid pays for two copies of the weights.** The residency table in
results/residency.md stands, and the fallback is baked ANE plus duplicated
attention only.

Steps 2, 3 and 4 are not attempted.
