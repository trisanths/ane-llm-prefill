# Track A: 8B residency under the memory recipe

Bonsai-shaped layers (d=4096, ff=12288), 1 layer per program, MLP tiled to
1024 rows, int4. M2 Pro, 16 GB, macOS 27.0. Measured with ~9 GB of swap already
held by other apps, so absolute RSS is inflated and the run could not complete;
the trend through 18 of 36 layers is the result.

## The measurement

| layer | fp16-equiv weights so far | process RSS | ANE-wired (neural) |
|---|---|---|---|
| 0 | 0.39 GB | 1095 MB | 181 MB |
| 6 | 2.70 GB | 690 MB | 182 MB |
| 12 | 5.02 GB | 1884 MB | 182 MB |
| 18 | 7.33 GB | 1802 MB | 182 MB |

## Two findings

**The ANE-wired footprint is flat.** `footprint`'s neural category holds at
182 MB from layer 0 to 18, while nominal weights climb past 7 GB. The engine does
not keep every compiled layer's weights wired; only about one layer's worth is
resident at once. This is the strongest single argument that the ANE side of an
8B model is not memory-bound: adding layers does not grow the wired set.

**Process RSS is sublinear and noisy.** At 7.33 GB of fp16-equivalent weights
(3.66 GB of params, ~2.7 GB baked at int4), process RSS is 1.8 GB, about 0.25x
the fp16-equivalent. It is non-monotonic because the allocator returns and reuses
the compiler arena between layers. This is far below the 1.0-1.4x that small runs
suggested and nowhere near the 4x of a single large program.

## Why it did not finish

The run stalled at layer 18. Free RAM was 68 MB and swap 9.2 GB, because other
applications held most of the 16 GB. Compilation did not crash; it thrashed to a
standstill. Killing it dropped swap by 3 GB immediately, confirming the run
itself was the marginal pressure that tipped the machine over.

## Verdict for the rent-a-bigger-Mac decision

The recipe works. The ANE-wired set is flat with depth and process RSS is ~0.25x
weights, so on a machine with real headroom an 8B model compiles and hosts
comfortably. This 16 GB machine cannot, but the blocker is the ~9 GB of ambient
memory pressure from other apps, not the ANE and not the recipe. On a 16 GB
machine with little else running, or any 32 GB machine, 8B should fit. That is
the concrete trigger: the wall here is total system RAM contention, and it is
crossed by either freeing ~4 GB locally or moving to 32 GB.
