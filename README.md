# Driving the Apple Neural Engine for LLM prefill

A measurement study of the Apple Neural Engine (ANE) as a prefill accelerator for
language models, paired with the GPU for decode, on a single Apple Silicon laptop.
The work started as a roofline question (is the ANE worth using at all?) and grew
into a characterization of how the engine behaves under real transformer shapes:
throughput, memory, power, attention, quantization, and what the compiled program
actually is on disk.

Every number here was measured on one machine. The running lab notebook is
[NOTES.md](NOTES.md); the polished write-ups are in [results/](results/); the
scripts that produced each number are at the repository root.

## Machine and versions

| | |
|---|---|
| Chip | Apple M2 Pro, 16-core GPU, 16 GB unified memory |
| OS | macOS 27.0, build 26A5425a |
| ANECompiler | 10.26.6 |
| ANEServices | 10.19 |
| aneforge | 0.4.0 |
| MLX | 0.32.2, mlx-lm 0.31.3 |
| coremltools | 9.0 |
| Python | 3.11.15 (uv venv) |

The ANE is reached through [ANEForge](https://github.com/sbryngelson/ANEForge),
which compiles a tensor graph to a single ANE program and dispatches it through
the same runtime Apple's own frameworks use. No kernel driver work and no
entitlements are involved. The GPU side is [MLX](https://github.com/ml-explore/mlx).

## What the ANE is, as a GEMM engine

On this machine the ANE is the faster matmul engine, not a smaller second one.
With per-call overhead amortized, fp16 matmul reaches about 12 TFLOPS against the
GPU's 5, and it does so at roughly a sixth of the power. That inverts the common
assumption that the ANE is a weak inference-only block.

| metric | ANE | GPU (MLX) |
|---|---|---|
| peak fp16 matmul | ~12 TFLOPS (tiled) | ~5 TFLOPS |
| power under load | 8.7 W | 21.8 W |
| efficiency | 1.40 TFLOPS/W | 0.23 TFLOPS/W |
| weight streaming | ~58 GB/s | ~159 GB/s |

The split that falls out: prefill is compute-bound and belongs on the ANE; decode
is bandwidth-bound and belongs on the GPU.

## Findings

Each row links to the full write-up, with the script that produced it.

| finding | where | script |
|---|---|---|
| ANE vs GPU fp16 roofline, the tiling law | [results/roofline_aneforge.md](results/roofline_aneforge.md) | `aneforge_roofline.py`, `tile2d.py` |
| Throughput tracks reduction-axis width, not weight bytes | [results/reduction_law.md](results/reduction_law.md) | `reduction_law.py` |
| One fused transformer block at 7.75 TFLOPS | [results/block_bench.md](results/block_bench.md) | `block2b.py` |
| Per-layer accuracy and W8A16 as a measured delta | [results/block_bench.md](results/block_bench.md) | `perlayer.py` |
| Power and efficiency vs the GPU | [results/power.md](results/power.md) | `power_gpu.sh`, `sustain_ane.py` |
| Attention per-op, blocking, and GPU SDPA | [results/trackc_attention.md](results/trackc_attention.md) | `trackc_perop.py`, `attn_bench.py` |
| MLX 2-bit decode roofline | [results/trackb_decode_roofline.md](results/trackb_decode_roofline.md) | `trackb_profile.py`, `trackb_gemv.py` |
| Hybrid: ANE prefill, KV bridge, GPU decode | [results/hybrid.md](results/hybrid.md) | `hybrid.py`, `contend.py` |
| Chunked prefill vs bucketed | [results/chunked_prefill.md](results/chunked_prefill.md) | `chunked.py` |
| Layer segmentation and the memory boundary | [results/segmentation.md](results/segmentation.md) | `seg_bench.py`, `full36.py` |
| The memory recipe (1 layer/program, tiling, int4) | [results/memory_reduction.md](results/memory_reduction.md) | `marginal.py`, `recipe.py` |
| Weight residency, no zero-copy with MLX | [results/residency.md](results/residency.md), [results/weight_sharing.md](results/weight_sharing.md) | `blobshare.py`, `sharetest.py` |
| int4 vs blockwise on a real 2-bit checkpoint | [results/tracka_int4_vs_blockwise.md](results/tracka_int4_vs_blockwise.md) | `tracka_cosine.py` |
| What the compiled bundle is on disk | [results/direct_backend_exp0.md](results/direct_backend_exp0.md) | `exp0_harness.py`, `exp0_manifest.py` |

## The results worth knowing even if you read nothing else

**Tiling beats quantization, and the reason is the reduction axis.** Holding
weight bytes and arithmetic fixed and changing only the contraction width swings
fp16 throughput by 4.4x. Splitting the reduction into tiles of about 2048 takes a
chained-matmul workload from 6.9 to 12.2 TFLOPS at no accuracy cost. A walked-back
claim lives here too: int8 is not a native 2x rate on this engine. Its speedup
ranges from 1.0x to 1.5x depending on the contraction and vanishes at wide K,
which is consistent with on-device dequantization rather than integer math.

**Prefill on the ANE runs alongside GPU decode for about 87% of decode
throughput.** Measured by interleaving idle and saturated conditions with the
decoder warmed. An earlier 99.8% figure was wrong, measured over too short a
window with an unwarmed baseline, and is corrected in the write-up.

**A transformer block runs end to end on the ANE at 7.75 TFLOPS in fp16**, at
cosine 0.99998 against an fp16 reference. Apple's native fused attention fails to
dispatch for these shapes, so attention is decomposed; the GPU's flash SDPA kernel
is about 2.4x faster than the ANE's best blocked attention, so any path where the
KV is already on the GPU should run attention there.

**The on-disk compiled bundle is a 2 KB manifest that the runtime never reads
back.** Every load recompiles the MIL program in process. The weight blob is
honored bit for bit with no validation; the MIL is validated at compile time.
There is no microcode file to inspect or patch.

**The memory cost of hosting weights was misunderstood.** Weights are not held
twice. The apparent doubling was the compiler's working arena. One layer per
program costs about 1.0x the weight bytes at the margin; one large program costs
about 4x. int4 programs add only a few megabytes of resident memory each, and the
ANE wires roughly one layer at a time rather than the whole model.

## Reproducing

The scripts live at the repository root and cross-import, so they run in place.

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

ANEForge compiles on first use and caches under `~/Models/.aneforge-cache`. A
failed compile can poison that cache, so clear it if a shape that should work
starts failing (see the note in NOTES.md). `diskguard.py` caps the cache and
`swapgate.py` refuses to run a measurement when the machine is under swap
pressure, which badly distorts timing on a 16 GB box.

Entry points by area:

- `aneforge_roofline.py`, `tile2d.py`, `reduction_law.py` for the GEMM rooflines
- `block2b.py`, `perlayer.py` for the transformer block
- `hybrid.py`, `contend.py`, `chunked.py` for the prefill-decode pipeline
- `trackb_profile.py`, `trackc_perop.py`, `attn_bench.py` for decode and attention
- `exp0_harness.py` for the compiled-bundle anatomy

## Upstream issues

Three issues were drafted against ANEForge 0.4.0, in [issues/](issues/). They have
version pins, minimal reproductions, and what was ruled out. They are drafts; no
account here has posted them.

- `af.sdpa` compiles then fails at dispatch with `verifyBundleAtPath: invalid model`
- A failed compile poisons the cache and the failure then reproduces from fresh processes
- The int4 LUT path wants group-wise scales to fit modern 2-bit checkpoints

## Honest limitations

- One machine, 16 GB. Several late runs were taken under heavy swap, which makes
  the absolute numbers a floor. Those write-ups say so at the top.
- The full 36-layer 8B residency and per-layer cosine runs did not complete under
  memory pressure. The trends through 12 to 18 layers were unanimous, and the
  write-ups report the partial coverage rather than extrapolating past it.
- An in-process tracing experiment to inventory the ANE submission path was not
  completed in this record.

## Lineage

This builds on open reverse-engineering of the Apple Neural Engine:
[ANEForge](https://github.com/sbryngelson/ANEForge), the
[Asahi Linux ANE driver](https://github.com/eileen-code4fun/ANE), and tinygrad's
`extra/accel/ane`. Those projects decoded the descriptor format and the dispatch
path that this study measures against.
