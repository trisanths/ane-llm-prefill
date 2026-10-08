# Power and efficiency, ANE against GPU

M2 Pro, 16-core GPU, 16 GB, macOS 27.0 build 26A5425a. Same arithmetic on both
engines: 8 chained 4096x4096 fp16 linears at M=1024, the ANE version tiled at
kt=1024. Sampled with `powermetrics` at 300 ms.

## Result

| | ANE | GPU (MLX) |
|---|---|---|
| throughput | 12.16 TFLOPS | 4.94 TFLOPS |
| power, loaded median | 8.70 W | 21.76 W |
| peak | 8.92 W | 46.7 W |
| efficiency | 1.40 TFLOPS/W | 0.23 TFLOPS/W |
| energy per PFLOP | 716 J | 4404 J |

The Neural Engine does 2.5 times the work at 2.5 times less power, so it is
about 6 times more efficient per joule on this workload.

For a prefill engine that is the whole argument. The throughput advantage alone
was 2.5x; the power advantage doubles the case on a laptop, where sustained GPU
prefill is what heats the machine and drains the battery.

## Why the first attempt failed, and what it was not

The two-phase capture gave only 5 GPU samples above 3 W across 285 s, against an
MLX benchmark that ran 8.9 s. The burst contained a physically impossible
reading, 678 mW sitting between a 24 W and a 46 W sample, so the rail looked
broken and no number was published from it.

Driving the GPU continuously for 60 s instead settles it: 379 of 500 samples
exceed 3 W, and inside the sustained window 99 percent do, with a tight median of
21.8 W. The rail reports correctly under sustained load.

So the rail is not broken in the way it appeared. What remains unexplained is why
an 8.9 s benchmark registered as 1.5 s of load. The short run is bracketed by
process startup, weight generation and teardown, and the GPU clock ramps, but
that does not fully account for the gap. The lesson is procedural rather than
diagnostic: measure power against a sustained load, never against a short
benchmark, and treat a sparse loaded-sample count as a reason to distrust the
capture rather than the hardware.

## Confidence

The GPU figure rests on 379 loaded samples and is solid.

The ANE figure rests on 12 consecutive samples from the two-phase run, which is
few, but they are tightly clustered between 8.16 and 8.92 W, the idle floor is a
clean 0 mW, and the combined rail corroborates independently: 21.2 W during the
phase against 11.8 W idle is a 9.4 W delta against the 8.7 W the ANE rail
reports. A sustained ANE capture would tighten it and has not been run.

## Reproducing

    ./power_gpu.sh              # 60 s sustained GPU load, self-terminating

`power_bench.sh` runs the two-phase version. Note that its predecessor leaked
three root `powermetrics` processes, because `sudo kill` cannot prompt for a
password from the background; both scripts now rely on `-n` so powermetrics ends
by itself.
