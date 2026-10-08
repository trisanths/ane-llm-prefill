# Item 4: weight residency, and the RAM tier table

This was answered in the previous round and lived only in NOTES.md, which is
presumably why it read as still open. Here it is as its own file, with the RAM
table updated for what the 1.7B work since has shown.

## Can an ANE program share a weight buffer with MLX?

No, and there is no partial version of this.

ANEForge bakes weights into the compiled program. They survive deleting the host
numpy array, so at steady state a weight exists exactly once per ANE program
rather than twice. There is no API to point a program at an externally allocated
buffer.

`Model.input_view()` returns a writable numpy view onto the program's own input
buffer, which removes one copy on the activation path: 5.27 ms against 5.99 ms
for passing numpy at M=1024, about 12 percent. That is ANEForge's buffer, not a
shared one, so it is copy avoidance rather than aliasing.

MLX exposes `__dlpack__` but not `__array_interface__`, and its arrays live in
Metal allocations. Nothing at the Python level lets the two engines name the same
IOSurface. **A hybrid that prefills on the ANE and decodes in MLX holds the
weights twice, once baked per engine.** That is the fact that decides the table
below, and it is what makes Qwen3-1.7B hybrid impossible on 16 GB.

Runtime weights are possible but not a workaround.
`af.einsum("mk,nk->mn", x, w)` with both operands as inputs compiles and runs,
which would let weights come from outside. It costs 1.7x: 2.99 TFLOPS against
5.13 baked. And it still would not alias MLX memory.

Copies themselves are cheap and not worth engineering around: 8.4 MB moves in
0.166 ms, about 50 GB/s, and MLX to numpy costs the same.

## Can ANEForge do int8/int4 storage with fp16 compute?

Yes, and it already does. This needs no work.

`compress="int8"` is weight-only quantization with fp16 activations, which is
W8A16. The disk numbers confirm it is real storage compression rather than a
compute-time trick: a 4096-square fp16 weight produces a 33.6 MB program, exactly
its fp16 size, and the same weight at `compress="int8"` produces 16.8 MB, exactly
half. `compress="int4"` quarters it.

The caveat is accuracy, not capability. int4 measured slower than int8 and highly
variable in step 2a, and int8 produces numerically broken output on Qwen3-1.7B
(mean logit cosine 0.019 against fp16). So the storage saving is real and the
compute path is right, but the quantizer cannot currently be trusted at that
model size. The table below therefore gives both an int8 column and an fp16
column, and the fp16 column is the one to plan against until the 1.7B int8
failure is understood.

## Largest dense model per RAM tier

Assumes 4 GB for OS and runtime, 1 GB of KV cache. Hybrid means an ANE prefill
copy plus an MLX decode copy, which the residency finding above makes
unavoidable.

| RAM | usable | ANE-only int8 | ANE-only fp16 | hybrid int8+int8 | hybrid fp16+fp16 |
|---|---|---|---|---|---|
| 16 GB | 12 GB | 8B (9.0) | 4B (9.0) | 4B (9.0) | 1.7B (7.8) |
| 32 GB | 28 GB | 14B (15.0) | 8B (17.0) | 8B (17.0) | 4B (17.0) |
| 48 GB | 44 GB | 32B (33.0) | 14B (29.0) | 14B (29.0) | 8B (33.0) |
| 64 GB | 60 GB | 32B (33.0) | 14B (29.0) | 14B (29.0) | 14B (57.0) |

Weights alone, in GB: 0.6B is 1.2/0.6/0.3 at fp16/int8/int4; 1.7B is 3.4/1.7/0.8;
4B is 8/4/2; 8B is 16/8/4; 14B is 28/14/7; 32B is 64/32/16.

Two readings that matter.

**The hybrid costs a full model tier.** Holding weights twice drops a 16 GB
machine from 8B to 4B on the int8 path, and to 1.7B on the fp16 path.

**This machine is at its limit.** The fp16 hybrid cell for 16 GB says 1.7B, and
measurement agrees it does not actually fit: prefill alone leaves 0.07 GB of free
RAM and the hybrid drives swap to 16 GB. The table's arithmetic and the observed
behaviour meet at the same place, which is some evidence the table is calibrated.

**Above 48 GB the table stops moving** on the ANE-only int8 column, because 32B
int8 already fits at 48 and the list has nothing between 32B and 70B.
