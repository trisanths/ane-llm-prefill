# Direct backend, Experiment 0: what ANECompiler emits, and what can be patched

macOS 27.0 build 26A5425a, ANECompiler 10.26.6, ANEServices 10.19, aneforge 0.4.0,
Apple M2 Pro (ANE family H14S). Target program: one fp16 linear, input [64, 4096],
weight [2048, 4096], output [64, 2048]. Harness in `../exp0_harness.py`,
`../exp0_weights_tight.py`, `../exp0_manifest.py`; recovery in `../recover.sh`;
raw rows in `exp0_patches.jsonl`.

## Verdict

**On this build the on-disk "compiled bundle" is a 2 KB manifest plus a hash that
the runtime never reads back. There is no microcode or descriptor file to patch.
Every load is a fresh in-process compile from `model.mil`, with the resulting
program handed to the daemon in memory.**

The only on-disk surfaces that reach the engine are therefore:

| file | validated? | effect of a one-byte patch |
|---|---|---|
| `weights.bin` (ANEForge's blob) | no | honoured bit for bit; NaN and Inf run to completion |
| `model.mil` | yes, at compile | rejected before dispatch, `err=11` |
| `H14S.e5` (manifest) | never read | none |
| `model.anehash` | never read | none |
| microcode / descriptors | do not exist on disk | not testable here |

Nothing hung. A byte-level descriptor or microcode patch requires intercepting
ANECompiler's output buffer inside the process, which is Experiment 1.

## 1. Inventory

ANEForge's cache entry for the program:

```
weights.bin    16,777,344 B   64-byte header, then 2048x4096 fp16 at offset 64
model.mil             512 B   MIL text; weight declared as BLOBFILE(path, offset=64)
cache/com.apple.e5rt.e5bundlecache/26A5425a/<key1>/<key2>.bundle/H14S.bundle/
    H14S.e5           2,352 B
    main/main_ane/model.anehash   129 B
```

**H14S.e5** is a FlatBuffer (root offset 16, `__sym_desc__` trailer). Its strings
are build metadata and op names, not code:

```
10.26.6                          espressoc-component-ANECompiler
3600.16.1                        espressoc-component-MIL
on-device-compilation = true     built-for-profiling = false
input-file-path = <path to model.mil>
main__Op0_AneInference  __arg_frame  __op_attrs
```

Numeric fields at fixed offsets are the shapes and byte counts of the op:
`0x0434` = 4096 (K), `0x044c` = 64 (M), `0x0454` = 2048 (N), `0x04dc` = 8192,
`0x03c8`/`0x0470`/`0x0790`/`0x0858` = 16,777,216 (weight bytes), `0x03ec` =
262,144 (input elements). Two manifests produced from different weights are
byte-identical.

**model.anehash** is ASCII: two 64-hex-character strings joined by `_`. The first
is constant across runs with different weights; the second changes when
`weights.bin` changes. It is `program-hash_weights-hash`. Neither is the plain
SHA-256 of any file in the entry.

**No microcode.** The manifest declares on-device compilation. `vmmap` shows
`ANECompiler.framework`, `Espresso.framework`, `MIL.framework` and
`AppleNeuralEngine.framework` mapped into the Python process; `aned` stays at
7 MB through compile, load and run. `/private/var/db/neuralengine` is empty. A
whole-disk search for `.hwx` and for any e5rt artifact over 100 KB found only
`bnns_program.bnnsir` files belonging to other apps, which are BNNS CPU programs.
The compiled ANE program exists only in memory.

## 2. Patch tests

Each test patches, runs the program in a fresh subprocess with a timeout, records
whether the runtime rewrote any file, compares the output bitwise against the
pristine ANE output, then restores. Pristine-vs-pristine is 0.0.

### a. weights.bin

| patch | offset | verdict | effect |
|---|---|---|---|
| row 100 elem 0, low byte xor 0xFF | 819,264 | wrong | max diff 0.398, exactly column 99 changed in all 64 rows |
| row 100 elem 0, high byte = 0x7C | 819,265 | wrong, non-finite | Inf in column 99 |
| row 100 elem 0, high byte = 0x7E | 819,265 | wrong, non-finite | NaN in column 99 |
| row 2047 elem 4095, low byte xor | 16,777,278 | wrong | max diff 0.002, column 2047 |

Every patch landed exactly where the layout predicts, so the blob is plain
row-major fp16 at offset 64 with no tiling on disk. No validation, no rejection,
no hang, and NaN/Inf propagate silently. This confirms the Orion observation.

Note on the first attempt: the initial harness judged against an fp32 reference
with a 1e-3 tolerance and reported every weights patch as "ok". A single low-byte
flip moves one weight by about 0.004 and one output column by less than the
tolerance. Judging bitwise against the pristine ANE output is what made the
effect visible.

### b. descriptor fields in H14S.e5

| patch | verdict | elapsed | bundle dirs | file rewritten? |
|---|---|---|---|---|
| N field at 0x0454, xor 0xFF | identical | 0.20 s | +1 | no |
| root offset byte 0 = 0xFF | identical | 0.20 s | +1 | no |
| truncate to 100 bytes | identical | 0.19 s | +1 | no |
| delete the file | identical | 0.21 s | +1 | no |

Every case ran bit-identically and left a new bundle directory beside the damaged
one. The runtime did not repair the damaged file; it ignored it and compiled
fresh. The manifest is never on the load path.

### c. microcode

No on-disk target. See inventory.

### d. hash zeroed, then repeat b

| patch | verdict | elapsed | bundle dirs |
|---|---|---|---|
| model.anehash zeroed | identical | 0.19 s | +1 |
| model.anehash deleted | identical | 0.21 s | +1 |
| entire e5bundlecache directory deleted | identical | 0.37 s | rebuilt |

Same story. The hash is written, never checked.

### e. model.mil (the real program source on this build)

| patch | verdict | elapsed |
|---|---|---|
| weight shape `[2048, 4096]` to `[2047, 4096]` | rejected | 0.14 s |
| `transpose_y` true to false | rejected | 0.14 s |
| blob `offset = uint64(64)` to `66` | rejected | 0.14 s |

All three fail inside `ane_e5rt_program_compile` before anything is dispatched.
The exact text ANEForge's dispatch library prints is:

```
compile(mask=0x4) err=11
RuntimeError: ane_e5rt_program_compile failed (mil=.../model.mil, mask=0x4).
See stderr for the actual Espresso error.
```

The underlying Espresso diagnostic is not surfaced: nothing beyond `err=11`
appears on stderr, and the unified log carries no matching entries under the
Espresso, ANE or process predicates in the window. So the compile-time validator
is real and fast, but opaque from this layer.

## 3. Recovery

Nothing in this experiment required it. NaN and Inf weights, corrupted and
missing manifests, a deleted bundle cache and three malformed MIL programs all
either ran or were rejected within a fifth of a second.

`recover.sh` provides the mechanism anyway:

- `submit <file.py>` runs a test in isolation under `ANE_TIMEOUT` (default
  120 s), kills it on expiry, then probes the engine with a known-good program in
  a fresh process. Verified: a deliberately hanging child is killed at 5 s and the
  probe passes afterwards.
- `probe` runs the health check alone.
- `recover` kills stray experiment children, re-probes, and if the probe still
  fails prints the two steps that need a human because they need sudo:
  `sudo launchctl kickstart -k system/com.apple.aned`, then reboot. The label is
  confirmed present in the system launchd domain.

## 4. Cross-version

Not testable, one machine. Structurally, the e5rt cache is keyed under the OS
build (`26A5425a/`), so a bundle is build-scoped by construction. Since the
bundle is never reloaded on this build, portability of the bundle is moot; what
would carry across builds is `model.mil` plus `weights.bin`, which is exactly
what ANEForge already treats as the artifact.

## What this means for a direct backend

The layer worth targeting is not a file. It is the in-memory program that
ANECompiler produces from MIL and that `ane_e5rt_program_compile` hands to the
daemon. Two consequences:

- Weight-level control is already complete at the file layer: any bytes written
  at the declared offset are what the engine multiplies with, unvalidated.
- Descriptor and microcode control means hooking inside the process, between
  ANECompiler's output and the IOSurface submission. That is a live-memory
  experiment with a different risk profile from anything here, and it is the
  first thing that could genuinely wedge the engine.
