"""ANE block vs MLX fp16 block: cosine, argmax agreement, and the GPU baseline."""
import json
import sys
import time
import warnings

import numpy as np
import aneforge as af

warnings.filterwarnings("ignore", category=af.DispatchFloorWarning)
warnings.filterwarnings("ignore", category=af.PrecisionWarning)

from block2b import CFG, build, flops, reference  # noqa: E402
import mlx_block_ref  # noqa: E402

out_path = sys.argv[1] if len(sys.argv) > 1 else None
for M in (256, 512, 1024):
    y, W, cos, sin = build(M, 2560, "manual", False)
    prog = af.compile(y)
    xi = np.random.default_rng(1).standard_normal((M, CFG["d"])).astype(np.float16)
    for _ in range(3):
        prog(xi)
    ts = []
    for _ in range(9):
        t = time.perf_counter(); prog(xi); ts.append(time.perf_counter() - t)
    ane_ms = float(np.median(ts)) * 1e3
    ane = np.asarray(prog(xi), dtype=np.float32)
    prog.release()

    gpu_s, mlx_out = mlx_block_ref.bench(W, cos, sin, M)
    ref32 = reference(xi, W, cos, sin)

    def cmp(a, b):
        af_, bf = a.ravel(), b.ravel()
        return {"cosine": float(af_ @ bf / (np.linalg.norm(af_) * np.linalg.norm(bf))),
                "argmax_match": float((a.argmax(-1) == b.argmax(-1)).mean()),
                "max_rel_err": float(np.abs(a - b).max() / np.abs(b).max())}

    fl = flops(M)
    r = {"bench": "compare", "M": M,
         "ane_ms": ane_ms, "gpu_ms": gpu_s * 1e3,
         "ane_tops": fl["total"] / (ane_ms / 1e3) / 1e12,
         "gpu_tops": fl["total"] / gpu_s / 1e12,
         "speedup_ane_over_gpu": gpu_s * 1e3 / ane_ms,
         "ane_vs_mlx_fp16": cmp(ane, mlx_out),
         "ane_vs_fp32": cmp(ane, ref32),
         "mlx_vs_fp32": cmp(mlx_out, ref32)}
    print(json.dumps(r), flush=True)
    if out_path:
        with open(out_path, "a") as f:
            f.write(json.dumps(r) + "\n")
