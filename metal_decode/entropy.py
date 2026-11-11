"""Order-0 entropy of an MLX affine-quantized checkpoint: codes, scales, biases.

Reports the stored bits per weight and the entropy bound for each part, so the
result shows how many bytes a lossless re-encoding could save per decode token.
"""
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np


def h(counts):
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log2(p)).sum())


def code_stats(packed, bits):
    w = np.asarray(packed).view(np.uint32)
    per = 32 // bits
    mask = (1 << bits) - 1
    codes = np.stack([(w >> (bits * i)) & mask for i in range(per)], -1).reshape(w.shape[0], -1)
    c = codes.astype(np.int64).ravel()
    h0 = h(np.bincount(c, minlength=1 << bits))
    pairs = codes[:, 0::2].astype(np.int64) * (1 << bits) + codes[:, 1::2]
    h2 = h(np.bincount(pairs.ravel(), minlength=1 << (2 * bits))) / 2
    return codes.size, h0, h2


def half_stats(a):
    u = np.asarray(a.view(mx.uint16)).ravel().astype(np.int64)
    full = h(np.bincount(u, minlength=1 << 16))
    hi = h(np.bincount(u >> 8, minlength=256))
    lo = h(np.bincount(u & 255, minlength=256))
    return u.size, full, hi + lo


def main(model_dir):
    model_dir = Path(model_dir)
    cfg = json.loads((model_dir / "config.json").read_text())
    q = cfg["quantization"]
    bits, gs = q["bits"], q["group_size"]
    w = mx.load(str(model_dir / "model.safetensors"))
    rows = []
    tot = dict(n=0, code_bits=0.0, h0=0.0, h2=0.0, sb_bits=0.0, sb_h=0.0, sb_split=0.0)
    for name in sorted(k for k in w if k.endswith(".weight") and k[:-7] + ".scales" in w):
        base = name[:-7]
        n, h0, h2 = code_stats(w[name], bits)
        ns, fs, ss = half_stats(w[base + ".scales"])
        nb, fb, sb = half_stats(w[base + ".biases"])
        rows.append((base, n, h0, h2, fs, fb))
        tot["n"] += n
        tot["code_bits"] += n * bits
        tot["h0"] += n * h0
        tot["h2"] += n * h2
        tot["sb_bits"] += (ns + nb) * 16
        tot["sb_h"] += ns * fs + nb * fb
        tot["sb_split"] += ns * ss + nb * sb
    n = tot["n"]
    print(f"model {model_dir.name}  bits={bits} group={gs}  quantized weights={n/1e6:.1f}M")
    print(f"{'part':<22}{'stored b/w':>12}{'entropy b/w':>13}")
    print(f"{'codes (order 0)':<22}{bits:>12.3f}{tot['h0']/n:>13.3f}")
    print(f"{'codes (pairs)':<22}{bits:>12.3f}{tot['h2']/n:>13.3f}")
    print(f"{'scales+biases (16b)':<22}{tot['sb_bits']/n:>12.3f}{tot['sb_h']/n:>13.3f}")
    print(f"{'scales+biases (hi|lo)':<22}{tot['sb_bits']/n:>12.3f}{tot['sb_split']/n:>13.3f}")
    stored = (tot["code_bits"] + tot["sb_bits"]) / n
    best = (min(tot["h0"], tot["h2"]) + tot["sb_h"]) / n
    print(f"{'total':<22}{stored:>12.3f}{best:>13.3f}   bound saves {100*(1-best/stored):.1f}% of bytes")
    print("\nper-tensor order-0 code entropy, min/median/max:",
          *(f"{v:.3f}" for v in np.percentile([r[2] for r in rows], [0, 50, 100])))
    worst = sorted(rows, key=lambda r: r[2])[:5]
    print("lowest-entropy tensors:", ", ".join(f"{r[0].split('model.')[-1]} {r[2]:.3f}" for r in worst))


if __name__ == "__main__":
    main(sys.argv[1])
