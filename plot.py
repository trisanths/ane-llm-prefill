"""Plot ANE vs MLX GPU curves from results_*.jsonl."""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load(path):
    rows = []
    for line in open(path):
        r = json.loads(line)
        if "error" not in r:
            rows.append(r)
    return rows


rows = load("results_ane.jsonl") + load("results_mlx.jsonl")
label = {"ane": "ANE (Core ML, cpuAndNeuralEngine)", "mlx_gpu": "GPU (MLX)"}

fig, axes = plt.subplots(1, 3, figsize=(17, 5))

# Panel 1: TFLOPS vs M at K=N=4096
ax = axes[0]
for units in ("ane", "mlx_gpu"):
    for layers, ls in ((1, "--"), (8, "-")):
        sel = sorted([r for r in rows if r["units"] == units and r["K"] == 4096 and r["N"] == 4096
                      and r["layers"] == layers], key=lambda r: r["M"])
        if sel:
            ax.plot([r["M"] for r in sel], [r["tflops_best"] for r in sel], ls, marker="o",
                    label=f"{label[units]}, {layers} layer(s)")
ax.set_xscale("log", base=2)
ax.set_xlabel("M (rows of input)")
ax.set_ylabel("fp16 TFLOPS (best of N)")
ax.set_title("Throughput vs M, weights 4096x4096")
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=8)

# Panel 2: weight bandwidth vs M
ax = axes[1]
for units in ("ane", "mlx_gpu"):
    for layers, ls in ((1, "--"), (8, "-")):
        sel = sorted([r for r in rows if r["units"] == units and r["K"] == 4096 and r["N"] == 4096
                      and r["layers"] == layers], key=lambda r: r["M"])
        if sel:
            ax.plot([r["M"] for r in sel], [r["weight_GBps_best"] for r in sel], ls, marker="o",
                    label=f"{label[units]}, {layers} layer(s)")
ax.set_xscale("log", base=2)
ax.set_xlabel("M (rows of input)")
ax.set_ylabel("weight bytes / time, GB/s")
ax.set_title("Weight-streaming bandwidth vs M")
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=8)

# Panel 3: TFLOPS vs weight size (square K=N) at fixed M
ax = axes[2]
for units in ("ane", "mlx_gpu"):
    sel = sorted([r for r in rows if r["units"] == units and r["K"] == r["N"] and r["M"] == 1024
                  and r["layers"] == 1], key=lambda r: r["K"])
    if sel:
        ax.plot([r["K"] for r in sel], [r["tflops_best"] for r in sel], marker="o", label=label[units])
ax.set_xscale("log", base=2)
ax.set_xlabel("K = N (square weight side)")
ax.set_ylabel("fp16 TFLOPS (best of N)")
ax.set_title("Throughput vs weight size, M=1024, 1 layer")
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=8)

plt.tight_layout()
out = sys.argv[1] if len(sys.argv) > 1 else "roofline.png"
plt.savefig(out, dpi=130)
print("wrote", out)
