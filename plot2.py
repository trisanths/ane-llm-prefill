"""Plot int8 sweep, transformer block, and concurrency results."""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load(path):
    try:
        return [json.loads(l) for l in open(path) if "error" not in json.loads(l)]
    except FileNotFoundError:
        return []


int8 = load("results_int8.jsonl")
mlx = [r for r in load("results_mlx.jsonl") if r["layers"] == 8 and r["K"] == 4096]
block = load("results_block.jsonl")
conc = load("results_concurrent.jsonl")

fig, axes = plt.subplots(1, 3, figsize=(17, 5))

ax = axes[0]
for q, lab in (("none", "ANE fp16"), ("w8", "ANE int8 weights, fp16 act"), ("w8a8", "ANE int8 weights + int8 act")):
    sel = sorted([r for r in int8 if r["quant"] == q], key=lambda r: r["M"])
    ax.plot([r["M"] for r in sel], [r["tflops_med"] for r in sel], marker="o", label=lab)
sel = sorted(mlx, key=lambda r: r["M"])
ax.plot([r["M"] for r in sel], [r["tflops_med"] for r in sel], marker="s", ls="--", color="k", label="GPU (MLX) fp16")
ax.set_xscale("log", base=2)
ax.set_xlabel("M (rows)")
ax.set_ylabel("TFLOPS / TOPS (median)")
ax.set_title("8 chained 4096x4096 linears")
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=8)

ax = axes[1]
labels, vals, cols = [], [], []
order = [("coreml_ane", "manual", "none", "ANE fp16\nmanual attn"), ("coreml_ane", "sdpa", "none", "ANE fp16\nSDPA op"),
         ("coreml_ane", "manual", "w8a8", "ANE W8A8\nmanual attn"), ("mlx_gpu", "sdpa", None, "GPU MLX\nfp16")]
for M, hatch in ((512, ""), (1024, "//")):
    for eng, attn, q, lab in order:
        sel = [r for r in block if r["engine"] == eng and r["attn"] == attn and r["M"] == M
               and (q is None or r.get("quant", "none") == q)]
        if sel:
            labels.append(f"{lab}\nM={M}")
            vals.append(sel[0]["tflops_med"])
            cols.append("C3" if eng == "mlx_gpu" else ("C2" if q == "w8a8" else "C0"))
ax.bar(range(len(vals)), vals, color=cols)
ax.set_xticks(range(len(vals)))
ax.set_xticklabels(labels, fontsize=7)
ax.set_ylabel("TFLOPS / TOPS (median)")
ax.set_title("One transformer block, D=4096, 32 heads, MLP 12288")
ax.grid(True, axis="y", alpha=0.3)

ax = axes[2]
ind = {tuple(r["engines"]): r for r in conc if r["mode"] == "independent"}
names, vals = [], []
if ("ane",) in ind:
    names.append("ANE alone"); vals.append(ind[("ane",)]["aggregate_tflops"])
if ("gpu",) in ind:
    names.append("GPU alone"); vals.append(ind[("gpu",)]["aggregate_tflops"])
if ("ane", "gpu") in ind:
    r = ind[("ane", "gpu")]
    for x in r["results"]:
        names.append(f"{x['engine'].upper()} while both run"); vals.append(x["tflops"])
    names.append("both, aggregate"); vals.append(r["aggregate_tflops"])
ax.bar(range(len(vals)), vals, color=["C0", "C3", "C0", "C3", "C2"][:len(vals)])
ax.set_xticks(range(len(vals)))
ax.set_xticklabels(names, fontsize=7, rotation=15)
ax.set_ylabel("TFLOPS (fp16, 12 s window)")
ax.set_title("ANE and GPU in separate processes, M=1024")
ax.grid(True, axis="y", alpha=0.3)

plt.tight_layout()
out = sys.argv[1] if len(sys.argv) > 1 else "int8_block_concurrent.png"
plt.savefig(out, dpi=130)
print("wrote", out)
