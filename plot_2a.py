"""Plots for results/roofline_aneforge.md (step 2a)."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

rows = [json.loads(l) for l in open("results/roofline_aneforge.jsonl")]
ok = [r for r in rows if "error" not in r]
chain = [r for r in ok if r["bench"] == "chain"]
t2d = [r for r in ok if r["bench"] == "tile2d"]
ct = [r for r in ok if r["bench"] == "chain_tiled"]

fig, axes = plt.subplots(1, 4, figsize=(21, 5))

ax = axes[0]
for prec, col in (("fp16", "C0"), ("int8", "C2"), ("int4", "C4")):
    for L, ls in ((1, "--"), (8, "-")):
        sel = sorted([r for r in chain if r["precision"] == prec and r["layers"] == L], key=lambda r: r["M"])
        if sel:
            ax.plot([r["M"] for r in sel], [r["tops_med"] for r in sel], ls, marker="o",
                    color=col, ms=4, label=f"{prec}, {L} layer(s)")
ax.axhline(5.0, ls=":", color="r", lw=1)
ax.text(1.2, 5.15, "GPU (MLX) fp16 ceiling", fontsize=7, color="r")
ax.set_xscale("log", base=2); ax.set_xlabel("M (rows)"); ax.set_ylabel("TFLOPS / TOPS (median)")
ax.set_title("ANEForge: chained 4096x4096 linears, untiled")
ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=7)

ax = axes[1]
sel = sorted([r for r in t2d if r["K"] == 16384 and r["kt"] == r["nt"]], key=lambda r: r["kt"])
ax.plot([r["kt"] for r in sel], [r["tops_med"] for r in sel], marker="o", color="C3", label="square tiles")
asym = sorted([r for r in t2d if r["K"] == 16384 and r["kt"] != r["nt"]], key=lambda r: r["kt"])
if asym:
    ax.scatter([r["kt"] for r in asym], [r["tops_med"] for r in asym], marker="s", color="C1",
               zorder=5, label="non-square tiles")
    for r in asym:
        ax.annotate(f"{r['kt']}x{r['nt']}", (r["kt"], r["tops_med"]), fontsize=6,
                    xytext=(3, -9), textcoords="offset points")
ax.set_xscale("log", base=2); ax.set_xlabel("reduction-axis tile width kt")
ax.set_ylabel("TFLOPS (median)")
ax.set_title("16384x16384 GEMM, M=1024\nrate tracks kt, not tile bytes")
ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=7)

ax = axes[2]
sel = [r for r in t2d if r["K"] == 16384]
ax.scatter([r["tile_MB"] for r in sel], [r["tops_med"] for r in sel],
           c=[r["kt"] for r in sel], cmap="viridis", norm=matplotlib.colors.LogNorm(), s=60)
for r in sel:
    ax.annotate(f"kt={r['kt']}", (r["tile_MB"], r["tops_med"]), fontsize=6, xytext=(4, 3),
                textcoords="offset points")
ax.axvline(32, ls=":", color="k", lw=1)
ax.text(33, 3, "32 MB", fontsize=7)
ax.set_xscale("log", base=2); ax.set_xlabel("weight tile size, MB")
ax.set_ylabel("TFLOPS (median)")
ax.set_title("Same data vs tile bytes\nno cliff at 32 MB")
ax.grid(True, which="both", alpha=0.3)

ax = axes[3]
for prec, col in (("fp16", "C0"), ("int8", "C2")):
    sel = sorted([r for r in ct if r["precision"] == prec], key=lambda r: r["kt"])
    if sel:
        ax.plot([r["kt"] for r in sel], [r["tops_med"] for r in sel], marker="o", color=col, label=prec)
ax.axhline(5.0, ls=":", color="r", lw=1)
ax.text(1100, 5.2, "GPU (MLX) fp16 ceiling", fontsize=7, color="r")
ax.set_xscale("log", base=2); ax.set_xlabel("reduction-axis tile width kt")
ax.set_ylabel("TFLOPS / TOPS (median)")
ax.set_title("8 chained 4096x4096 layers, M=1024\nthe step 2b configuration")
ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("results/roofline_aneforge.png", dpi=130)
print("wrote results/roofline_aneforge.png")
