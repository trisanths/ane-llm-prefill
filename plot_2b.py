"""Plots for results/block_bench.md (step 2b)."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

rows = [json.loads(l) for l in open("results/block_bench.jsonl")]
ok = [r for r in rows if "error" not in r]
blk = [r for r in ok if r["bench"] == "block"]
mlp = [r for r in ok if r["bench"] == "mlp"]
cmp_ = [r for r in ok if r["bench"] == "compare"]

fig, axes = plt.subplots(1, 3, figsize=(17, 5))

ax = axes[0]
names = ["attention\ncore", "full block\nfp16", "MLP only\nfp16", "full block\nint8"]
vals = [1.38,
        next(r["tops_total"] for r in blk if r["M"] == 1024 and r["kt"] == 2560
             and r["precision"] == "fp16" and not r["ln_trick"]),
        next(r["tops_med"] for r in mlp if r["M"] == 1024),
        next((r["tops_total"] for r in blk if r["M"] == 1024 and r["precision"] == "int8"), 0)]
bars = ax.bar(range(4), vals, color=["C3", "C0", "C2", "C4"])
ax.axhline(6.0, ls="--", color="k", lw=1)
ax.text(2.5, 6.15, "2b target, 6 TFLOPS", fontsize=7)
ax.axhline(4.85, ls=":", color="r", lw=1)
ax.text(-0.4, 4.35, "GPU on the same block", fontsize=7, color="r")
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.15, f"{v:.2f}", ha="center", fontsize=8)
ax.set_xticks(range(4)); ax.set_xticklabels(names, fontsize=8)
ax.set_ylabel("TFLOPS / TOPS (median)")
ax.set_title("Qwen3-14B-shaped block, M=1024\nattention is the weak part")
ax.grid(True, axis="y", alpha=0.3)

ax = axes[1]
sel = sorted([r for r in blk if r["M"] == 1024 and r["precision"] == "fp16"
              and not r["ln_trick"] and r["attn"] == "manual"], key=lambda r: r["kt"])
ax.plot([r["kt"] for r in sel], [r["tops_total"] for r in sel], marker="o", color="C0")
for r in sel:
    ax.annotate(f"{r['tops_total']:.2f}", (r["kt"], r["tops_total"]), fontsize=7,
                xytext=(4, 4), textcoords="offset points")
ax.set_xscale("log", base=2)
ax.set_xlabel("reduction-axis tile width kt")
ax.set_ylabel("TFLOPS (median)")
ax.set_title("Step 2a's tiling law holds on a real block")
ax.grid(True, which="both", alpha=0.3)

ax = axes[2]
sel = sorted(cmp_, key=lambda r: r["M"])
ms = [r["M"] for r in sel]
w = 0.35
ax.bar([i - w / 2 for i in range(len(ms))], [r["ane_tops"] for r in sel], w, label="ANE (ANEForge)", color="C0")
ax.bar([i + w / 2 for i in range(len(ms))], [r["gpu_tops"] for r in sel], w, label="GPU (MLX)", color="C3")
for i, r in enumerate(sel):
    ax.text(i, max(r["ane_tops"], r["gpu_tops"]) + 0.15, f"{r['speedup_ane_over_gpu']:.2f}x",
            ha="center", fontsize=8)
ax.set_xticks(range(len(ms))); ax.set_xticklabels([f"M={m}" for m in ms])
ax.set_ylabel("TFLOPS (median)")
ax.set_title("Same block, same weights, both engines")
ax.grid(True, axis="y", alpha=0.3); ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("results/block_bench.png", dpi=130)
print("wrote results/block_bench.png")
