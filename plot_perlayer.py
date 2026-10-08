"""Per-layer residual drift and the W8A16 delta."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

pl = [json.loads(l) for l in open("results/perlayer.jsonl") if "error" not in json.loads(l)]
w8 = [json.loads(l) for l in open("results/w8a16.jsonl") if "error" not in json.loads(l)]

fig, axes = plt.subplots(1, 3, figsize=(17, 5))

ax = axes[0]
style = {("fp16", 1024): ("C0", "-"), ("fp16", 2048): ("C0", "--"),
         ("int8", 1024): ("C2", "-"), ("int8", 2048): ("C2", "--")}
for r in pl:
    c, ls = style.get((r["precision"], r["kt"]), ("grey", ":"))
    lab = ("fp16" if r["precision"] == "fp16" else "W8A16") + f", kt={r['kt']}"
    ax.plot(range(1, len(r["cos_vs_fp32"]) + 1), r["cos_vs_fp32"], ls, color=c, marker="o", ms=4, label=lab)
ax.axhline(0.999, ls=":", color="r", lw=1)
ax.text(1, 0.99905, "0.999 acceptance bar", fontsize=7, color="r")
ax.set_xlabel("layer depth"); ax.set_ylabel("cosine vs fp32 reference")
ax.set_title("Residual-stream drift with depth\ntile width changes speed, not accuracy")
ax.grid(True, alpha=0.3); ax.legend(fontsize=7)

ax = axes[1]
for r in pl:
    if "cos_vs_fp16" not in r:
        continue
    n = len(r["cos_vs_fp16"])
    ax.plot(range(1, n + 1), r["cos_vs_fp16"], marker="o", ms=4, label=f"W8A16 vs fp16, kt={r['kt']}")
    slope = (1 - r["cos_vs_fp16"][-1]) / n
    xs = list(range(1, 41))
    ax.plot(xs, [1 - slope * x for x in xs], ":", lw=1,
            label=f"extrapolated, {slope:.1e}/layer")
ax.axhline(0.999, ls=":", color="r", lw=1)
ax.set_xlabel("layer depth"); ax.set_ylabel("cosine, W8A16 vs fp16")
ax.set_title("W8A16 against fp16 on identical inputs\nextrapolated to model depth")
ax.grid(True, alpha=0.3); ax.legend(fontsize=7)

ax = axes[2]
f = {r["M"]: r for r in w8 if r["precision"] == "fp16"}
i = {r["M"]: r for r in w8 if r["precision"] == "int8"}
ms = sorted(set(f) & set(i))
ax.plot(ms, [i[m]["tops_total"] / f[m]["tops_total"] for m in ms], marker="o", color="C2",
        label="block, kt=2048 (tiled)")
tile = {1024: 1.12, 2048: 1.16, 4096: 1.53}
ax.scatter([1024] * 0 + [], [])
ax.axhline(1.62, ls="--", color="C1", lw=1)
ax.text(300, 1.64, "untiled chain: 1.62x", fontsize=7, color="C1")
ax.axhline(1.0, ls=":", color="k", lw=1)
ax.text(300, 1.02, "no gain", fontsize=7)
ax.set_xscale("log", base=2)
ax.set_xlabel("M (rows)"); ax.set_ylabel("W8A16 speedup over fp16")
ax.set_ylim(0.95, 1.8)
ax.set_title("The int8 gain mostly disappears\nonce the reduction is tiled")
ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig("results/perlayer_w8a16.png", dpi=130)
print("wrote results/perlayer_w8a16.png")
