import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

rows = [json.loads(l) for l in open("results/attn.jsonl") if "error" not in json.loads(l)]
spl = [json.loads(l) for l in open("results/split.jsonl")]

fig, axes = plt.subplots(1, 3, figsize=(17, 5))

ax = axes[0]
for M, c in ((512, "C0"), (1024, "C1"), (2048, "C3")):
    sel = sorted([r for r in rows if r["M"] == M and r["mode"] == "blocked"], key=lambda r: r["block"])
    full = [r for r in rows if r["M"] == M and r["mode"] == "full"]
    if sel:
        ax.plot([r["block"] for r in sel], [r["tflops_med"] for r in sel], marker="o", color=c, label=f"M={M} blocked")
    if full:
        ax.axhline(full[0]["tflops_med"], ls=":", color=c, lw=1)
ax.text(140, 1.16, "dotted = full M x M baseline, same colour", fontsize=7)
ax.set_xscale("log", base=2)
ax.set_xlabel("key block size B"); ax.set_ylabel("TFLOPS (median)")
ax.set_title("Key-axis blocked attention\nwin grows with M")
ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=8)

ax = axes[1]
for M, c in ((512, "C0"), (1024, "C1"), (2048, "C3")):
    sel = sorted([r for r in rows if r["M"] == M], key=lambda r: r["peak_score_MB"])
    ax.plot([r["peak_score_MB"] for r in sel], [r["tflops_med"] for r in sel], marker="o", color=c, label=f"M={M}")
ax.set_xscale("log", base=2)
ax.set_xlabel("peak score tensor, MB"); ax.set_ylabel("TFLOPS (median)")
ax.set_title("Throughput against the largest intermediate")
ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=8)

ax = axes[2]
r = [x for x in spl if x["M"] == 1024][0]
parts = ["ANE\nfront", "host->\nMLX", "GPU\nattn", "MLX->\nhost", "ANE\nback"]
vals = [r["ane_front_ms"], r["host_to_mlx_ms"], r["gpu_attn_ms"], r["mlx_to_host_ms"], r["ane_back_ms"]]
cols = ["C0", "C7", "C3", "C7", "C0"]
ax.bar(range(5), vals, color=cols)
ax.axhline(r["monolithic_ane_ms"], ls="--", color="k", lw=1)
ax.text(0.1, r["monolithic_ane_ms"] + 1.5, f"monolithic ANE block: {r['monolithic_ane_ms']:.0f} ms", fontsize=7)
ax.plot([], [], color="C7", label=f"transfer total {r['transfer_ms']:.1f} ms ({100*r['transfer_frac_of_split']:.0f}%)")
ax.plot([], [], color="C3", label=f"ANE blocked attn would be {r['ane_attn_blocked_ms']:.1f} ms")
ax.set_xticks(range(5)); ax.set_xticklabels(parts, fontsize=7)
ax.set_ylabel("ms (median)")
ax.set_title(f"Per-layer split at M=1024\nsplit {r['split_total_ms']:.0f} ms vs monolithic {r['monolithic_ane_ms']:.0f} ms")
ax.grid(True, axis="y", alpha=0.3); ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig("results/attention.png", dpi=130)
print("wrote results/attention.png")
