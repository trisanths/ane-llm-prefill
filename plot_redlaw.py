import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

rows = [json.loads(l) for l in open("results/redlaw.jsonl") if "error" not in json.loads(l)]
a3 = [r for r in rows if r["M"] == 1024 and r["K"] * r["N"] == 4096 * 4096]

fig, axes = plt.subplots(1, 3, figsize=(17, 5))

ax = axes[0]
for p, c in (("fp16", "C0"), ("int8", "C2")):
    sel = sorted([r for r in a3 if r["precision"] == p], key=lambda r: r["K"])
    ax.plot([r["K"] for r in sel], [r["tflops_med"] for r in sel], marker="o", color=c,
            label="fp16" if p == "fp16" else "W8A16")
ax.set_xscale("log", base=2)
ax.set_xlabel("reduction width K  (N = 16.7M / K, volume fixed at 33.6 MB)")
ax.set_ylabel("TFLOPS (median)")
ax.set_title("3a: same weight bytes, same FLOPs\nonly the contraction shape changes")
ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=8)

ax = axes[1]
f = {r["K"]: r["tflops_med"] for r in a3 if r["precision"] == "fp16"}
i = {r["K"]: r["tflops_med"] for r in a3 if r["precision"] == "int8"}
ks = sorted(set(f) & set(i))
ax.plot(ks, [i[k] / f[k] for k in ks], marker="o", color="C2")
ax.axhline(1.0, ls=":", color="k", lw=1)
ax.axhline(2.0, ls="--", color="grey", lw=1)
ax.text(1100, 2.02, "a native 2x integer rate would sit here, flat", fontsize=7)
ax.set_xscale("log", base=2); ax.set_ylim(0.8, 2.3)
ax.set_xlabel("reduction width K"); ax.set_ylabel("W8A16 speedup over fp16")
ax.set_title("3a: the int8 gain depends on K\nso it is not a flat multiply rate")
ax.grid(True, which="both", alpha=0.3)

ax = axes[2]
by = {}
for r in rows:
    by.setdefault((r["K"], r["M"]), {})[r["precision"]] = r["tflops_med"]
for K, c, lab in ((1024, "C1", "narrow K=1024, N=16384"), (4096, "C3", "wide K=4096, N=4096")):
    ms = sorted(m for (k, m) in by if k == K and "fp16" in by[(k, m)])
    ms = [m for m in ms if m in (128, 256, 512, 1024, 2048)]
    if ms:
        ax.plot(ms, [by[(K, m)]["fp16"] for m in ms], marker="o", color=c, label=lab + ", fp16")
        ax.plot(ms, [by[(K, m)].get("int8", 0) for m in ms], marker="s", ls="--", color=c,
                label=lab + ", W8A16")
ax.set_xscale("log", base=2)
ax.set_xlabel("M (rows)"); ax.set_ylabel("TFLOPS (median)")
ax.set_title("3b: fixed volume, sweep M\nnarrow K wins once past dispatch")
ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig("results/reduction_law.png", dpi=130)
print("wrote results/reduction_law.png")
