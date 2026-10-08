"""Experiment 0 harness: patch bytes in a compiled ANE bundle, reload through
ANEForge's e5rt path in an isolated subprocess, classify the outcome, restore.

Outcomes:
  ok          ran, output matches the unpatched reference
  wrong       ran to completion, output differs (patch took effect silently)
  rejected    load or execute raised; the error text is recorded
  hang        subprocess exceeded the timeout; it is killed and ANE health is
              re-checked with a known-good program in a fresh process

Every test restores the original files from a backup before the next one, and
a health probe runs after any hang or rejection so a wedged engine is caught
immediately rather than corrupting the next result.
"""
import json
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.expanduser("~/Models/.aneforge-cache")
PY = os.path.expanduser("~/Research/ane-roofline/.venv/bin/python")
W_PATH = os.path.expanduser("~/Research/ane-roofline/exp0_W.npy")
REF_PATH = os.path.expanduser("~/Research/ane-roofline/exp0_ref.npy")
N, K, M = 2048, 4096, 64

CHILD = r'''
import sys, json, warnings, numpy as np
warnings.filterwarnings("ignore")
import aneforge as af
W = np.load(sys.argv[1]); ref = np.load(sys.argv[2])
try:
    x = af.input([%d, %d]); p = af.compile(x.linear(W))
    xi = np.random.default_rng(1).standard_normal((%d, %d)).astype(np.float16)
    out = np.asarray(p(xi), dtype=np.float32)
    rel = float(np.abs(out - ref).max() / (np.abs(ref).max() + 1e-9))
    fin = bool(np.isfinite(out).all())
    print(json.dumps({"status": "ran", "max_rel_err": rel, "finite": fin,
                      "build_s": getattr(p, "_build_s", None)}))
    p.release()
except Exception as e:
    print(json.dumps({"status": "raised", "type": type(e).__name__, "msg": str(e)[:600]}))
''' % (M, K, M, K)


def entry_dir():
    ds = [os.path.join(ROOT, d) for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d))]
    assert len(ds) == 1, f"expected exactly one cache entry, found {len(ds)}"
    return ds[0]


def find(root, name):
    for r, _, fs in os.walk(root):
        for f in fs:
            if f == name or f.endswith(name):
                return os.path.join(r, f)
    return None


def run_child(timeout_s=90):
    t0 = time.perf_counter()
    try:
        cp = subprocess.run([PY, "-c", CHILD, W_PATH, REF_PATH], capture_output=True,
                            text=True, timeout=timeout_s)
        el = time.perf_counter() - t0
        line = [l for l in cp.stdout.splitlines() if l.startswith("{")]
        if line:
            d = json.loads(line[-1]); d["elapsed_s"] = round(el, 2); d["rc"] = cp.returncode
            if d["status"] == "raised":
                d["stderr_tail"] = cp.stderr[-600:]
            return d
        return {"status": "no_output", "rc": cp.returncode, "elapsed_s": round(el, 2),
                "stderr_tail": cp.stderr[-800:], "stdout_tail": cp.stdout[-300:]}
    except subprocess.TimeoutExpired as e:
        return {"status": "hang", "timeout_s": timeout_s,
                "stderr_tail": (e.stderr or b"")[-600:].decode(errors="replace") if isinstance(e.stderr, bytes) else str(e.stderr)[-600:]}


def classify(d, tol=1e-3):
    if d["status"] == "hang":
        return "hang"
    if d["status"] == "raised" or d["status"] == "no_output":
        return "rejected"
    if not d.get("finite", True):
        return "wrong(nonfinite)"
    return "ok" if d["max_rel_err"] < tol else "wrong"


def health_probe():
    """Fresh process, unpatched graph must run and match."""
    return run_child(timeout_s=120)


def backup(paths):
    for p in paths:
        shutil.copy2(p, p + ".orig")


def restore(paths):
    for p in paths:
        if os.path.exists(p + ".orig"):
            shutil.copy2(p + ".orig", p)


def patch_byte(path, offset, value=None, xor=0xFF):
    with open(path, "r+b") as f:
        f.seek(offset)
        b = f.read(1)
        nb = bytes([value]) if value is not None else bytes([b[0] ^ xor])
        f.seek(offset)
        f.write(nb)
    return b[0], nb[0]


def zero_range(path, start, end):
    with open(path, "r+b") as f:
        f.seek(start)
        f.write(b"\x00" * (end - start))


def main():
    d = entry_dir()
    e5 = find(d, ".e5")
    hsh = find(d, "model.anehash")
    wb = os.path.join(d, "weights.bin")
    mil = os.path.join(d, "model.mil")
    files = [e5, hsh, wb, mil]
    print(json.dumps({"entry": d, "e5": os.path.relpath(e5, d), "e5_bytes": os.path.getsize(e5),
                      "anehash_bytes": os.path.getsize(hsh), "weights_bytes": os.path.getsize(wb)}))
    backup(files)

    # reference output from the pristine bundle
    if not os.path.exists(REF_PATH):
        import numpy as np
        W = np.load(W_PATH)
        xi = np.random.default_rng(1).standard_normal((M, K)).astype(np.float16)
        np.save(REF_PATH, (xi.astype(np.float32) @ W.astype(np.float32).T))
    base = run_child()
    print(json.dumps({"test": "baseline_pristine", "result": base, "verdict": classify(base)}), flush=True)

    tests = json.loads(sys.argv[1]) if len(sys.argv) > 1 else []
    results = []
    for t in tests:
        restore(files)
        name = t["name"]
        target = {"e5": e5, "hash": hsh, "weights": wb, "mil": mil}[t["file"]]
        note = {}
        if t.get("zero"):
            zero_range(target, t["zero"][0], t["zero"][1]); note = {"zeroed": t["zero"]}
        if "offset" in t:
            old, new = patch_byte(target, t["offset"], t.get("value"), t.get("xor", 0xFF))
            note.update({"offset": t["offset"], "old": old, "new": new})
        r = run_child(timeout_s=t.get("timeout", 90))
        v = classify(r)
        rec = {"test": name, "file": t["file"], **note, "result": r, "verdict": v}
        if v in ("hang", "rejected"):
            restore(files)
            hp = health_probe()
            rec["health_after"] = classify(hp)
            rec["health_detail"] = {k: hp.get(k) for k in ("status", "max_rel_err", "elapsed_s", "type")}
        results.append(rec)
        print(json.dumps(rec), flush=True)
    restore(files)
    for p in files:
        try:
            os.remove(p + ".orig")
        except OSError:
            pass
    out = os.path.expanduser("~/Research/ane-roofline/results/exp0_patches.jsonl")
    with open(out, "a") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
