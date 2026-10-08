"""Swap gate + results-file header. User standing rule (2026-09-27):
no measurement if swap > 2 GB; log swap and free RAM at the top of every
results file; target swap < 1 GB.

Import and call require_clear() at the top of any measurement script, and
header() to stamp every results file.
"""
import subprocess, sys, time

HARD_LIMIT_GB = 2.0
TARGET_GB = 1.0

def swap_used_gb():
    out = subprocess.check_output(["sysctl", "-n", "vm.swapusage"], text=True)
    # "total = 6144.00M  used = 5502.00M  free = 642.00M"
    return float(out.split("used =")[1].split("M")[0].strip()) / 1024

def free_ram_gb():
    out = subprocess.check_output(["vm_stat"], text=True)
    free = int(out.split("Pages free:")[1].split(".")[0])
    spec = int(out.split("Pages speculative:")[1].split(".")[0])
    return (free + spec) * 16384 / 1e9

def state():
    return {"swap_used_gb": round(swap_used_gb(), 2), "free_ram_gb": round(free_ram_gb(), 2)}

def require_clear(what="measurement"):
    s = swap_used_gb()
    st = state()
    if s > HARD_LIMIT_GB:
        print(f"GATE FAIL: swap {s:.2f} GB > {HARD_LIMIT_GB} GB hard limit. "
              f"{what} refused. Free RAM {st['free_ram_gb']:.2f} GB. "
              f"Quit apps until swap < {TARGET_GB} GB.", file=sys.stderr)
        sys.exit(3)
    if s > TARGET_GB:
        print(f"GATE WARN: swap {s:.2f} GB is under the {HARD_LIMIT_GB} GB limit "
              f"but above the {TARGET_GB} GB target; numbers may be a floor.", file=sys.stderr)
    return st

def header(path, title):
    """Write/overwrite a results file with a swap-state header block."""
    st = state()
    with open(path, "w") as f:
        f.write(f"# {title}\n\n")
        f.write(f"Machine state at run: swap {st['swap_used_gb']} GB, "
                f"free RAM {st['free_ram_gb']} GB. "
                f"(gate: <{TARGET_GB} GB target, <{HARD_LIMIT_GB} GB hard limit)\n\n")
    return st

if __name__ == "__main__":
    import json
    print(json.dumps(state()))
    s = swap_used_gb()
    print("PASS" if s < TARGET_GB else "MARGINAL" if s < HARD_LIMIT_GB else "FAIL")
