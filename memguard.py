"""Resident and swap accounting, so a measurement can be marked invalid."""
import os
import resource
import subprocess


def swap_used_gb():
    try:
        out = subprocess.check_output(["sysctl", "-n", "vm.swapusage"], text=True)
        # "total = 17408.00M  used = 16052.06M  free = 1355.94M"
        used = out.split("used =")[1].split("M")[0].strip()
        return float(used) / 1024
    except Exception:
        return float("nan")


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9


def free_ram_gb():
    try:
        out = subprocess.check_output(["vm_stat"], text=True)
        page = 16384
        free = int(out.split("Pages free:")[1].split(".")[0])
        spec = int(out.split("Pages speculative:")[1].split(".")[0])
        return (free + spec) * page / 1e9
    except Exception:
        return float("nan")


def stamp(rec=None):
    d = {"rss_gb": round(rss_gb(), 2), "swap_used_gb": round(swap_used_gb(), 2),
         "free_ram_gb": round(free_ram_gb(), 2)}
    if rec is not None:
        rec.update(d)
        return rec
    return d
