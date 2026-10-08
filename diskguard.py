"""Disk hygiene for ANEForge work.

Two problems this addresses, both hit repeatedly in this project:

1. The compile cache is unbounded. Every distinct graph, every prompt-length
   bucket and every precision is a separate directory holding that program's
   baked weights. A single session reached 20 GB twice and filled the volume.

2. A full volume surfaces as `ane_e5rt_program_compile failed (mask=0x4)`, which
   reads like a compiler verdict. The real cause, `[Errno 28] No space left on
   device`, reached the surface once by accident. Worse, the failed compile
   leaves a poisoned cache entry, so the error then reproduces from fresh
   processes even after space is freed.

So: cap the cache, and stamp free space into every result row.
"""
import json
import os
import shutil
import time

CACHE = os.path.expanduser("~/Models/.aneforge-cache")


def free_gb(path="~"):
    s = os.statvfs(os.path.expanduser(path))
    return s.f_bavail * s.f_frsize / 1e9


def cache_size_gb(cache=CACHE):
    total = 0
    for root, _, files in os.walk(cache):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total / 1e9


def entries(cache=CACHE):
    """(path, mtime, bytes) per top-level cache entry, oldest first."""
    out = []
    if not os.path.isdir(cache):
        return out
    for name in os.listdir(cache):
        p = os.path.join(cache, name)
        if not os.path.isdir(p):
            continue
        sz = 0
        newest = 0.0
        for root, _, files in os.walk(p):
            for f in files:
                fp = os.path.join(root, f)
                try:
                    st = os.stat(fp)
                except OSError:
                    continue
                sz += st.st_size
                newest = max(newest, st.st_mtime)
        out.append((p, newest, sz))
    out.sort(key=lambda e: e[1])
    return out


def cap_cache(limit_gb=6.0, min_free_gb=10.0, cache=CACHE, verbose=True):
    """Evict oldest entries until the cache is under limit_gb and free space is
    at least min_free_gb. Returns what was removed."""
    removed, freed = [], 0
    ents = entries(cache)
    total = sum(e[2] for e in ents)
    while ents and (total / 1e9 > limit_gb or free_gb() < min_free_gb):
        p, _, sz = ents.pop(0)
        try:
            shutil.rmtree(p)
            removed.append(os.path.basename(p))
            freed += sz
            total -= sz
        except OSError:
            pass
    if verbose and removed:
        print(f"[diskguard] evicted {len(removed)} entries, freed {freed/1e9:.1f} GB, "
              f"cache now {total/1e9:.1f} GB, free {free_gb():.1f} GB")
    return {"evicted": len(removed), "freed_gb": freed / 1e9,
            "cache_gb": total / 1e9, "free_gb": free_gb()}


def stamp(record=None):
    """Disk facts to merge into a result row."""
    d = {"free_gb": round(free_gb(), 2), "cache_gb": round(cache_size_gb(), 2),
         "stamped_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if record is not None:
        record.update(d)
        return record
    return d


def guard(limit_gb=6.0, min_free_gb=10.0):
    """Call at the top of a benchmark: cap the cache and warn if space is short."""
    before = free_gb()
    r = cap_cache(limit_gb, min_free_gb)
    if free_gb() < min_free_gb:
        print(f"[diskguard] WARNING: only {free_gb():.1f} GB free after eviction "
              f"(wanted {min_free_gb}). Compiles may fail as mask=0x4.")
    elif before < min_free_gb:
        print(f"[diskguard] free space restored to {free_gb():.1f} GB")
    return r


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "cap":
        lim = float(sys.argv[2]) if len(sys.argv) > 2 else 6.0
        print(json.dumps(cap_cache(lim), indent=2))
    else:
        print(json.dumps({"free_gb": round(free_gb(), 2),
                          "cache_gb": round(cache_size_gb(), 2),
                          "entries": len(entries())}, indent=2))
