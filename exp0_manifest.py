"""Is the .e5 manifest / anehash / mil consumed on reload? Track regeneration."""
import os, json, subprocess, sys, shutil, hashlib, time
import numpy as np
ROOT=os.path.expanduser("~/Models/.aneforge-cache"); PY=sys.executable
SCR="/private/tmp/claude-501/-Users-pyro-Research/586113ea-37c2-4108-812a-de992699169f/scratchpad"
d=[os.path.join(ROOT,x) for x in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT,x))][0]
def find(name):
    for r,_,fs in os.walk(d):
        for f in fs:
            if f.endswith(name): return os.path.join(r,f)
e5=find(".e5"); hsh=find("model.anehash"); mil=os.path.join(d,"model.mil"); wb=os.path.join(d,"weights.bin")
bdir=os.path.join(d,"cache/com.apple.e5rt.e5bundlecache/26A5425a")
files=[e5,hsh,mil,wb]
for f in files: shutil.copy2(f,f+".orig")
def sig(p):
    if not os.path.exists(p): return "MISSING"
    return f"{hashlib.sha256(open(p,'rb').read()).hexdigest()[:10]}@{int(os.path.getmtime(p))}"
CHILD=r'''
import sys,warnings,numpy as np,os; warnings.filterwarnings("ignore"); import aneforge as af
W=np.load(os.path.expanduser("~/Research/ane-roofline/exp0_W.npy"))
x=af.input([64,4096]); p=af.compile(x.linear(W))
xi=np.random.default_rng(1).standard_normal((64,4096)).astype(np.float16)
out=np.asarray(p(xi),dtype=np.float16); np.save(sys.argv[1],out); p.release(); print("ok")
'''
def run(tag,timeout=120):
    o=f"{SCR}/m_{tag}.npy"
    try:
        cp=subprocess.run([PY,"-c",CHILD,o],capture_output=True,text=True,timeout=timeout)
    except subprocess.TimeoutExpired: return None,"HANG"
    if "ok" not in cp.stdout: return None,(cp.stderr.strip().splitlines() or ["?"])[-1][:300]
    return np.load(o),None
pristine,_=run("pristine"); assert pristine is not None
def restore():
    for f in files:
        if os.path.exists(f+".orig"): shutil.copy2(f+".orig",f)
def patch(p,off,val):
    with open(p,"r+b") as f: f.seek(off); f.write(bytes([val]))
tests={
 "e5_root_offset_byte0=0xFF": lambda: patch(e5,0,0xFF),
 "e5_truncate_to_100B": lambda: open(e5,"r+b").truncate(100),
 "e5_deleted": lambda: os.remove(e5),
 "e5_N_field_0x454_xor": lambda: patch(e5,0x454, open(e5,'rb').read()[0x454]^0xFF),
 "anehash_zeroed": lambda: open(hsh,"wb").write(b"\x00"*129),
 "anehash_deleted": lambda: os.remove(hsh),
 "whole_bundle_dir_deleted": lambda: shutil.rmtree(bdir),
 "mil_weight_shape_2048->2047": lambda: open(mil,"w").write(open(mil+".orig").read().replace("[2048, 4096]> t1_w","[2047, 4096]> t1_w",1)),
 "mil_transpose_y_true->false": lambda: open(mil,"w").write(open(mil+".orig").read().replace("transpose_y = bool(true)","transpose_y = bool(false)",1)),
 "mil_blob_offset_64->66": lambda: open(mil,"w").write(open(mil+".orig").read().replace("offset = uint64(64)","offset = uint64(66)",1)),
}
for name,fn in tests.items():
    restore(); nb=len(os.listdir(bdir)) if os.path.isdir(bdir) else 0
    fn(); before={k:sig(v) for k,v in {"e5":e5,"anehash":hsh,"mil":mil}.items()}
    t0=time.time(); out,err=run(name); el=round(time.time()-t0,2)
    after={k:sig(v) for k,v in {"e5":e5,"anehash":hsh,"mil":mil}.items()}
    regen=[k for k in before if before[k]!=after[k]]
    na=len(os.listdir(bdir)) if os.path.isdir(bdir) else 0
    rec={"test":name,"elapsed_s":el,"new_bundle_dirs":na-nb,"files_regenerated_by_e5rt":regen}
    if out is None: rec.update(verdict="hang" if err=="HANG" else "rejected", error=err)
    else:
        diff=np.abs(out.astype(np.float32)-pristine.astype(np.float32)); 
        rec.update(verdict="identical" if float(np.nan_to_num(diff,nan=1).max())==0 else "wrong", max_abs_diff=float(diff.max()), finite=bool(np.isfinite(out).all()))
    print(json.dumps(rec),flush=True)
restore()
for f in files:
    try: os.remove(f+".orig")
    except OSError: pass
