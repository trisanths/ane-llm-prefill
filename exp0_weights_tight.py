"""Weights patch, judged bitwise against the PRISTINE ANE output, not fp32."""
import os, json, subprocess, sys, shutil
import numpy as np
ROOT=os.path.expanduser("~/Models/.aneforge-cache"); PY=sys.executable
d=[os.path.join(ROOT,x) for x in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT,x))][0]
wb=os.path.join(d,"weights.bin")
CHILD=r'''
import sys,warnings,numpy as np,os; warnings.filterwarnings("ignore"); import aneforge as af
W=np.load(os.path.expanduser("~/Research/ane-roofline/exp0_W.npy"))
x=af.input([64,4096]); p=af.compile(x.linear(W))
xi=np.random.default_rng(1).standard_normal((64,4096)).astype(np.float16)
out=np.asarray(p(xi),dtype=np.float16); np.save(sys.argv[1],out); p.release(); print("ok")
'''
def run(tag):
    o=f"/private/tmp/claude-501/-Users-pyro-Research/586113ea-37c2-4108-812a-de992699169f/scratchpad/exp0_out_{tag}.npy"
    cp=subprocess.run([PY,"-c",CHILD,o],capture_output=True,text=True,timeout=120)
    if "ok" not in cp.stdout: return None, cp.stderr[-400:]
    return np.load(o), None
shutil.copy2(wb, wb+".orig")
pristine,err=run("pristine"); assert pristine is not None, err
again,_=run("pristine2")
print(json.dumps({"pristine_vs_pristine_max_abs":float(np.abs(pristine.astype(np.float32)-again.astype(np.float32)).max())}))
W=np.load(os.path.expanduser("exp0_W.npy")); K=4096
tests=[("row100_low_byte_xor", 64+ (100*K+0)*2 , None, 0xFF),
       ("row100_high_byte_0x7C_makes_huge", 64+(100*K+0)*2+1, 0x7C, None),   # exp field -> ~65504
       ("row100_high_byte_0x7E_NaN", 64+(100*K+0)*2+1, 0x7E, None),           # 0x7Exx = NaN
       ("row2047_last_elem_low", 64+(2047*K+4095)*2, None, 0xFF)]
for name,off,val,xor in tests:
    shutil.copy2(wb+".orig", wb)
    with open(wb,"r+b") as f:
        f.seek(off); b=f.read(1)[0]; nb= val if val is not None else (b ^ xor); f.seek(off); f.write(bytes([nb]))
    out,err=run(name)
    shutil.copy2(wb+".orig", wb)
    if out is None: print(json.dumps({"test":name,"verdict":"rejected","err":err})); continue
    diff=np.abs(out.astype(np.float32)-pristine.astype(np.float32))
    rows=np.where(diff.max(1)>0)[0]; cols=np.where(diff.max(0)>0)[0]
    print(json.dumps({"test":name,"offset":off,"old":b,"new":nb,"verdict":"wrong" if diff.max()>0 else "identical",
                      "max_abs_diff":float(diff.max()),"finite":bool(np.isfinite(out).all()),
                      "n_rows_changed":int(len(rows)),"cols_changed":cols[:6].tolist(),"n_cols_changed":int(len(cols))}))
os.remove(wb+".orig")
