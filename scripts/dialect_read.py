"""dialect_read.py — door #45's probe-crossed bars: the dialect
reader's own accuracy on tail-wrong slots vs the majority reader,
plus the not-a-second-tongue agreement clause + shadow potential."""
import os, sys, json
if os.environ.get("DIAL_GATECHK")!="1": os.environ["ALG_DIAL"]="1"
for k,v in [("ALG2","1"),("ALG_FTYPES","8"),("ALG_HW","512"),("ALG_DUP","1"),("ALG_WIDE","1"),("ALG_INV","1")]: os.environ.setdefault(k,v)
os.environ.setdefault("ALG_TEST",".cache/algebra_nl_bigtest.jsonl"); os.environ.setdefault("ALG_TEST_NAME","bigtest")
sys.path.insert(0,'.'); sys.path.insert(0,'scripts')
import numpy as np, re
from phase1_algebra_head import T_ALG, build_params, forward, load_alg
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load
samples, states, tokmask, gold, sent = load_alg("test")
p=build_params(0); sd=safe_load(os.environ["CK"])
for k in p: p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
ORD=re.compile(r"(first|second|third|fourth|fifth) number")
TW=[]; TC=[]   # tail-wrong / tail-correct-indirect (slot-grain records)
for s0 in range(0,len(samples),8):
    sl=np.arange(s0,min(s0+8,len(samples)))
    pad=8-len(sl); slp=np.concatenate([sl,sl[:1].repeat(pad)]) if pad else sl
    o=forward(p,Tensor(states[slp].astype(np.float32),dtype=dtypes.float),
              Tensor(tokmask[slp].astype(np.float32),dtype=dtypes.float),
              Tensor(sent[slp].astype(np.int32),dtype=dtypes.int))
    keys=("args","iargs","pres","ftype") if os.environ.get("DIAL_GATECHK")!="1" else ("args","pres","ftype")
    onp={k2:o[k2].realize().numpy() for k2 in keys}
    if "iargs" not in onp: onp["iargs"]=onp["args"]
    for bi,ri in enumerate(sl):
        if not ORD.search(samples[ri]["text"]): continue
        for j,f in enumerate(samples[ri]["factors"]):
            if f.get("ftype")!="rel" or len(set(f.get("args",[])))!=2: continue
            if onp["pres"][bi,j]<=0 or onp["ftype"][bi,j].argmax()!=0: continue
            a0=sorted(f["args"])[0]
            am=int(np.argsort(-onp["args"][bi,j])[0]); am2=sorted(np.argsort(-onp["args"][bi,j])[:2].tolist())
            im=int(np.argsort(-onp["iargs"][bi,j])[0]); im2=sorted(np.argsort(-onp["iargs"][bi,j])[:2].tolist())
            ga=sorted(f["args"])
            rec=(ri,j,a0,am2,im2,ga)
            if am2!=ga: TW.append(rec)
            else: TC.append(rec)
tw_i=sum(1 for r in TW if r[4]==r[5])/max(len(TW),1)
tc_i=sum(1 for r in TC if r[4]==r[5])/max(len(TC),1)
print(f"[dialect] TAIL-WRONG slots n={len(TW)}: iargs full-args acc {tw_i:.3f}  (args-head 0.000 by construction)",flush=True)
print(f"[dialect] TAIL-CORRECT slots n={len(TC)}: iargs agrees-with-gold {tc_i:.3f}  (bar >=0.9)",flush=True)
rows_fixed=len(set(r[0] for r in TW if r[4]==r[5]))
print(f"[shadow] rows with >=1 wrong-arg slot fully corrected by iargs: {rows_fixed}",flush=True)
json.dump({"n_tw":len(TW),"acc_tw":tw_i,"acc_tc":tc_i,"rows_fixed":rows_fixed},
          open('.cache/dialect_read.json','w'))
