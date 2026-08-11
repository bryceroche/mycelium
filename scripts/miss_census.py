import os, sys, json, re
for k,v in [("ALG2","1"),("ALG_FTYPES","8"),("ALG_HW","512"),("ALG_DUP","1"),("ALG_WIDE","1")]: os.environ.setdefault(k,v)
os.environ.setdefault("ALG_TEST",".cache/algebra_nl_bigtest.jsonl"); os.environ.setdefault("ALG_TEST_NAME","bigtest")
import json as _j
_GATE = os.environ.get("CENSUS_CKPT") or _j.load(open(".cache/GENERATION.json"))["parser_ckpt"]
os.environ["ALG_CKPT"]=_GATE
sys.path.insert(0,'.'); sys.path.insert(0,'scripts')
import numpy as np
from collections import Counter
from phase1_algebra_head import T_ALG, build_params, forward, decode, load_alg, build_slot_masks
from tta_alg2_dials import solve2
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load
samples, states, tokmask, gold, sent = load_alg("test")
p=build_params(0); sd=safe_load(_GATE)
for k in p: p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
ORD=re.compile(r"(first|second|third|fourth|fifth) number")
SUB=re.compile(r"minus|difference|exceeds|reduced|away from|less than|fewer|decreased")
miss_fams=Counter(); n_miss=0; MISS_IDX=[]
for s0 in range(0,len(samples),8):
    sl=np.arange(s0,min(s0+8,len(samples)))
    pad=8-len(sl); slp=np.concatenate([sl,sl[:1].repeat(pad)]) if pad else sl
    tr=Tensor(states[slp].astype(np.float32),dtype=dtypes.float)
    tk=Tensor(tokmask[slp].astype(np.float32),dtype=dtypes.float)
    se=Tensor(sent[slp].astype(np.int32),dtype=dtypes.int)
    o=forward(p,tr,tk,se)
    if os.environ.get("TWO_PASS")=="1" and "W_bo" in p:
        _o0={k:o[k].realize().numpy() for k in ("fat","args","res")}
        _mk=build_slot_masks(_o0, sent[slp])
        from tinygrad import Tensor as _T, dtypes as _dt
        o=forward(p,tr,tk,se,slot_mask=_T(_mk,dtype=_dt.float))
    keys=("pres","ftype","op","islit","dig","args","res","query")+(("sel",) if "sel" in o else ())+(("dup",) if "dup" in o else ())+(("sgn",) if "sgn" in o else ())
    onp={k:o[k].realize().numpy() for k in keys}
    for bi,ri in enumerate(sl):
        smp=samples[ri]
        facs,q=decode({k:onp[k][bi] for k in onp})
        a=solve2(facs,q,{"n_vars":24,"m":300})
        goldans=smp["solution"][smp["query_var"]]
        if a==goldans: continue
        n_miss+=1
        t=smp["text"]
        fams=[]
        if ORD.search(t): fams.append("ordinal")
        if SUB.search(t): fams.append("subsurface")
        ng=sum(1 for f in smp["factors"] if f.get("ftype")=="given")
        if ng>=6: fams.append("crowded6+")
        if not fams: fams.append("other")
        for f_ in fams: miss_fams[f_]+=1
        MISS_IDX.append(int(ri))
    if s0%400==0: print(f"  [{s0}/{len(samples)}]",flush=True)
print(f"[census] straight-view misses {n_miss}")
print("[families]", dict(miss_fams.most_common()))
tl=tokmask.sum(1)
mi=np.array(MISS_IDX); pa=np.array([i for i in range(len(samples)) if i not in set(MISS_IDX)])
print(f"[len] miss p50 {np.percentile(tl[mi],50):.0f} p90 {np.percentile(tl[mi],90):.0f} | pass p50 {np.percentile(tl[pa],50):.0f} p90 {np.percentile(tl[pa],90):.0f}")
print(f"[len] miss share above 189 (mix p90): {(tl[mi]>189).mean():.3f} | pass share: {(tl[pa]>189).mean():.3f}")
json.dump({'gate':_GATE,'n_miss':n_miss,'families':dict(miss_fams),'miss_idx':[int(x) for x in MISS_IDX],
 'miss_len_p50':float(np.percentile(tl[mi],50)),'pass_len_p50':float(np.percentile(tl[pa],50))},
 open(os.environ.get('CENSUS_OUT','.cache/miss_census_gen41.json'),'w'))
