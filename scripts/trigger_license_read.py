"""trigger_license_read.py — organ 2's license (2026-08-04; pins in
ledger). Literal slots on bigtest under rings ckpt: proxy = attention
mass at claimed-value token positions; licensed by FP on correct and
agreement with gold-span mislocation."""
import os, sys, json, re
os.environ["ALG_BREATH"]="3"; os.environ["ALG_RINGS"]="1"
os.environ.setdefault("ALG2","1"); os.environ.setdefault("ALG_FTYPES","8")
os.environ.setdefault("ALG_HW","512"); os.environ.setdefault("ALG_DUP","1")
os.environ.setdefault("ALG_WIDE","1")
os.environ.setdefault("ALG_TEST",".cache/algebra_nl_bigtest.jsonl")
os.environ.setdefault("ALG_TEST_NAME","bigtest")
sys.path.insert(0,'.'); sys.path.insert(0,'scripts')
import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load
from phase1_algebra_head import (build_params, forward, load_alg, decode,
                                 build_slot_masks, L_FAC, TOKENIZER_JSON, T_ALG)
from tokenizers import Tokenizer
tok=Tokenizer.from_file(TOKENIZER_JSON)

samples, states, tokmask, gold, sent = load_alg("test")
p=build_params(0); sd=safe_load('.cache/g24_rings_rings.safetensors')
for k in p: p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
n=len(samples)
rows=[]
for s0 in range(0,n,8):
    sl=np.arange(s0,min(s0+8,n)); pad=8-len(sl)
    sl_p=np.concatenate([sl,sl[:1].repeat(pad)]) if pad else sl
    t_tr=Tensor(states[sl_p].astype(np.float32),dtype=dtypes.float)
    t_tk=Tensor(tokmask[sl_p].astype(np.float32),dtype=dtypes.float)
    t_se=Tensor(sent[sl_p].astype(np.int32),dtype=dtypes.int)
    o0=forward(p,t_tr,t_tk,t_se)
    onp0={k:o0[k].realize().numpy() for k in ("fat","args","res")}
    mk=build_slot_masks(onp0,sent[sl_p])
    o=forward(p,t_tr,t_tk,t_se,slot_mask=Tensor(mk,dtype=dtypes.float))
    keys=["pres","ftype","op","islit","dig","sgn","args","res","query","fat"]
    if "sel" in o: keys.append("sel")
    if "dup" in o: keys.append("dup")
    onp={k:o[k].realize().numpy() for k in keys}
    for bi,i in enumerate(sl):
        i=int(i); smp=samples[i]
        enc=tok.encode(smp["text"]); offs=list(enc.offsets)
        facs,q=decode({k:onp[k][bi] for k in onp if k!="fat"})
        for f in facs:
            if f.get("ftype")!="given": continue
            j=None
            # match decoded literal slot to head slot by res pointer:
            # find slot with islit>0 and res argmax == f var and value match
            for jj in range(L_FAC):
                if onp["pres"][bi,jj]>0 and onp["res"][bi,jj].argmax()==f["var"] \
                   and onp["ftype"][bi,jj].argmax()==1:
                    j=jj; break
            if j is None: continue
            # gold label: slot correct (ftype+res match at this slot)
            correct=(gold["ftype"][i,j]==1 and gold["res"][i,j]==onp["res"][bi,j].argmax()
                     and int(gold["digits"][i,j] @ (10**np.arange(gold["digits"].shape[-1]-1,-1,-1)))
                         ==abs(int(f["value"])))
            span=gold["fspan"][i,j]>0
            a=onp["fat"][bi,j]; a=a/max(a.sum(),1e-9)
            gold_mis = float(a[span].sum())<0.5 if span.any() else None
            # proxy: attention mass at claimed-value digit positions in text
            vs=str(abs(int(f["value"])))
            hits=[mm.span() for mm in re.finditer(re.escape(vs),smp["text"])]
            mask=np.zeros(T_ALG,bool)
            for ti,(cs,ce) in enumerate(offs[:T_ALG]):
                if ce<=cs: continue
                for (hs,he) in hits:
                    if cs<he and ce>hs: mask[ti]=True
            proxy = float(a[mask].sum()) if mask.any() else 0.0
            rows.append({"correct":bool(correct),"proxy":proxy,
                         "gold_mis":gold_mis,"found":bool(mask.any())})
r=[x for x in rows if x["found"] and x["gold_mis"] is not None]
pc=np.array([x["proxy"] for x in r]); yc=np.array([x["correct"] for x in r])
gm=np.array([x["gold_mis"] for x in r])
thr=0.5
flag=pc<thr
fp=float(flag[yc].mean()); catch=float(flag[~yc].mean()) if (~yc).any() else float('nan')
agree=float((flag==gm).mean())
print(f"[license] literal slots read n={len(r)} (correct {int(yc.sum())} wrong {int((~yc).sum())})")
print(f"[license] proxy-flag (mass<0.5 at claimed value): FP on correct {fp:.3%}  catch on wrong {catch:.1%}")
print(f"[license] agreement with gold-span mislocation verdict: {agree:.1%}")
v=("LICENSED" if fp<=0.02 and agree>=0.9 else
   "UNLICENSED-FP" if fp>0.02 else "PROXY-MEASURES-ELSE")
print(f"VERDICT (pinned): {v}")
json.dump({"n":len(r),"fp":fp,"catch":catch,"agree":agree,"verdict":v},
          open(".cache/trigger_license.json","w"),indent=1)
print("[saved] .cache/trigger_license.json")
