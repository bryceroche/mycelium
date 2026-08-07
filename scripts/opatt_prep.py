"""opatt_prep.py — operand-attention GOLD sidecar (2026-08-07; arm 1 of
dup_staging_cure): for every dup-rel row in the mix, mark the SECOND
occurrence tokens of the repeated letter (a TARGET, never a text hint)."""
import os, sys, json, re
sys.path.insert(0,'.'); sys.path.insert(0,'scripts')
os.environ.setdefault("ALG2","1"); os.environ.setdefault("ALG_FTYPES","8")
os.environ.setdefault("ALG_HW","512"); os.environ.setdefault("ALG_DUP","1")
import numpy as np
from phase1_algebra_head import T_ALG, TOKENIZER_JSON
from tokenizers import Tokenizer
L="abcdefghijklmnopqrstuvwx"
tok=Tokenizer.from_file(TOKENIZER_JSON)
rows=[json.loads(l) for l in open('.cache/gen23_mix.jsonl')]
OP=np.zeros((len(rows),24,T_ALG),np.uint8)
n_marked=0
for ri,r in enumerate(rows):
    dups=[(si,f) for si,f in enumerate(r.get("factors",[])) if f.get("ftype")=="rel" and len(f.get("args",[]))==2 and f["args"][0]==f["args"][1]]
    if not dups: continue
    t=r["text"]; e=tok.encode(t); offs=list(e.offsets)[:T_ALG]
    for si,f in dups:
        let=L[f["args"][0]]
        occ=[m.start() for m in re.finditer(r'\b'+let+r'\b',t)]
        if len(occ)<3: continue          # roster + 1st + 2nd use minimum
        # the repeated pair = the two occurrences inside ONE sentence:
        best=None
        for i in range(len(occ)-1):
            seg=t[occ[i]:occ[i+1]]
            if "." not in seg: best=(occ[i],occ[i+1])
        if best is None: continue
        c2=best[1]
        for ti,(cs,ce) in enumerate(offs):
            if ce>cs and cs<=c2<ce: OP[ri,si,ti]=1; n_marked+=1; break
    if ri%20000==0: print(f"  {ri}/{len(rows)}",flush=True)
np.save('.cache/opatt_gold.npy',OP)
print(f"[opatt] rows {len(rows)}; marked slots {n_marked}")
