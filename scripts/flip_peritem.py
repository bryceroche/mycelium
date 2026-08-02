"""flip_peritem.py — THE PER-ITEM PASS (2026-08-02, the word given).
Closes the AUG fire's two open reads with per-item banking (the
bank-don't-read lesson applied prospectively):

  1. SLOT EMISSION beside binding, separately (the parseability
     rider): per-view non-None answer fraction, orig and transformed,
     per ckpt.
  2. THE MUL PREDICTION (pinned blind at fire design): mul's 68%
     table coverage (lowest; pct 96) predicts a SMALLER mul flip
     effect; equal-to-pct refutes coverage-as-mechanism.

PINNED BEFORE THIS RUN (nothing else bends after):
  - REPRODUCTION CHECK: same seeds as flip_corrected.py (101000/
    102000 + 10j) — per-ckpt aggregate flip counts must reproduce
    .cache/flip_corrected.json EXACTLY or the pass aborts (the
    instrument must be the same instrument).
  - SUPPORT FLOOR: the mul-vs-pct comparison reads only if BOTH
    family cells hold n>=15 counted items; below floor the
    prediction closes UNREADABLE-AT-SUPPORT (closed, not void).
  - EFFECT = g22_family_rate - arm_family_rate (positive = arm
    improved). Family tags mirror the census's word-substring
    grain (punctuation-blind BY CONSTRUCTION — the scope line
    rides the verdict).
"""
import sys, os, json, re
sys.path.insert(0,'.'); sys.path.insert(0,'scripts')
os.environ.setdefault("ALG2","1"); os.environ.setdefault("ALG_FTYPES","8")
os.environ.setdefault("ALG_HW","512"); os.environ.setdefault("ALG_DUP","1")
import numpy as np
from collections import Counter
from phase1_algebra_head import T_ALG, build_params, forward, decode, sent_indices, TOKENIZER_JSON
from beacon_closing_arm import recompute_states
from tta_views import permuted_view
from tta_alg2_dials import solve2
from binding_invariance_read import transform
from tokenizers import Tokenizer
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load
tok = Tokenizer.from_file(TOKENIZER_JSON)

FAMILY_PATTERNS = {  # mirrors .cache/harvest_phrasing_census.json grains
    "mul": r"\*|\\cdot|\\times|twice|double|product of|times|multipl",
    "add": r"\+|sum of|add|total|combined|more than|plus",
    "div": r"\\frac|/|\bper\b|half|third|divid|quotient|split|share",
    "sub": r"[\d)a-z]\s*-\s*[\da-z(]|less than|subtract|difference|fewer|exceeds",
    "pct": r"%|percent",
    "sel": r"\bmax|\bmin|larger|greater|smaller|least",
}
def families_of(text):
    tl = text.lower()
    return [f for f,p in FAMILY_PATTERNS.items() if re.search(p, tl)]

recs=[json.loads(l) for l in open('.cache/wild_ledger_v1.jsonl')]
ans=[r for r in recs if r["tier"]=="answered"]
h=[json.loads(l) for l in open('.cache/math_harvest_v0.jsonl')]
items=[]
for j,r in enumerate(ans):
    t=h[r["harvest_idx"]]["problem"]
    nt,kind=transform(t)
    if nt: items.append((j,t,nt,kind,families_of(t)))
print(f"[peritem] transformable items: {len(items)}/124")

def parse_batch(p, texts):
    n=len(texts); N=((n+7)//8)*8
    ids=np.zeros((N,T_ALG),np.int32); msk=np.zeros((N,T_ALG),np.float32); snt=np.zeros((N,T_ALG),np.int32)
    for i,t in enumerate(texts):
        e=tok.encode(t); Ln=min(len(e.ids),T_ALG)
        ids[i,:Ln]=e.ids[:Ln]; msk[i,:Ln]=1.0
        snt[i]=sent_indices(t,list(e.offsets),msk[i])
    st=recompute_states(ids)
    out_r=[]
    for s0 in range(0,N,8):
        out=forward(p,Tensor(st[s0:s0+8].astype(np.float32),dtype=dtypes.float),
                    Tensor(msk[s0:s0+8].astype(np.float32),dtype=dtypes.float),
                    Tensor(snt[s0:s0+8].astype(np.int32),dtype=dtypes.int))
        keys=("pres","ftype","op","islit","dig","args","res","query")+(("sel",) if "sel" in out else ())+(("dup",) if "dup" in out else ())
        o={k:out[k].realize().numpy() for k in keys}
        for bi in range(8):
            if s0+bi<n: out_r.append(decode({k:o[k][bi] for k in o}))
    return out_r

def quorum_full(p, text, base):
    vt=[text]+[permuted_view(text,base+k) for k in range(1,5)]
    a=[solve2(f,q,{"n_vars":24,"m":300}) for f,q in parse_batch(p,vt)]
    nn=[x for x in a if x is not None]
    c=Counter(nn).most_common(1)
    top = c[0] if c else (None,0)
    return top, len(nn)  # (quorum answer, count), emitted-view count

peritem={}
for name,ck in (("g22",".cache/g22.safetensors"),("vlow",".cache/g24_vlow.safetensors"),("vfull",".cache/g24_vfull.safetensors")):
    p=build_params(0); sd=safe_load(ck)
    for k in p: p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
    rows=[]
    for idx,(j,t,nt,kind,fams) in enumerate(items):
        (po,co),emo=quorum_full(p,t,101000+10*j)
        rec={"j":j,"kind":kind,"families":fams,"emit_orig":emo,"orig_count":co}
        if co<3: rec["status"]="uncounted"
        else:
            (pt,ct),emt=quorum_full(p,nt,102000+10*j)
            rec["emit_trans"]=emt
            rec["status"]="stable" if (ct>=3 and pt==po) else "flipped"
        rows.append(rec)
        if (idx+1)%25==0: print(f"  [{name} {idx+1}/{len(items)}]",flush=True)
    peritem[name]=rows
    fl=sum(1 for r in rows if r["status"]=="flipped"); st=sum(1 for r in rows if r["status"]=="stable")
    un=sum(1 for r in rows if r["status"]=="uncounted")
    print(f"[{name}] flip {fl}/{fl+st} = {fl/max(fl+st,1):.1%} (uncounted {un})",flush=True)

# ---- REPRODUCTION CHECK (aborts the reads if the instrument drifted) ----
agg=json.load(open('.cache/flip_corrected.json'))
for name in peritem:
    fl=sum(1 for r in peritem[name] if r["status"]=="flipped")
    st=sum(1 for r in peritem[name] if r["status"]=="stable")
    un=sum(1 for r in peritem[name] if r["status"]=="uncounted")
    ok=(fl==agg[name]["flips"] and st==agg[name]["stable"] and un==agg[name]["uncounted"])
    print(f"[reproduce {name}] {fl}/{st}/{un} vs banked {agg[name]['flips']}/{agg[name]['stable']}/{agg[name]['uncounted']} -> {'OK' if ok else 'DRIFT'}")
    assert ok, f"instrument drift on {name} — per-item pass is NOT the banked instrument; no read fires"

json.dump(peritem,open('.cache/flip_peritem.json','w'),indent=0)
print("[saved] .cache/flip_peritem.json")

# ---- READ 1: SLOT EMISSION beside binding, separately ----
print("\n=== SLOT EMISSION (per-view non-None fraction; parseability rider) ===")
for name in ("g22","vlow","vfull"):
    rows=peritem[name]
    eo=np.mean([r["emit_orig"] for r in rows])/5.0
    et_rows=[r for r in rows if "emit_trans" in r]
    et=np.mean([r["emit_trans"] for r in et_rows])/5.0
    print(f"[{name}] orig {eo:.3f}  transformed {et:.3f} (n_trans={len(et_rows)})")

# ---- READ 2: THE MUL PREDICTION (pinned blind at fire design) ----
print("\n=== THE MUL PREDICTION (effect = g22 - arm, per family; floor n>=15 both cells) ===")
def fam_rate(name,fam):
    rows=[r for r in peritem[name] if fam in r["families"] and r["status"]!="uncounted"]
    fl=sum(1 for r in rows if r["status"]=="flipped")
    return (fl/len(rows) if rows else None), len(rows)
for fam in ("mul","add","div","sub","pct","sel"):
    g,ng=fam_rate("g22",fam)
    line=f"  {fam}: g22 {g:.1%} (n={ng})" if g is not None else f"  {fam}: n=0"
    for arm in ("vlow","vfull"):
        a,na=fam_rate(arm,fam)
        if g is not None and a is not None:
            line+=f"  {arm} {a:.1%} (n={na}, effect {g-a:+.1%})"
    print(line)
gm,nm=fam_rate("g22","mul"); gp,np_=fam_rate("g22","pct")
readable = nm>=15 and np_>=15
if not readable:
    print(f"VERDICT (pinned): UNREADABLE-AT-SUPPORT — mul n={nm}, pct n={np_} (floor 15). "
          "The blind prediction CLOSES as unreadable at this fixture's support; not void, read.")
else:
    effs={}
    for fam in ("mul","pct"):
        g,_=fam_rate("g22",fam)
        arms=[fam_rate(a,fam)[0] for a in ("vlow","vfull")]
        effs[fam]=g-np.mean(arms)
    print(f"mul effect {effs['mul']:+.1%} vs pct effect {effs['pct']:+.1%}")
    if effs["mul"]<effs["pct"]:
        print("VERDICT (pinned): mul effect SMALLER — coverage-as-mechanism consistent.")
    else:
        print("VERDICT (pinned): mul effect >= pct — coverage-as-mechanism REFUTED at this support.")
print("\nScope line (standing): family tags are census-grain word substrings — punctuation-blind by construction.")
