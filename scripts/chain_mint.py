"""chain_mint.py — door #56's sliver: product-cascade PROSE one floor
up (gold = givens + CHAIN_MUL macro; the parse target is the macro;
expansion/grading stay primitive as always)."""
import json, re, sys
import os as _os
import numpy as np
sys.path.insert(0,'.')
from mycelium.era import MINT_ROW_REQUIRED
L="abcdefghijklmnopqrstuvwx"
FORMS=[
 "The product of {xs} is {r}.",
 "Multiplying {xs} together gives {r}.",
 "{r} is the product of {xs}.",
]
PAIR=[
 "The product of {a} and {b} is {c}.",
 "Multiplying {a} by {b} gives {c}.",
 "{c} is {a} multiplied by {b}.",
 "{a} times {b} makes {c}.",
]
GIV=["{v} is {n}.","It is known that {v} is {n}.","{v} has the value {n}."]
def xs_phrase(vs):
    if len(vs)==2: return f"{vs[0]} and {vs[1]}"
    return ", ".join(vs[:-1])+f" and {vs[-1]}"
rng=np.random.RandomState(int(_os.environ.get("MINT_SEED","56000")))
rows=[]; seen=set()
while len(rows)<2000:
    if _os.environ.get("INBAND")=="1":
        k=int(rng.randint(3,5)); nd=0              # door #58: inside the band
    else:
        k=int(rng.randint(3,6))
        nd=int(rng.randint(0,4))                   # distractor givens
    if _os.environ.get("DIVERSE_VALS")=="1":
        while True:
            smalls=[int(x) for x in rng.choice([2,3,4,5],k-1,replace=False)]
            sp=1
            for v in smalls: sp*=v
            if 300//sp < 6: continue
            big=int(rng.randint(6,min(40,300//sp)+1))
            vals=smalls+[big]; rng.shuffle(vals)
            prod=sp*big
            if prod<=300 and len(set(vals))==k: break
    else:
        while True:
            vals=[int(rng.randint(2,5)) for _ in range(k)]
            prod=1
            for v in vals: prod*=v
            if prod<=300: break
    gv=[int(rng.randint(2,90)) for _ in range(nd)]
    nv=nd+k+1
    xs=list(range(nd,nd+k)); res=nd+k
    sents=[(i,GIV[rng.randint(3)].format(v=L[i],n=gv[i])) for i in range(nd)]
    sents+=[(nd+i,GIV[rng.randint(3)].format(v=L[nd+i],n=vals[i])) for i in range(k)]
    if _os.environ.get("SEQ_SURFACE")=="1":
        # sequential pairwise prose through named intermediates; the gold
        # macro FUSES the sentences (its span covers the whole run)
        ts=[nv+t for t in range(k-2)]              # intermediate vars (named in text via extended roster)
        nv2=nv+k-2
        pre_vars=nv2
        chain_sents=[]
        acc=xs[0]
        for t,v in enumerate(xs[1:]):
            tgt=res if t==k-2 else ts[t]
            f_=PAIR[rng.randint(len(PAIR))]
            chain_sents.append(f_.format(a=L[acc],b=L[v],c=L[tgt]))
            acc=tgt
        msent=" ".join(chain_sents)
        nv=nv2
        pos=rng.randint(3); ins=0 if pos==0 else (len(sents)//2 if pos==1 else len(sents))
        order=sents[:ins]+[("M",msent)]+sents[ins:]
    else:
        mform=FORMS[rng.randint(len(FORMS))]
        msent=mform.format(xs=xs_phrase([L[v] for v in xs]), r=L[res])
        pos=rng.randint(3); ins=0 if pos==0 else (len(sents)//2 if pos==1 else len(sents))
        order=sents[:ins]+[("M",msent)]+sents[ins:]
    pre=f"Consider the numbers {', '.join(L[:nv])}. "
    text=pre; fsp=[]
    for tag,ss in order:
        a=len(text); text+=ss+" "; fsp.append((tag,(a,a+len(ss))))
    text=text.strip()+f" What is {L[res]}?"
    if text in seen: continue
    seen.add(text)
    sol=[0]*nv
    for i in range(nd): sol[i]=gv[i]
    for i in range(k): sol[nd+i]=vals[i]
    sol[res]=prod
    if _os.environ.get("SEQ_SURFACE")=="1":
        accv=vals[0]
        for t in range(k-2):
            accv*=vals[t+1]; sol[nd+k+1+t]=accv
    factors=[]
    for tag,(a,b) in fsp:
        if tag=="M":
            factors.append({"ftype":"macro","name":"CHAIN_MUL","xs":xs,"result":res,"spans":[[a,b]]})
        else:
            factors.append({"ftype":"given","var":tag,"value":sol[tag],"spans":[[a,b]]})
    mentions={}
    ok=True
    for i in range(nv):
        mentions[str(i)]=[[m.start(),m.end()] for m in re.finditer(rf"\b{L[i]}\b",text)]
        if not mentions[str(i)]: ok=False
    if not ok: continue
    row={"text":text,"factors":factors,"query_var":res,"n_vars":nv,"m":300,
         "mentions":mentions,"solution":sol,"decisions":0,"gen":"chain56"}
    for kk in MINT_ROW_REQUIRED: assert kk in row, kk
    rows.append(row)
with open(_os.environ.get('SLIVER_OUT','.cache/chain_sliver.jsonl'),'w') as f:
    for r in rows: f.write(json.dumps(r)+"\n")
base=[l for l in open(_os.environ.get('BASE_MIX','.cache/form_mix3.jsonl'))]
with open(_os.environ.get('OUT_MIX','.cache/form_mix4.jsonl'),'w') as f:
    for l in base: f.write(l)
    for r in rows: f.write(json.dumps(r)+"\n")
n0=len(base)
json.dump(list(range(n0,n0+len(rows))), open(_os.environ.get('IDX_OUT','.cache/chain_sliver_idx.json'),'w'))
dup=json.load(open('.cache/dup_only_idx.json'))
json.dump(sorted(set(dup)|set(range(n0,n0+len(rows)))), open(_os.environ.get('RATION_OUT','.cache/ration56_t3_idx.json'),'w'))
print(f"[mint] {len(rows)} cascade rows; mix {n0}->{n0+len(rows)}; 3x tier = dup + chain sliver")
