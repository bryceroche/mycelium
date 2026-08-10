"""formation_mint.py — door #37's sliver: crowded mul-dup rows over
licensed surfaces (fixture phrasing fenced). Full corpus schema."""
import json, re, sys
import numpy as np
sys.path.insert(0,'.')
from mycelium.era import MINT_ROW_REQUIRED
T=json.load(open('.cache/aug_table_v3.json'))["licensed"]
MUL=[e["fmt"] for e in T if e["construction"]=="dup" and any(w in e["fmt"] for w in ("times","*","square","multiplied","x ","roduct"))]
assert "lots of" not in " ".join(MUL), "the fixture phrasing is FENCED"
L="abcdefghijklmnopqrstuvwx"
GIV=["{v} is {n}.","It is known that {v} is {n}.","{v} has the value {n}."]
rng=np.random.RandomState(41000)
rows=[]; seen=set()
while len(rows)<700:
    nd=int(rng.randint(0,7))                      # door #41: 0-6
    x=int(rng.randint(2,13)); prod=x*x
    gv=[int(rng.randint(2,90)) for _ in range(nd)]
    nv=nd+2; dv=nd; res=nd+1
    sents=[(i, GIV[rng.randint(3)].format(v=L[i],n=gv[i])) for i in range(nd)]
    sents.append((dv, GIV[rng.randint(3)].format(v=L[dv],n=x)))
    fmt=MUL[rng.randint(len(MUL))]
    dsent=fmt.format(a=L[dv],c=L[res])
    pos=rng.randint(3)                             # dup position: 0 first,1 middle,2 last
    ins=0 if pos==0 else (len(sents)//2 if pos==1 else len(sents))
    order=sents[:ins]+[("DUP",dsent)]+sents[ins:]
    pre=f"Consider the numbers {', '.join(L[:nv])}. "
    text=pre
    fspans=[]
    for tag,s in order:
        a=len(text); text+=s+" "; fspans.append((tag,(a,a+len(s))))
    text=text.strip()+f" What is {L[res]}?"
    if text in seen: continue
    seen.add(text)
    sol=[0]*nv
    for i in range(nd): sol[i]=gv[i]
    sol[dv]=x; sol[res]=prod
    factors=[]
    for tag,(a,b) in fspans:
        if tag=="DUP":
            factors.append({"ftype":"rel","op":"mul","args":[dv,dv],"result":res,"spans":[[a,b]]})
        else:
            factors.append({"ftype":"given","var":tag,"value":sol[tag],"spans":[[a,b]]})
    mentions={}
    for i in range(nv):
        mentions[str(i)]=[[m.start(),m.end()] for m in re.finditer(rf"\b{L[i]}\b",text)]
        assert mentions[str(i)], (i,text)
    row={"text":text,"factors":factors,"query_var":res,"n_vars":nv,"m":300,
         "mentions":mentions,"solution":sol,"decisions":0,"gen":"form37"}
    for k in MINT_ROW_REQUIRED: assert k in row, k
    rows.append(row)
with open('.cache/formation_sliver_v3add.jsonl','w') as f:
    for r in rows: f.write(json.dumps(r)+"\n")
base=[l for l in open('.cache/form_mix2.jsonl')]
seen_base=set()
with open('.cache/form_mix3.jsonl','w') as f:
    for l in base: f.write(l)
    for r in rows: f.write(json.dumps(r)+"\n")
n0=len(base)
form_all=list(range(91400,n0))+list(range(n0,n0+len(rows)))
size_idx=json.load(open('.cache/size_sliver_idx.json'))
json.dump(sorted(size_idx+form_all),open('.cache/ration41_idx.json','w'))
dup_idx=json.load(open('.cache/dup_rehearsal_idx.json'))
json.dump(sorted(dup_idx),open('.cache/dup_only_idx.json','w'))
print(f"[mint] +{len(rows)} rows; mix {n0}->{n0+len(rows)}; 8x tier {len(size_idx)+len(form_all)} (size 3000 + form {len(form_all)}); 3x tier {len(dup_idx)}")
