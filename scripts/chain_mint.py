"""chain_mint.py — door #56's sliver: product-cascade PROSE one floor
up (gold = givens + CHAIN_MUL macro; the parse target is the macro;
expansion/grading stay primitive as always)."""
import json, re, sys
import numpy as np
sys.path.insert(0,'.')
from mycelium.era import MINT_ROW_REQUIRED
L="abcdefghijklmnopqrstuvwx"
FORMS=[
 "The product of {xs} is {r}.",
 "Multiplying {xs} together gives {r}.",
 "{r} is the product of {xs}.",
]
GIV=["{v} is {n}.","It is known that {v} is {n}.","{v} has the value {n}."]
def xs_phrase(vs):
    if len(vs)==2: return f"{vs[0]} and {vs[1]}"
    return ", ".join(vs[:-1])+f" and {vs[-1]}"
rng=np.random.RandomState(56000)
rows=[]; seen=set()
while len(rows)<2000:
    k=int(rng.randint(3,6))
    nd=int(rng.randint(0,4))                       # distractor givens
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
with open('.cache/chain_sliver.jsonl','w') as f:
    for r in rows: f.write(json.dumps(r)+"\n")
base=[l for l in open('.cache/form_mix3.jsonl')]
with open('.cache/form_mix4.jsonl','w') as f:
    for l in base: f.write(l)
    for r in rows: f.write(json.dumps(r)+"\n")
n0=len(base)
json.dump(list(range(n0,n0+len(rows))), open('.cache/chain_sliver_idx.json','w'))
dup=json.load(open('.cache/dup_only_idx.json'))
json.dump(sorted(set(dup)|set(range(n0,n0+len(rows)))), open('.cache/ration56_t3_idx.json','w'))
print(f"[mint] {len(rows)} cascade rows; mix {n0}->{n0+len(rows)}; 3x tier = dup + chain sliver")
