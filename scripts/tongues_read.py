"""tongues_read.py — gut investigation: 'the slots are speaking in
tongues on purpose'. (A) SET-BASED re-audit of slot stations: pair
predicted rel slots to gold factors by span-attention matching
(order-free), re-ridge fst -> first-arg. (B) INDEX-CONTROLLED
transfer: is the dialect per-slot-index (fq's own codes)?"""
import os, sys, json
for k,v in [("ALG2","1"),("ALG_FTYPES","8"),("ALG_HW","512"),("ALG_DUP","1"),("ALG_WIDE","1")]: os.environ.setdefault(k,v)
os.environ.setdefault("ALG_TEST",".cache/algebra_nl_bigtest.jsonl"); os.environ.setdefault("ALG_TEST_NAME","bigtest")
sys.path.insert(0,'.'); sys.path.insert(0,'scripts')
import numpy as np, re
from scipy.special import erf
import phase1_algebra_head as PH
from tinygrad.nn.state import safe_load
CK=os.environ.get("CK",".cache/g41_onemass_refold.safetensors")
sd=safe_load(CK); P={k:v.numpy().astype(np.float32) for k,v in sd.items()}
samples, ids2, mask, offsets = PH.tokenize(os.environ["ALG_TEST"])
sent=np.stack([PH.sent_indices(s["text"],o,mask[i]) for i,(s,o) in enumerate(zip(samples,offsets))])
trunk=np.load('.cache/phase1_alg_states_bigtest.npz')['states']
NH=8; HW=512; hd=HW//NH
ORD=re.compile(r"(first|second|third|fourth|fifth) number")
def gelu(h): return 0.5*h*(1.0+erf(h/np.sqrt(2.0)))
def spans_tokmask(spans, offs, T):
    m=np.zeros(T,np.float32)
    for a,b in spans:
        for t,(x,y) in enumerate(offs):
            if t>=T: break
            if x<b and y>a and y>x: m[t]=1.0
    return m
def run(i):
    h=np.asarray(trunk[i]).astype(np.float32) @ P["waist_w"] + P["waist_b"]
    w=gelu(h)+P["sent_emb"][sent[i]]
    V=w @ P["attn_wv"] + P["attn_wv_b"]; K=w @ P["attn_wk"] + P["attn_wk_b"]
    q=(P["fq"] @ P["attn_wq"] + P["attn_wq_b"])
    m=mask[i].astype(np.float32)
    qh=q.reshape(24,NH,hd); kh=K.reshape(-1,NH,hd); vh=V.reshape(-1,NH,hd)
    pooled=np.zeros((24,HW),np.float32); at=np.zeros((24,len(m)),np.float32)
    for hz in range(NH):
        sc=(qh[:,hz] @ kh[:,hz].T)/np.sqrt(hd); sc=sc+(1.0-m)[None,:]*-1e4
        e=np.exp(sc-sc.max(-1,keepdims=True)); a_=e/e.sum(-1,keepdims=True)
        pooled[:,hz*hd:(hz+1)*hd]=a_ @ vh[:,hz]; at+=a_/NH
    st=pooled @ P["attn_wo"] + P["attn_wo_b"] + P["fq"]
    st=st+gelu(st @ P["ffn_w1"] + P["ffn_b1"]) @ P["ffn_w2"] + P["ffn_b2"]
    return st, at
ORD_ROWS=[i for i in range(len(samples)) if ORD.search(samples[i]["text"])]
HEALTHY=[i for i in range(len(samples)) if not ORD.search(samples[i]["text"])][:400]
def collect(rows):
    X=[];Y=[];J=[]
    for i in rows:
        st,at=run(i)
        T=len(mask[i])
        for f in samples[i]["factors"]:
            if f.get("ftype")!="rel" or len(set(f.get("args",[])))!=2: continue
            fm=spans_tokmask(f.get("spans") or [], offsets[i], at.shape[1])
            if fm.sum()==0: continue
            j=int(np.argmax(at @ fm))          # span-attention pairing (order-free)
            X.append(st[j]); Y.append(sorted(f["args"])[0]); J.append(j)
    return np.array(X),np.array(Y,np.int32),np.array(J)
XO,YO,JO=collect(ORD_ROWS[:350]); XH,YH,JH=collect(HEALTHY)
print(f"[pools] ordinal slots {len(XO)}  healthy slots {len(XH)}",flush=True)
def ridge(Xtr,Ytr):
    mu,sg=Xtr.mean(0),Xtr.std(0)+1e-6
    Xs=(Xtr-mu)/sg
    W=np.linalg.lstsq(Xs.T@Xs+10.0*np.eye(Xtr.shape[1]), Xs.T@np.eye(24)[Ytr], rcond=None)[0]
    return lambda X: (((X-mu)/sg)@W).argmax(1)
# (A) set-based re-audit
pr=ridge(XH,YH)
print(f"[A set-based] healthy->healthy(fit) {(pr(XH)==YH).mean():.3f}  healthy->ORDINAL transfer {(pr(XO)==YO).mean():.3f}",flush=True)
h=len(XO)//2; pr2=ridge(XO[:h],YO[:h])
print(f"[A set-based] ordinal->ordinal holdout {(pr2(XO[h:])==YO[h:]).mean():.3f}",flush=True)
# (B) index-controlled transfer
common=[j for j in set(JO)&set(JH) if (JH==j).sum()>=40 and (JO==j).sum()>=20]
accs=[]
for j in sorted(common)[:8]:
    prj=ridge(XH[JH==j],YH[JH==j])
    accs.append(((prj(XO[JO==j])==YO[JO==j]).mean(), j, (JO==j).sum()))
for a,j,n in accs: print(f"[B index j={j}] healthy_j->ordinal_j transfer {a:.3f} (n={n})",flush=True)
if accs: print(f"[B] mean within-index transfer {np.mean([a for a,_,_ in accs]):.3f}  vs pooled cross {(pr(XO)==YO).mean():.3f}",flush=True)
