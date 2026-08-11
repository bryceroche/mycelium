"""contour_rent.py — gut #191's rent: within-row causal commitment.
The deepest rel slot's readout quality: full attention field vs
fields with other factors' sentences MASKED (commitment simulated
at inference; trunk fixed)."""
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
def pool(i, tokm):
    h=np.asarray(trunk[i]).astype(np.float32) @ P["waist_w"] + P["waist_b"]
    w=gelu(h)+P["sent_emb"][sent[i]]
    V=w @ P["attn_wv"] + P["attn_wv_b"]; K=w @ P["attn_wk"] + P["attn_wk_b"]
    q=(P["fq"] @ P["attn_wq"] + P["attn_wq_b"])
    qh=q.reshape(24,NH,hd); kh=K.reshape(-1,NH,hd); vh=V.reshape(-1,NH,hd)
    pooled=np.zeros((24,HW),np.float32); at=np.zeros((24,len(tokm)),np.float32)
    for hz in range(NH):
        sc=(qh[:,hz] @ kh[:,hz].T)/np.sqrt(hd); sc=sc+(1.0-tokm)[None,:]*-1e4
        e=np.exp(sc-sc.max(-1,keepdims=True)); a_=e/e.sum(-1,keepdims=True)
        pooled[:,hz*hd:(hz+1)*hd]=a_ @ vh[:,hz]; at+=a_/NH
    st=pooled @ P["attn_wo"] + P["attn_wo_b"] + P["fq"]
    st=st+gelu(st @ P["ffn_w1"] + P["ffn_b1"]) @ P["ffn_w2"] + P["ffn_b2"]
    return st, at
ROWS=[i for i in range(len(samples)) if ORD.search(samples[i]["text"])][:300]
# build probe on HEALTHY full-field slots (the sound instrument from tongues)
HE=[i for i in range(len(samples)) if not ORD.search(samples[i]["text"])][:300]
X=[];Y=[]
for i in HE:
    st,at=pool(i, mask[i].astype(np.float32))
    for f in samples[i]["factors"]:
        if f.get("ftype")!="rel" or len(set(f.get("args",[])))!=2: continue
        fm=spans_tokmask(f.get("spans") or [], offsets[i], at.shape[1])
        if fm.sum()==0: continue
        j=int(np.argmax(at @ fm)); X.append(st[j]); Y.append(sorted(f["args"])[0])
X=np.array(X);Y=np.array(Y,np.int32)
mu,sg=X.mean(0),X.std(0)+1e-6
W=np.linalg.lstsq(((X-mu)/sg).T@((X-mu)/sg)+10.0*np.eye(HW), ((X-mu)/sg).T@np.eye(24)[Y], rcond=None)[0]
rd=lambda v: int((((v-mu)/sg)@W).argmax())
res={"full":[], "half":[], "max":[]}
for i in ROWS:
    fs=[f for f in samples[i]["factors"] if f.get("spans")]
    rels=[f for f in fs if f.get("ftype")=="rel" and len(set(f.get("args",[])))==2]
    if len(rels)<2: continue
    tgt=rels[-1]                              # the deepest rel factor
    T=mask.shape[1]
    tm_full=mask[i].astype(np.float32)
    others=[f for f in fs if f is not tgt]
    om=[spans_tokmask(f["spans"],offsets[i],T) for f in others]
    tm_max=tm_full.copy()
    for m_ in om: tm_max=tm_max*(1.0-m_)
    tm_half=tm_full.copy()
    for m_ in om[:len(om)//2]: tm_half=tm_half*(1.0-m_)
    a0=sorted(tgt["args"])[0]
    for name,tm in [("full",tm_full),("half",tm_half),("max",tm_max)]:
        st,at=pool(i,tm)
        fm=spans_tokmask(tgt["spans"],offsets[i],T)
        j=int(np.argmax(at @ fm))
        res[name].append(rd(st[j])==a0)
for k2 in ("full","half","max"):
    a=np.mean(res[k2]); print(f"[contour] deepest-slot readout, {k2:4s} field: {a:.3f} (n={len(res[k2])})",flush=True)
