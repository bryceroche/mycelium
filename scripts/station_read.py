"""station_read.py — gut #189's rent: readouts at every transport
station, CPU on banked states. Where does the referent die?
Stations: (1) waist tokens [0.992 banked] -> (2) V-projected tokens
-> (3) slot-attended pool pre-wo -> (4) fst [0.190-class banked]."""
import os, sys, json
for k,v in [("ALG2","1"),("ALG_FTYPES","8"),("ALG_HW","512"),("ALG_DUP","1"),("ALG_WIDE","1")]: os.environ.setdefault(k,v)
os.environ.setdefault("ALG_TEST",".cache/algebra_nl_bigtest.jsonl"); os.environ.setdefault("ALG_TEST_NAME","bigtest")
os.environ["ALG_REF"]="1"
sys.path.insert(0,'.'); sys.path.insert(0,'scripts')
import numpy as np, re
from scipy.special import erf
import phase1_algebra_head as PH
from tinygrad.nn.state import safe_load
CK=os.environ.get("CK",".cache/g41_onemass_refold.safetensors")
sd=safe_load(CK); P={k:v.numpy().astype(np.float32) for k,v in sd.items()}
samples, ids2, mask, offsets = PH.tokenize(os.environ["ALG_TEST"])
g=PH.build_gold(samples, offsets); rv=g["refvar"]
sent=np.stack([PH.sent_indices(s["text"],o,mask[i]) for i,(s,o) in enumerate(zip(samples,offsets))])
trunk=np.load('.cache/phase1_alg_states_bigtest.npz')['states']
NH=8; HW=512; hd=HW//NH
ORD=re.compile(r"(first|second|third|fourth|fifth) number")
rows=[i for i in range(len(samples)) if ORD.search(samples[i]["text"]) and (rv[i]>=0).any()][:500]
def gelu(h): return 0.5*h*(1.0+erf(h/np.sqrt(2.0)))
def stations(i):
    h=np.asarray(trunk[i]).astype(np.float32) @ P["waist_w"] + P["waist_b"]
    w=gelu(h)+P["sent_emb"][sent[i]]                       # (T,512) station 1
    V=w @ P["attn_wv"] + P["attn_wv_b"]                    # (T,512) station 2
    K=w @ P["attn_wk"] + P["attn_wk_b"]
    q=(P["fq"] @ P["attn_wq"] + P["attn_wq_b"])            # (24,512)
    m=mask[i].astype(np.float32)
    qh=q.reshape(24,NH,hd); kh=K.reshape(-1,NH,hd); vh=V.reshape(-1,NH,hd)
    pooled=np.zeros((24,HW),np.float32); at_all=np.zeros((24,mask.shape[1]),np.float32)
    for hzz in range(NH):
        sc=(qh[:,hzz] @ kh[:,hzz].T)/np.sqrt(hd)
        sc=sc+(1.0-m)[None,:]*-1e4
        e=np.exp(sc-sc.max(-1,keepdims=True)); at=e/e.sum(-1,keepdims=True)
        pooled[:,hzz*hd:(hzz+1)*hd]=at @ vh[:,hzz]
        at_all+=at/NH
    st=pooled @ P["attn_wo"] + P["attn_wo_b"] + P["fq"]
    st=st+gelu(st @ P["ffn_w1"] + P["ffn_b1"]) @ P["ffn_w2"] + P["ffn_b2"]  # station 4 (fst)
    return w, V, pooled, st, at_all
# gold per slot: for each rel slot j with 2 distinct args, first-arg var; use decode-free gold: factors
def slot_gold(i):
    out={}
    for j,f in enumerate(samples[i]["factors"]):
        if f.get("ftype")=="rel" and len(set(f.get("args",[])))==2:
            out[j]=sorted(f["args"])[0]
    return out
X={1:[],2:[],3:[],4:[]}; Y=[]
XT={1:[],2:[],3:[],4:[]}; YT=[]
for n_,i in enumerate(rows):
    w,V,pooled,st,at=stations(i)
    sg=slot_gold(i)
    ts=np.where(rv[i]>=0)[0]
    for t in ts[:6]:
        X[1].append(w[t]); X[2].append(V[t]); Y.append(rv[i][t])
    for j,a0 in list(sg.items())[:4]:
        XT[3].append(pooled[j]); XT[4].append(st[j]); YT.append(a0)
    if n_%100==0: print(f"  [{n_}/{len(rows)}]",flush=True)
def ridge_acc(Xl,Yl,name):
    X_=np.array(Xl); Y_=np.array(Yl,np.int32)
    n=len(X_); tr=np.arange(n)%2==0; te=~tr
    mu,sg_=X_[tr].mean(0),X_[tr].std(0)+1e-6
    Xs=(X_-mu)/sg_
    W=np.linalg.lstsq(Xs[tr].T@Xs[tr]+10.0*np.eye(X_.shape[1]), Xs[tr].T@np.eye(24)[Y_[tr]], rcond=None)[0]
    a=((Xs[te]@W).argmax(1)==Y_[te]).mean()
    print(f"[station] {name}: {a:.3f} (n={te.sum()})",flush=True)
ridge_acc(X[1],Y,"1 waist tokens (indirect sites)")
ridge_acc(X[2],Y,"2 V-projected tokens (same sites)")
ridge_acc(XT[3],YT,"3 slot pool pre-wo (first-arg)")
ridge_acc(XT[4],YT,"4 fst post-wo+ffn (first-arg)")
