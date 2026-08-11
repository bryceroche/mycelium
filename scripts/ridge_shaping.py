"""ridge_shaping.py — the shaping read (CPU, banked states): was the
token-grain referent ALREADY on g41's waist pre-REF? Ridge from waist
tokens at bigtest indirect sites -> var id. Present = pure transport
work (REF retires); absent = REF stays as resolver."""
import os, sys, json
for k,v in [("ALG2","1"),("ALG_FTYPES","8"),("ALG_HW","512"),("ALG_DUP","1"),("ALG_WIDE","1")]: os.environ.setdefault(k,v)
os.environ.setdefault("ALG_TEST",".cache/algebra_nl_bigtest.jsonl"); os.environ.setdefault("ALG_TEST_NAME","bigtest")
os.environ["ALG_REF"]="1"   # gold derivation only (refvar); params loaded per-ckpt below
sys.path.insert(0,'.'); sys.path.insert(0,'scripts')
import numpy as np
import phase1_algebra_head as PH
from tinygrad.nn.state import safe_load
samples, ids2, mask, offsets = PH.tokenize(os.environ["ALG_TEST"])
g=PH.build_gold(samples, offsets)
rv=g["refvar"]; sent=np.stack([PH.sent_indices(s["text"],o,mask[i]) for i,(s,o) in enumerate(zip(samples,offsets))])
st=np.load('.cache/phase1_alg_states_bigtest.npz')
trunk=np.load('.cache/phase1_alg_states_bigtest_states.npy', mmap_mode='r') if os.path.exists('.cache/phase1_alg_states_bigtest_states.npy') else st["states"]
def waist_of(ck):
    os.environ["ALG_REF"]="0"
    sd=safe_load(ck)
    W=sd["waist_w"].numpy().astype(np.float32); b=sd["waist_b"].numpy().astype(np.float32)
    SE=sd["sent_emb"].numpy().astype(np.float32)
    X=[]; Y=[]
    for i in range(len(samples)):
        ts=np.where(rv[i]>=0)[0]
        if not len(ts): continue
        h=np.asarray(trunk[i]).astype(np.float32) @ W + b
        h=h*(h*0.7978845608*(1+0.044715*h*h)).tanh()*0+np.where(h>0,h,0)*0+h  # placeholder
        X.append(h[ts]); Y.append(rv[i][ts])
    return np.vstack(X), np.concatenate(Y), W, b, SE
# NOTE: exact gelu+sent_emb: recompute properly
def waist_exact(ck):
    sd=safe_load(ck)
    W=sd["waist_w"].numpy().astype(np.float32); b=sd["waist_b"].numpy().astype(np.float32)
    SE=sd["sent_emb"].numpy().astype(np.float32)
    from scipy.special import erf
    X=[]; Y=[]
    for i in range(len(samples)):
        ts=np.where(rv[i]>=0)[0]
        if not len(ts): continue
        h=np.asarray(trunk[i]).astype(np.float32) @ W + b
        h=0.5*h*(1.0+erf(h/np.sqrt(2.0)))          # gelu
        h=h+SE[sent[i]]
        X.append(h[ts]); Y.append(rv[i][ts].astype(np.int32))
    return np.vstack(X), np.concatenate(Y)
for name,ck in [("g41 (pre-REF)",".cache/g41_onemass_refold.safetensors"),
                ("g23v5 (the gate lineage)",".cache/g23v5.safetensors")]:
    X,Y=waist_exact(ck)
    n=len(X); tr=np.arange(n)%2==0; te=~tr
    mu,sg=X[tr].mean(0),X[tr].std(0)+1e-6
    Xs=(X-mu)/sg
    Wr=np.linalg.lstsq(Xs[tr].T@Xs[tr]+10.0*np.eye(X.shape[1]), Xs[tr].T@np.eye(24)[Y[tr]], rcond=None)[0]
    acc=(Xs[te]@Wr).argmax(1)==Y[te]
    print(f"[shaping] {name}: ridge holdout {acc.mean():.3f} (n={te.sum()})",flush=True)
print("[shaping] h_ref on g43 read 0.882 — compare above",flush=True)
