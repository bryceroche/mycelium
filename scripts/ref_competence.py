"""ref_competence.py — door #43's discriminating read: h_ref accuracy
on (a) mix indirect sites (did it learn?), (b) bigtest indirect sites
(did it transfer?). Decides refuted-vs-underweighted-vs-register."""
import os, sys, json
os.environ["ALG_REF"]="1"
for k,v in [("ALG2","1"),("ALG_FTYPES","8"),("ALG_HW","512"),("ALG_DUP","1"),("ALG_WIDE","1"),("ALG_INV","1")]: os.environ.setdefault(k,v)
os.environ.setdefault("ALG_TEST",".cache/algebra_nl_bigtest.jsonl"); os.environ.setdefault("ALG_TEST_NAME","bigtest")
os.environ.setdefault("ALG_TRAIN",".cache/form_mix3.jsonl"); os.environ.setdefault("ALG_TRAIN_NAME","form3")
os.environ.setdefault("ALG_ALLOW_PEN_TRAIN","1")
sys.path.insert(0,'.'); sys.path.insert(0,'scripts')
import numpy as np
from phase1_algebra_head import T_ALG, build_params, forward, load_alg, build_gold
import phase1_algebra_head as PH
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load
p=build_params(0); sd=safe_load(".cache/g43_efloor.safetensors")
for k in p: p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
def acc_on(samples, states, tokmask, gold, sent, name, cap=400):
    ok=n=0
    rng=np.random.RandomState(7)
    order=rng.permutation(len(samples))[:cap]
    for s0 in range(0,len(order),8):
        sl=order[s0:s0+8]
        pad=8-len(sl); slp=np.concatenate([sl,sl[:1].repeat(pad)]) if pad else sl
        o=forward(p,Tensor(states[slp].astype(np.float32),dtype=dtypes.float),
                  Tensor(tokmask[slp].astype(np.float32),dtype=dtypes.float),
                  Tensor(sent[slp].astype(np.int32),dtype=dtypes.int))
        ref=o["ref"].realize().numpy()
        for bi,ri in enumerate(sl):
            rv=gold["refvar"][ri]
            for t in np.where(rv>=0)[0]:
                ok+= int(ref[bi,t].argmax())==int(rv[t]); n+=1
    print(f"[ref-competence] {name}: {ok}/{n} = {ok/max(n,1):.3f}",flush=True)
    return ok/max(n,1)
tr=load_alg("train")
acc_on(*tr, "MIX indirect sites (train-fit)")
# bigtest: derive refvar gold live (same mentions logic)
samples, states, tokmask, gold, sent = load_alg("test")
if "refvar" not in gold:
    samples2, ids2, mask2, offs2 = PH.tokenize(os.environ["ALG_TEST"])
    g2=build_gold(samples2, offs2)
    gold=dict(gold); gold["refvar"]=g2["refvar"]
acc_on(samples, states, tokmask, gold, sent, "BIGTEST indirect sites (transfer)")
