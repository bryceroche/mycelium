"""waist_miner_v3.py — THE MLIR MINER (the word given): slot states per
breath cycle = the dialect ladder's raw material. Compound key (cluster,
cycle); slots are the segments (direct gold kinds); Welford per centroid;
TRANSITION EDGES (basin@k -> basin@k+1). Mask provenance: deployment
two-pass masks (recorded). Fence: recognition proposes; the key certifies."""
import os, sys, json, sqlite3
sys.path.insert(0,'.'); sys.path.insert(0,'scripts')
import numpy as np
os.environ.setdefault("ALG_MINE_BREATHS","1"); os.environ.setdefault("ALG_BREATH","7")
from phase1_algebra_head import build_params, forward, load_alg, build_slot_masks, L_FAC
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load
samples, states, tokmask, gold, sent = load_alg("train")
KINDS=["rel","given","mod","sel","pct","fdiv","macro","frac","chain"]
p=build_params(0); sd=safe_load(os.environ["ALG_CKPT"])
for k in p: p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
CAP=int(os.environ.get("MINER_CAP","200"))
K_B=int(os.environ.get("ALG_BREATH","3"))
rng=np.random.RandomState(7)
rows=rng.choice(states.shape[0],min(CAP,states.shape[0]),replace=False)
tabs={}; DROPPED=[0]
def leader(cyc,v):
    t=tabs.setdefault(cyc,{"means":[],"cnt":[],"m2":[],"kc":[]})
    j=-1
    if t["means"]:
        M=np.stack(t["means"]); Mn=M/np.maximum(np.linalg.norm(M,axis=1,keepdims=True),1e-9)
        c=Mn@v; j=int(c.argmax())
        if c[j]<0.92: j=-1
    if j<0:
        if len(t["means"])>=8192: DROPPED[0]+=1; return -1
        t["means"].append(v.copy()); t["cnt"].append(0); t["m2"].append(np.zeros(v.shape[0],np.float32)); t["kc"].append({})
        j=len(t["means"])-1
    return j
edges={}
for s0 in range(0,CAP,8):
    sl=[int(r) for r in rows[s0:s0+8]]; pad=8-len(sl); slp=sl+sl[:1]*pad
    ts=Tensor(states[slp].astype(np.float32),dtype=dtypes.float)
    tk=Tensor(tokmask[slp].astype(np.float32),dtype=dtypes.float)
    se=Tensor(sent[slp].astype(np.int32),dtype=dtypes.int)
    o0=forward(p,ts,tk,se)
    o0n={k:o0[k].realize().numpy() for k in ("fat","args","res")}
    mk=build_slot_masks(o0n,sent[slp])
    o=forward(p,ts,tk,se,slot_mask=Tensor(mk,dtype=dtypes.float))
    if "breaths_all" not in o: raise SystemExit("hook not engaged")
    B=[b.realize().numpy() for b in o["breaths_all"]]
    for bi,ri in enumerate(sl):
        for j in range(L_FAC):
            if gold["presence"][ri,j]<=0: continue
            kind=KINDS[int(gold["ftype"][ri,j])]
            prev=-1
            for cyc,bstate in enumerate(B):
                v=bstate[bi,j].astype(np.float32); v=v/max(np.linalg.norm(v),1e-9)
                cid=leader(cyc,v)
                if cid>=0:
                    t=tabs[cyc]; t["cnt"][cid]+=1
                    d=v-t["means"][cid]; t["means"][cid]=t["means"][cid]+d/t["cnt"][cid]
                    t["m2"][cid]=t["m2"][cid]+d*(v-t["means"][cid])
                    t["kc"][cid][kind]=t["kc"][cid].get(kind,0)+1
                    if prev>=0: edges[(cyc-1,prev,cid)]=edges.get((cyc-1,prev,cid),0)+1
                prev=cid
db=sqlite3.connect(os.path.join(os.path.dirname(__file__),'..','.cache','campaign.db'))
db.execute("""CREATE TABLE IF NOT EXISTS waist_patterns_v3(
  cluster_id INTEGER, breath_cycle INTEGER, count INTEGER, mean BLOB, m2 BLOB,
  kind_counts TEXT, mask_provenance TEXT, PRIMARY KEY(cluster_id,breath_cycle))""")
db.execute("""CREATE TABLE IF NOT EXISTS v3_transitions(
  cycle INTEGER, from_id INTEGER, to_id INTEGER, count INTEGER,
  PRIMARY KEY(cycle,from_id,to_id))""")
db.execute("DELETE FROM waist_patterns_v3"); db.execute("DELETE FROM v3_transitions")
for cyc,t in tabs.items():
    for j in range(len(t["means"])):
        db.execute("INSERT INTO waist_patterns_v3 VALUES(?,?,?,?,?,?,?)",
            (j,cyc,t["cnt"][j],t["means"][j].astype(np.float32).tobytes(),
             t["m2"][j].tobytes(),json.dumps(t["kc"][j]),"deploy-twopass-g41era"))
for (cyc,a,b),n in edges.items():
    db.execute("INSERT INTO v3_transitions VALUES(?,?,?,?)",(cyc,a,b,n))
db.commit(); db.close()
for cyc in sorted(tabs):
    t=tabs[cyc]
    top=int(np.argmax(t["cnt"]))
    kc=t["kc"][top]; dom=max(kc,key=kc.get) if kc else "?"
    print(f"[v3] cycle {cyc}: {len(t['means'])} clusters; top n={t['cnt'][top]} dom={dom} pur={kc.get(dom,0)/max(sum(kc.values()),1):.2f}")
strong=sorted(edges.items(),key=lambda x:-x[1])[:3]
print(f"[v3] transitions recorded {len(edges)}; strongest: {strong}")
print(f"[v3] DROPPED {DROPPED[0]}")
print("== V3 MINER COMPLETE ==")
