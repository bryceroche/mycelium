"""waist_miner_sent.py — THE V2 MINER (segment-first): one silhouette PER
SENTENCE (sent indices = the segments), leader-clustered (cos 0.92) with
Welford into waist_patterns_sent — SEPARATE table (grains never cross-match;
sharp partitions at the schema). Each cluster tallies the gold's factor-KIND
at its sentences (fspan-mass argmax): segment-then-classify with mechanical
labels. Harvest-only fence. Matching accelerates, never decides."""
import os, sys, json, sqlite3
sys.path.insert(0,'.')
import numpy as np
SRC=os.environ.get("MINER_SRC",".cache/phase1_alg_states_form8.npz")
NPY=os.environ.get("MINER_NPY",".cache/phase1_alg_states_form8_states.npy")
assert "test" not in SRC and "big" not in SRC
CAP=int(os.environ.get("MINER_CAP","500"))
CLU_CAP=int(os.environ.get("MINER_CLUCAP","4096"))
DROPPED=[0]
z=np.load(SRC); tk=z["tokmask"]; sent=z["sent"]; gft=z["g_ftype"]; gfs=z["g_fspan"]
st=np.load(NPY, mmap_mode='r')
P=np.random.RandomState(41).randn(2048,512)/np.sqrt(2048)
KINDS=["rel","given","mod","sel","pct","fdiv","macro","frac","chain"]
TAB=os.environ.get('MINER_TABLE','waist_patterns_sent')
db=sqlite3.connect('.cache/campaign.db')
db.execute(f"""CREATE TABLE IF NOT EXISTS {TAB}(
  cluster_id INTEGER PRIMARY KEY, count INTEGER, mean BLOB, m2 BLOB,
  kind_counts TEXT, register TEXT)""")
means=[]; cnt=[]; m2=[]; kc=[]
rng=np.random.RandomState(7)
rows=rng.choice(min(94100,st.shape[0]), CAP, replace=False)
nsent=0
for ri in rows:
    a=np.asarray(st[ri]).astype(np.float32); m=tk[ri].astype(np.float32); sn=sent[ri]
    smax=int(sn[m>0].max()) if (m>0).any() else 0
    for s in range(0,smax+1):
        sel=(sn==s)&(m>0)
        if sel.sum()<2: continue
        v=a[sel].mean(0)@P; v=v/max(np.linalg.norm(v),1e-9)
        # the sentence's kind: factor with max fspan mass here
        mass=gfs[ri][:,sel].sum(1)
        kind=KINDS[int(gft[ri][int(mass.argmax())])] if mass.max()>0 else "none"
        j=-1
        if means:
            M=np.stack(means)
            Mn=M/np.maximum(np.linalg.norm(M,axis=1,keepdims=True),1e-9)
            c=Mn@v; j=int(c.argmax())
            # VARIANCE-AWARE (2026-08-20): rigid clusters demand closeness —
            # threshold tightens toward 3*spread, floored, capped at 0.92-base
            if cnt[j]>=8:
                sp=float(np.sqrt(np.maximum(m2[j],0)/max(cnt[j],1)).mean())
                thr=1.0-min(0.08,max(0.02,3.0*sp))
            else:
                thr=0.92
            if c[j]<thr: j=-1
        if j<0 and len(means)<CLU_CAP:
            means.append(v.copy()); cnt.append(0); m2.append(np.zeros(512,np.float32)); kc.append({})
            j=len(means)-1
        if j<0: DROPPED[0]+=1          # no-silent-caps: counted, loud
        if j>=0:
            cnt[j]+=1
            d=v-means[j]; means[j]=means[j]+d/cnt[j]; m2[j]=m2[j]+d*(v-means[j])
            kc[j][kind]=kc[j].get(kind,0)+1
        nsent+=1
db.execute(f"DELETE FROM {TAB}")
for j in range(len(means)):
    db.execute(f"INSERT INTO {TAB} VALUES(?,?,?,?,?,?)",
        (j,cnt[j],means[j].astype(np.float32).tobytes(),m2[j].tobytes(),
         json.dumps(kc[j]),"form8-sentgrain"))
db.commit()
order=np.argsort(cnt)[::-1][:8]
print(f"[v2-miner] {nsent} sentences from {CAP} rows -> {len(means)} clusters (cap {CLU_CAP}; DROPPED {DROPPED[0]})")
for j in order:
    tot=sum(kc[j].values()); dom=max(kc[j],key=kc[j].get) if kc[j] else "?"
    pur=kc[j].get(dom,0)/max(tot,1)
    print(f"  c{j}: n={cnt[j]}  dominant={dom} purity={pur:.2f}")
print("== V2 MINER COMPLETE ==")
