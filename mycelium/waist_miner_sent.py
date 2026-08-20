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
z=np.load(SRC); tk=z["tokmask"]; sent=z["sent"]; gft=z["g_ftype"]; gfs=z["g_fspan"]
st=np.load(NPY, mmap_mode='r')
P=np.random.RandomState(41).randn(2048,512)/np.sqrt(2048)
KINDS=["rel","given","mod","sel","pct","fdiv","macro","frac","chain"]
db=sqlite3.connect('.cache/campaign.db')
db.execute("""CREATE TABLE IF NOT EXISTS waist_patterns_sent(
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
            M=np.stack(means); c=M@v; j=int(c.argmax())
            if c[j]<0.92: j=-1
        if j<0 and len(means)<4096:
            means.append(v.copy()); cnt.append(0); m2.append(np.zeros(512,np.float32)); kc.append({})
            j=len(means)-1
        if j>=0:
            cnt[j]+=1
            d=v-means[j]; means[j]=means[j]+d/cnt[j]; m2[j]=m2[j]+d*(v-means[j])
            kc[j][kind]=kc[j].get(kind,0)+1
        nsent+=1
db.execute("DELETE FROM waist_patterns_sent")
for j in range(len(means)):
    db.execute("INSERT INTO waist_patterns_sent VALUES(?,?,?,?,?,?)",
        (j,cnt[j],means[j].astype(np.float32).tobytes(),m2[j].tobytes(),
         json.dumps(kc[j]),"form8-sentgrain"))
db.commit()
order=np.argsort(cnt)[::-1][:8]
print(f"[v2-miner] {nsent} sentences from {CAP} rows -> {len(means)} clusters")
for j in order:
    tot=sum(kc[j].values()); dom=max(kc[j],key=kc[j].get) if kc[j] else "?"
    pur=kc[j].get(dom,0)/max(tot,1)
    print(f"  c{j}: n={cnt[j]}  dominant={dom} purity={pur:.2f}")
print("== V2 MINER COMPLETE ==")
