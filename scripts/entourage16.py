"""entourage16.py — ENTOURAGE-16 (2026-07-22): the chain's second edit.
Gen-16 params: gate = crown_reader_v4 (ALG_FTYPES=8), native family =
crown_v4 corpus, E16 seeds; NEW STAGE: the collapse-ratio re-read under
v4 (the pocket dashboard's first accrual — the quotient-deepening
question's first datapoint).
Stages: repair corpora -> states -> specialist remine vs gen-15 ->
centroids (gen-15 fst, crownv4 family) -> mouth rebuild (crownv4 family) ->
census -> DISSENT-OVERLAP READ (the owed column) -> manifest refresh.
"""
import json, os, subprocess, sys
sys.path.insert(0, "."); sys.path.insert(0, "scripts")

ENV = {"DEV": "AMD", "ALG2": "1", "ALG_FTYPES": "8", "ALG_DUP": "1"}
PARSER = ".cache/crown_reader_v4.safetensors"
NEW_NACK = ".cache/phase1_gen16_nack.safetensors"


def sh(cmd, extra=None, tail=2):
    env = dict(os.environ); env.update(ENV); env.update(extra or {})
    r = subprocess.run(cmd, shell=True, env=env, capture_output=True, text=True)
    for l in (r.stdout + r.stderr).strip().splitlines()[-tail:]:
        print(f"    {l}", flush=True)
    if r.returncode != 0:
        raise RuntimeError(f"stage failed: {cmd[:90]}")


print("=== E16 1/8: fresh repair corpora ===", flush=True)
sh(".venv/bin/python3 scripts/algebra_nl_gen.py --n 800 --seed 161 --out .cache/g16r_v1.jsonl --teeth 0.8", tail=1)
sh(".venv/bin/python3 scripts/algebra2_nl_gen.py --n 800 --seed 162 --out .cache/g16r_v2.jsonl --teeth 0.8 --token-budget 250", tail=1)
sh(".venv/bin/python3 scripts/algebra3_nl_gen.py --n 800 --seed 163 --out .cache/g16r_v3.jsonl --teeth 0.8 --token-budget 250", tail=1)
sh(".venv/bin/python3 scripts/algebra_verbose_gen.py 600 164 .cache/g16r_vb", tail=1)
sh(".venv/bin/python3 scripts/algebra_dag7_gen.py 800 165 .cache/g16r_dag.jsonl", tail=1)
sh("cat .cache/g16r_v1.jsonl .cache/g16r_v2.jsonl .cache/g16r_v3.jsonl "
   ".cache/g16r_vb_verbose.jsonl .cache/g16r_dag.jsonl > .cache/gen16_repair.jsonl "
   "&& wc -l .cache/gen16_repair.jsonl", tail=1)

print("=== E16 2/8: precompute repair states ===", flush=True)
sh(".venv/bin/python3 scripts/phase1_algebra_head.py --precompute",
   {"ALG_TRAIN": ".cache/gen16_repair.jsonl", "ALG_TRAIN_NAME": "gen16repair",
    "PRECOMPUTE_ONLY": "gen16repair"}, tail=1)

print("=== E16 3/8: specialist remine vs gen-15 ===", flush=True)
e = {"ALG_TRAIN": ".cache/gen16_repair.jsonl", "ALG_TRAIN_NAME": "gen16repair",
     "ALG_CKPT": PARSER, "NACK_CKPT": NEW_NACK, "NACK_SPLIT": "train"}
sh(".venv/bin/python3 scripts/phase1_algebra_nack.py --prep", e, tail=2)
e2 = dict(e); e2.update({"STEPS": "4000", "LR": "1e-4", "BATCH": "8", "SEED": "16"})
sh(".venv/bin/python3 scripts/phase1_algebra_nack.py --train", e2, tail=2)

print("=== E16 4/8: monitor centroids (gen-15 fst space, crownv4 family) ===", flush=True)
S4 = r'''
import sys, os, json; sys.path.insert(0,"."); sys.path.insert(0,"scripts")
os.environ["ALG2"]="1"; os.environ["ALG_FTYPES"]="8"; os.environ["ALG_DUP"]="1"
import numpy as np
from phase1_algebra_head import L_FAC, build_params
from waist_abstention_probe import compute_fst
from tinygrad.nn.state import safe_load
def head_kinds(p, fst_rows):
    g = lambda k: p[k].detach().numpy()
    hp, hpb = g("h_pres"), g("h_pres_b")
    hf, hfb = g("h_ftype"), g("h_ftype_b")
    ho, hob = g("h_op"), g("h_op_b")
    out = []
    for v in fst_rows:
        if v @ hp[:, 0] + hpb[0] <= 0:
            out.append(None); continue
        ft = int(np.argmax(v @ hf + hfb))
        out.append(("given","rel_add","rel_mul","mod","sel","pct","fdiv","macro","frac")[
            1 + int(np.argmax(v @ ho + hob)) if ft == 0 else
            (0 if ft == 1 else (3 if ft == 2 else
             (4 if ft == 3 else (5 if ft == 4 else (6 if ft == 5 else (7 if ft == 6 else 8))))))])
    return out
p = build_params(0); sd = safe_load(".cache/crown_reader_v4.safetensors")
for k in p: p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
z = np.load(".cache/phase1_alg_states_crownv4.npz")
st = np.load(".cache/phase1_alg_states_crownv4_states.npy", mmap_mode="r")
idx = list(range(0, st.shape[0], max(1, st.shape[0]//3000)))[:3000]
fst = compute_fst(p, st, z["tokmask"], z["sent"], idx)
by = {}
for r in range(len(idx)):
    for j, kd in enumerate(head_kinds(p, fst[r])):
        if kd: by.setdefault(kd, []).append(fst[r, j])
lib = {k: (lambda c: c/np.linalg.norm(c))(np.mean(v,0)) for k,v in by.items()}
np.savez(".cache/monitor_centroids_gen16.npz", **lib)
print("centroids:", sorted(lib))
'''
open(".cache/_e16_s4.py", "w").write(S4)
sh(".venv/bin/python3 .cache/_e16_s4.py", tail=1)

print("=== E16 5/8: mouth rebuild (crownv4 family) + length refit ===", flush=True)
S5 = r'''
import sys, os, json; sys.path.insert(0,"."); sys.path.insert(0,"scripts")
os.environ["ALG2"]="1"; os.environ["ALG_FTYPES"]="8"
import numpy as np
def pooled_npz(path, npy=None, cap=None):
    z = np.load(path)
    st = z["states"] if "states" in z.files else np.load(npy, mmap_mode="r")
    tk = z["tokmask"]
    n = st.shape[0] if cap is None else min(st.shape[0], cap)
    out = np.zeros((n, st.shape[2]), np.float32)
    for s0 in range(0, n, 256):
        sl = slice(s0, min(s0+256, n))
        a = np.asarray(st[sl]).astype(np.float32); m = tk[sl].astype(np.float32)
        out[sl] = (a*m[:,:,None]).sum(1)/np.maximum(m.sum(1)[:,None],1)
    return out/np.linalg.norm(out,axis=1,keepdims=True), tk[:n].sum(1)
fam, _ = pooled_npz(".cache/phase1_alg_states_crownv4.npz",
                    ".cache/phase1_alg_states_crownv4_states.npy", cap=12000)
rng = np.random.RandomState(16)
bank = fam[rng.choice(len(fam), 2000, replace=False)]
nat, natL = [], []
for nm in ("vtest", "alg4test", "bigtest"):
    v, L = pooled_npz(f".cache/phase1_alg_states_{nm}.npz")
    nat.append(v); natL.append(L)
native = np.vstack(nat); nL = np.concatenate(natL).astype(np.float64)
d = np.sort(1.0 - native @ bank.T, axis=1)[:, :8].mean(1)
X = np.stack([np.ones_like(nL), 1.0/np.maximum(nL,1)], 1)
coef, *_ = np.linalg.lstsq(X, d, rcond=None)
res = d - X @ coef
thr = float(np.percentile(res, 99))
np.savez(".cache/recognition_mouth_gen16.npz", bank=bank,
         thr_knn=np.float32(thr), coef=coef.astype(np.float32))
print(f"[mouth-16] bank rebuilt (crownv4 family); length-controlled thr {thr:.4f}")
'''
open(".cache/_e16_s5.py", "w").write(S5)
sh(".venv/bin/python3 .cache/_e16_s5.py", tail=1)

print("=== E16 6/8: disjoint census under the fresh mouth ===", flush=True)
sh(".venv/bin/python3 scripts/gen11_census.py",
   {"GATE_CKPT": PARSER, "CENSUS_DISJOINT": "1"}, tail=2)

print("=== E16 7/8: DISSENT-OVERLAP READ (the owed column) ===", flush=True)
S7 = r'''
import json
from collections import Counter
def maj(v):
    vs=[x for x in v if x is not None]
    return Counter(vs).most_common(1)[0] if vs else (None,0)
armb = json.load(open(".cache/lattice_armB.json"))["bigtest"]
c2x = json.load(open(".cache/lattice_cap2x.json"))["bigtest"]
def dissent_set(gate_votes):
    out = set()
    for i in range(1500):
        gt, gc = maj(gate_votes[i]); at,_ = maj(armb[i]); ct,_ = maj(c2x[i])
        if gc == 5 and not (at == gt and ct == gt):
            out.add(i)
    return out
d14 = dissent_set(json.load(open(".cache/lattice_gen15_A.json"))["bigtest"])
d15 = dissent_set(json.load(open(".cache/lattice_gen16_V4.json"))["bigtest"])
ov = d14 & d15
print(f"[dissent-overlap] gen-15: {len(d14)} | gen-16: {len(d15)} | "
      f"OVERLAP {len(ov)} ({len(ov)/max(len(d15),1):.0%} of gen-15's) — "
      f"{'a STABLE dissent family (structural)' if len(ov) >= 0.5*len(d15) else 'dissent ROTATES (population-driven)'}")
json.dump({"d14": sorted(d14), "d15": sorted(d15), "overlap": sorted(ov)},
          open(".cache/dissent_overlap_16.json", "w"))
'''
open(".cache/_e16_s7.py", "w").write(S7)
sh(".venv/bin/python3 .cache/_e16_s7.py", tail=1)

print("=== E16 8/9: THE COLLAPSE RE-READ under v4 (dashboard accrual) ===", flush=True)
sh(".venv/bin/python3 scripts/collapse_probe.py",
   {"COLLAPSE_CKPT": ".cache/crown_reader_v4.safetensors"}, tail=3)

print("=== E16 9/9: manifest member refresh ===", flush=True)
m = json.load(open(".cache/GENERATION.json"))
m["specialist_ckpt"] = NEW_NACK
m["monitor_centroids"] = ".cache/monitor_centroids_gen16.npz"
m["mouth"] = ".cache/recognition_mouth_gen16.npz"
m["waivers"] = {"panel": "cert-v2 members armB + cap2x (panel-eligible bench, "
                "now incl. crown-readers fire_armC1/B)"}
m["notes"] = (m.get("notes", "") +
              " | 2026-07-21 ENTOURAGE-15 PAID (entourage15.py, the committed "
              "chain's first edit): specialist remined vs gen-15, centroids "
              "(8 kinds incl. macro) + mouth rebuilt on the crownv4 family, "
              "dissent-overlap column banked, specialist waiver RETIRED.")
json.dump(m, open(".cache/GENERATION.json", "w"), indent=1)
print("[manifest] entourage-15 complete — members refreshed, waiver retired", flush=True)
print("=== ENTOURAGE-16 DONE ===", flush=True)
