"""entourage41.py — THE ESTATE SETTLEMENT FOR GEN-23 (2026-08-03; adapted
from entourage22 by name-map: parser g23 + ALG_WIDE env + gen23 states pair
(88,400 rows, sentinel-verified at the fire) + dissent overlap g22-vs-g23;
sign-only scope per the manifest, wide_not_claimed carried):
an unattributable gate reads nothing. Adapted from entourage21 with its
fossils fixed: (a) centroids/mouth read the G23-ALIGNED state pair
(phase1_alg_states_form3.npz + _g22_states.npy, 82,400 rows, verified at
the fire — entourage21 mixed a g22 npz with g20 states and survived by
prefix accident); (b) the dissent-overlap read takes the panel lattices
FROM THE MANIFEST, not hardcoded names; (c) manifest refresh RE-HASHES
the swapped members (the atomic-truth law extends to the entourage's
write). NEW STANDING STAGE: the wild frontier fixture, both tiers,
under the new gate AND the new mouth — the outside-air check rides the
entourage, every generation (the #101 yield, now duty).
Stages: repair corpora -> states -> specialist remine vs g22 ->
centroids (g22 fst space) -> mouth rebuild (g22 family) + length refit
-> disjoint census -> dissent-overlap (g22 vs g22) -> collapse re-read
-> manifest refresh + re-hash -> standing rehearsal."""
import json, os, subprocess, sys, hashlib
sys.path.insert(0, "."); sys.path.insert(0, "scripts")

ENV = {"DEV": "AMD", "ALG2": "1", "ALG_FTYPES": "8", "ALG_DUP": "1", "ALG_HW": "512", "ALG_WIDE": "1"}
PARSER = ".cache/g41_onemass_refold.safetensors"
NEW_NACK = ".cache/phase1_gen41_nack.safetensors"


def sh(cmd, extra=None, tail=2):
    env = dict(os.environ); env.update(ENV); env.update(extra or {})
    r = subprocess.run(cmd, shell=True, env=env, capture_output=True, text=True)
    for l in (r.stdout + r.stderr).strip().splitlines()[-tail:]:
        print(f"    {l}", flush=True)
    if r.returncode != 0:
        raise RuntimeError(f"stage failed: {cmd[:90]}")


import os as _os
def _lines(f):
    try: return sum(1 for _ in open(f))
    except Exception: return -1
print("=== E41 1/10: fresh repair corpora ===", flush=True)
if _lines(".cache/gen41_repair.jsonl") == 3800:
    print("    [skip] repair corpora banked", flush=True)
else:
    sh(".venv/bin/python3 scripts/algebra_nl_gen.py --n 800 --seed 411 --out .cache/g41r_v1.jsonl --teeth 0.8", tail=1)
    sh(".venv/bin/python3 scripts/algebra2_nl_gen.py --n 800 --seed 412 --out .cache/g41r_v2.jsonl --teeth 0.8 --token-budget 250", tail=1)
    sh(".venv/bin/python3 scripts/algebra3_nl_gen.py --n 800 --seed 413 --out .cache/g41r_v3.jsonl --teeth 0.8 --token-budget 250", tail=1)
    sh(".venv/bin/python3 scripts/algebra_verbose_gen.py 600 414 .cache/g41r_vb", tail=1)
    sh(".venv/bin/python3 scripts/algebra_dag7_gen.py 800 415 .cache/g41r_dag.jsonl", tail=1)
    sh("cat .cache/g41r_v1.jsonl .cache/g41r_v2.jsonl .cache/g41r_v3.jsonl "
   ".cache/g41r_vb_verbose.jsonl .cache/g41r_dag.jsonl > .cache/gen41_repair.jsonl "
   "&& wc -l .cache/gen41_repair.jsonl", tail=1)

print("=== E41 2/10: precompute repair states ===", flush=True)
if _os.path.exists(".cache/phase1_alg_states_gen41repair.npz"):
    print("    [skip] repair states banked", flush=True)
else:
    sh(".venv/bin/python3 scripts/phase1_algebra_head.py --precompute",
       {"ALG_TRAIN": ".cache/gen41_repair.jsonl", "ALG_TRAIN_NAME": "gen41repair",
        "PRECOMPUTE_ONLY": "gen41repair"}, tail=1)

print("=== E41 3/10: specialist remine vs g22 ===", flush=True)
if _os.path.exists(".cache/phase1_gen23v5_nack.safetensors"):
    print("    [skip] specialist banked", flush=True)
    e = None
else:
    e = {"ALG_TRAIN": ".cache/gen41_repair.jsonl", "ALG_TRAIN_NAME": "gen41repair",
     "ALG_CKPT": PARSER, "NACK_CKPT": NEW_NACK, "NACK_SPLIT": "train"}
if e is not None:
    sh(".venv/bin/python3 scripts/phase1_algebra_nack.py --prep", e, tail=2)
    e2 = dict(e); e2.update({"STEPS": "4000", "LR": "1e-4", "BATCH": "8", "SEED": "22"})
    sh(".venv/bin/python3 scripts/phase1_algebra_nack.py --train", e2, tail=2)

print("=== E41 4/10: monitor centroids (g22 fst space, g22 family) ===", flush=True)
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
p = build_params(0); sd = safe_load(".cache/g41_onemass_refold.safetensors")
for k in p: p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
z = np.load(".cache/phase1_alg_states_form3.npz")
st = np.load(".cache/phase1_alg_states_form3_states.npy", mmap_mode="r")
assert st.shape[0] == z["tokmask"].shape[0] == 94100   # ALIGNED pair (fossil fixed)
idx = list(range(0, st.shape[0], max(1, st.shape[0]//3000)))[:3000]
fst = compute_fst(p, st, z["tokmask"], z["sent"], idx)
by = {}
for r in range(len(idx)):
    for j, kd in enumerate(head_kinds(p, fst[r])):
        if kd: by.setdefault(kd, []).append(fst[r, j])
lib = {k: (lambda c: c/np.linalg.norm(c))(np.mean(v,0)) for k,v in by.items()}
np.savez(".cache/monitor_centroids_gen41.npz", **lib)
print("centroids:", sorted(lib))
'''
open(".cache/_e41_s4.py", "w").write(S4)
sh(".venv/bin/python3 .cache/_e41_s4.py", tail=1)

print("=== E41 5/10: mouth rebuild (g22 family) + length refit ===", flush=True)
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
fam, _ = pooled_npz(".cache/phase1_alg_states_form3.npz",
                    ".cache/phase1_alg_states_form3_states.npy", cap=12000)
rng = np.random.RandomState(41)
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
np.savez(".cache/recognition_mouth_gen41.npz", bank=bank,
         thr_knn=np.float32(thr), coef=coef.astype(np.float32))
print(f"[mouth-41] bank rebuilt (g22 family); length-controlled thr {thr:.4f}")
'''
open(".cache/_e41_s5.py", "w").write(S5)
sh(".venv/bin/python3 .cache/_e41_s5.py", tail=1)

print("=== E41 6/10: disjoint census under the fresh mouth ===", flush=True)
# THE CENSUS_DISJOINT NO-OP FIX (consumer audit 2026-08-02: the env var was
# never read; every entourage 'disjoint census' since gen-13 was full-pool.
# Ruled: 'fix rides the next entourage' — this is that entourage). SKIP_IDX
# derived MECHANICALLY: census-pool items trained-verbatim in the deployed
# mix (sha ∩ mix — the corrected consumption key).
_skip = subprocess.run(
    [".venv/bin/python3", "-c", """
import json, hashlib, re
h=[json.loads(l) for l in open('.cache/math_harvest_v0.jsonl')]
filt=[i for i,x in enumerate(h) if x['level'] in ('Level 1','Level 2','Level 3')
      and len(x['problem'])<300 and 'asy]' not in x['problem']
      and all(int(n)<=300 for n in re.findall(r'\\d+',x['problem']))]
pool=filt[:100]
mix={hashlib.sha256(json.loads(l)['text'].encode()).hexdigest()
     for l in open('.cache/form_mix3.jsonl')}
skip=[str(k) for k,i in enumerate(pool)
      if hashlib.sha256(h[i]['problem'].encode()).hexdigest() in mix]
print(','.join(skip))
"""], capture_output=True, text=True).stdout.strip()
print(f"    [skip-idx] {len(_skip.split(',')) if _skip else 0} census items trained-verbatim (sha∩mix)", flush=True)
sh(".venv/bin/python3 scripts/gen11_census.py",
   {"GATE_CKPT": PARSER, "SKIP_IDX": _skip}, tail=2)

print("=== E41 7/10: DISSENT-OVERLAP READ (manifest panel) ===", flush=True)
# the waived member-votes stage produces the new gate's lattice here
if not os.path.exists(".cache/lattice_gen41_H.json"):
    print("    [pre] member votes for g23 (the waived battery stage)", flush=True)
    sh(".venv/bin/python3 scripts/lattice_member_votes.py",
       {"MEMBER_CKPT": ".cache/g41_onemass_refold.safetensors", "MEMBER_HW": "512",
        "MEMBER_DUP": "1", "OUT": ".cache/lattice_gen41_H.json"}, tail=1)
S7 = r'''
import json
from collections import Counter
def maj(v):
    vs=[x for x in v if x is not None]
    return Counter(vs).most_common(1)[0] if vs else (None,0)
m = json.load(open(".cache/GENERATION.json"))
lf = sorted(m["panel"]["lattices"].values())
armb = json.load(open(lf[0]))["bigtest"]
c2x = json.load(open(lf[1]))["bigtest"]
def dissent_set(gate_votes):
    out = set()
    for i in range(1500):
        gt, gc = maj(gate_votes[i]); at,_ = maj(armb[i]); ct,_ = maj(c2x[i])
        if gc == 5 and not (at == gt and ct == gt):
            out.add(i)
    return out
d23 = dissent_set(json.load(open(".cache/lattice_gen23_H.json"))["bigtest"])
d41 = dissent_set(json.load(open(".cache/lattice_gen41_H.json"))["bigtest"])
ov = d23 & d41
print(f"[dissent-overlap] gen-23: {len(d23)} | gen-41: {len(d41)} | "
      f"OVERLAP {len(ov)} — "
      f"{'STABLE dissent family (structural)' if len(ov) >= 0.5*max(len(d41),1) else 'dissent ROTATES (population-driven)'}")
json.dump({"d23": sorted(d23), "d41": sorted(d41), "overlap": sorted(ov)},
          open(".cache/dissent_overlap_41.json", "w"))
'''
open(".cache/_e41_s7.py", "w").write(S7)
sh(".venv/bin/python3 .cache/_e41_s7.py", tail=1)

print("=== E41 8/10: collapse re-read (dashboard accrual) ===", flush=True)
sh(".venv/bin/python3 scripts/collapse_probe.py",
   {"COLLAPSE_CKPT": PARSER}, tail=3)

print("=== E41 9/10: manifest member refresh + RE-HASH ===", flush=True)
m = json.load(open(".cache/GENERATION.json"))
m["specialist_ckpt"] = NEW_NACK
m["monitor_centroids"] = ".cache/monitor_centroids_gen41.npz"
m["mouth"] = ".cache/recognition_mouth_gen41.npz"
m["waivers"] = {"panel": "cert-v2 members per panel.lattices (re-audition "
                "rides the bench queue; lineage-adjacency re-priced at next refresh)"}
m["notes"] = (m.get("notes", "") +
              " | 2026-08-10 ENTOURAGE-41 PAID (entourage22.py): specialist "
              "remined vs g22, centroids + mouth rebuilt in g22 space (aligned "
              "state pair, fossils fixed), dissent-overlap banked, standing "
              "rehearsal riding as stage 10.")


def _h16(path): return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]


m["hashes"] = {"parser": _h16(m["parser_ckpt"]),
               "specialist": _h16(m["specialist_ckpt"]),
               "centroids": _h16(m["monitor_centroids"]),
               "mouth": _h16(m["mouth"]),
               "train": _h16(m["corpora"]["train"])}
json.dump(m, open(".cache/GENERATION.json", "w"), indent=1)
print("manifest refreshed + re-hashed (the atomic law, entourage edition)")

print("=== E41 10/10: THE STANDING REHEARSAL (outside air, both tiers) ===", flush=True)
sh(".venv/bin/python3 scripts/wild_frontier_fixture.py",
   {"WFF_CKPT": PARSER, "WFF_OUT": ".cache/wild_frontier_fixture_gen41_entourage.json"},
   tail=4)
print("=== ENTOURAGE-41 SETTLED — the gate may read ===", flush=True)
