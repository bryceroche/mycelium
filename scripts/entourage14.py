"""entourage14.py — ENTOURAGE-14 (2026-07-20): the owed duty paid, and the
inline-chain era closed — the entourage is a COMMITTED SCRIPT from this
generation forward (discipline -> mechanism, the house conversion).

Stages (the e13 chain's shape, gen-14 params):
  1. fresh repair corpora (5 registers, E14 seeds)   -> gen14_repair.jsonl
  2. precompute repair states
  3. specialist remine vs THE GEN-14 PARSER          -> phase1_gen14_nack
  4. monitor centroids in GEN-14 fst space           -> monitor_centroids_gen14
  5. mouth rebuild (m13train family) + length refit  -> recognition_mouth_gen14
  6. disjoint census under the fresh mouth (informational — the
     post-entourage read)
  7. MANIFEST MEMBER REFRESH (same-generation transaction, the sync
     law's cure): specialist/monitor/mouth -> gen-14 artifacts, the
     one-generation waiver RETIRED, the panama guard + adversarial
     fixture enter as WATCHER members (decision-path wiring still rides
     the next promotion's battery — stated in the manifest note).
Every artifact is built ALONGSIDE the prior generation's; a failure at
any stage leaves the composed stack exactly as it was.
"""
import json, os, subprocess, sys
sys.path.insert(0, "."); sys.path.insert(0, "scripts")

ENV = {"DEV": "AMD", "ALG2": "1", "ALG_FTYPES": "6", "ALG_DUP": "1"}
PARSER = ".cache/phase1_gen14_head.safetensors"
NEW_NACK = ".cache/phase1_gen14_nack.safetensors"


def sh(cmd, extra=None, tail=2):
    env = dict(os.environ); env.update(ENV); env.update(extra or {})
    r = subprocess.run(cmd, shell=True, env=env, capture_output=True, text=True)
    for l in (r.stdout + r.stderr).strip().splitlines()[-tail:]:
        print(f"    {l}", flush=True)
    if r.returncode != 0:
        raise RuntimeError(f"stage failed: {cmd[:90]}")


print("=== E14 1/7: fresh repair corpora ===", flush=True)
sh(".venv/bin/python3 scripts/algebra_nl_gen.py --n 800 --seed 141 --out .cache/g14r_v1.jsonl --teeth 0.8", tail=1)
sh(".venv/bin/python3 scripts/algebra2_nl_gen.py --n 800 --seed 142 --out .cache/g14r_v2.jsonl --teeth 0.8 --token-budget 250", tail=1)
sh(".venv/bin/python3 scripts/algebra3_nl_gen.py --n 800 --seed 143 --out .cache/g14r_v3.jsonl --teeth 0.8 --token-budget 250", tail=1)
sh(".venv/bin/python3 scripts/algebra_verbose_gen.py 600 144 .cache/g14r_vb", tail=1)
sh(".venv/bin/python3 scripts/algebra_dag7_gen.py 800 145 .cache/g14r_dag.jsonl", tail=1)
sh("cat .cache/g14r_v1.jsonl .cache/g14r_v2.jsonl .cache/g14r_v3.jsonl "
   ".cache/g14r_vb_verbose.jsonl .cache/g14r_dag.jsonl > .cache/gen14_repair.jsonl "
   "&& wc -l .cache/gen14_repair.jsonl", tail=1)

print("=== E14 2/7: precompute repair states ===", flush=True)
sh(".venv/bin/python3 scripts/phase1_algebra_head.py --precompute",
   {"ALG_TRAIN": ".cache/gen14_repair.jsonl", "ALG_TRAIN_NAME": "gen14repair",
    "PRECOMPUTE_ONLY": "gen14repair"}, tail=1)

print("=== E14 3/7: specialist remine vs gen-14 ===", flush=True)
e = {"ALG_TRAIN": ".cache/gen14_repair.jsonl", "ALG_TRAIN_NAME": "gen14repair",
     "ALG_CKPT": PARSER, "NACK_CKPT": NEW_NACK, "NACK_SPLIT": "train"}
sh(".venv/bin/python3 scripts/phase1_algebra_nack.py --prep", e, tail=2)
e2 = dict(e); e2.update({"STEPS": "4000", "LR": "1e-4", "BATCH": "8", "SEED": "14"})
sh(".venv/bin/python3 scripts/phase1_algebra_nack.py --train", e2, tail=2)

print("=== E14 4/7: monitor centroids (gen-14 fst space, m13train family) ===", flush=True)
S4 = r'''
import sys, os, json; sys.path.insert(0,"."); sys.path.insert(0,"scripts")
os.environ["ALG2"]="1"; os.environ["ALG_FTYPES"]="6"; os.environ["ALG_DUP"]="1"
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
        out.append(("given","rel_add","rel_mul","mod","sel","pct","fdiv")[
            1 + int(np.argmax(v @ ho + hob)) if ft == 0 else
            (0 if ft == 1 else (3 if ft == 2 else
             (4 if ft == 3 else (5 if ft == 4 else 6))))])
    return out
p = build_params(0); sd = safe_load(".cache/phase1_gen14_head.safetensors")
for k in p: p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
z = np.load(".cache/phase1_alg_states_m13train.npz")
if "states" in z.files:
    st = z["states"]
else:
    tk = z["tokmask"]
    st = np.load(".cache/phase1_alg_states_m13train_states.npy", mmap_mode="r")
idx = list(range(0, st.shape[0], max(1, st.shape[0]//3000)))[:3000]
fst = compute_fst(p, st, z["tokmask"], z["sent"], idx)
by = {}
for r in range(len(idx)):
    for j, kd in enumerate(head_kinds(p, fst[r])):
        if kd: by.setdefault(kd, []).append(fst[r, j])
lib = {k: (lambda c: c/np.linalg.norm(c))(np.mean(v,0)) for k,v in by.items()}
np.savez(".cache/monitor_centroids_gen14.npz", **lib)
print("centroids:", sorted(lib))
'''
open(".cache/_e14_s4.py", "w").write(S4)
sh(".venv/bin/python3 .cache/_e14_s4.py", tail=1)

print("=== E14 5/7: mouth rebuild (m13train family) + length refit ===", flush=True)
S5 = r'''
import sys, os, json; sys.path.insert(0,"."); sys.path.insert(0,"scripts")
os.environ["ALG2"]="1"; os.environ["ALG_FTYPES"]="6"
import numpy as np
from phase1_algebra_head import T_ALG
def pooled_npz(path, npy=None):
    z = np.load(path)
    st = z["states"] if "states" in z.files else np.load(npy, mmap_mode="r")
    tk = z["tokmask"]
    n = st.shape[0]; out = np.zeros((n, st.shape[2]), np.float32)
    Ls = tk.sum(1)
    for s0 in range(0, n, 256):
        sl = slice(s0, min(s0+256, n))
        a = np.asarray(st[sl]).astype(np.float32); m = tk[sl].astype(np.float32)
        out[sl] = (a*m[:,:,None]).sum(1)/np.maximum(m.sum(1)[:,None],1)
    return out/np.linalg.norm(out,axis=1,keepdims=True), Ls
fam, _ = pooled_npz(".cache/phase1_alg_states_m13train.npz",
                    ".cache/phase1_alg_states_m13train_states.npy")
rng = np.random.RandomState(14)
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
np.savez(".cache/recognition_mouth_gen14.npz", bank=bank,
         thr_knn=np.float32(thr), coef=coef.astype(np.float32))
print(f"[mouth-14] bank rebuilt (m13train family); length-controlled thr {thr:.4f}")
'''
open(".cache/_e14_s5.py", "w").write(S5)
sh(".venv/bin/python3 .cache/_e14_s5.py", tail=1)

print("=== E14 6/7: disjoint census under the fresh mouth (informational) ===", flush=True)
sh(".venv/bin/python3 scripts/gen11_census.py",
   {"GATE_CKPT": PARSER, "CENSUS_DISJOINT": "1"}, tail=2)

print("=== E14 7/7: manifest member refresh (same-generation transaction) ===", flush=True)
m = json.load(open(".cache/GENERATION.json"))
m["specialist_ckpt"] = NEW_NACK
m["monitor_centroids"] = ".cache/monitor_centroids_gen14.npz"
m["mouth"] = ".cache/recognition_mouth_gen14.npz"
m["watchers"] = {
    "panama_guard_lexicon": ".cache/panama_guard_lexicon.json",
    "adversarial_scope_fixture": ".cache/adversarial_scope_fixture.jsonl",
    "note": "guard decision-path wiring rides the NEXT promotion's battery; "
            "lexicon regenerates per generation (panama_guard.py --build)"}
m["waivers"] = {"panel": m.get("waivers", {}).get("panel",
                "cert-v2 members armB + cap2x (panel-eligible bench)")}
m["notes"] = (m.get("notes", "") +
              " | 2026-07-20 ENTOURAGE-14 PAID (committed chain, entourage14.py): "
              "specialist remined vs gen-14, centroids+mouth rebuilt in-generation, "
              "one-generation waiver RETIRED, guard+fixture enter as watchers.")
json.dump(m, open(".cache/GENERATION.json", "w"), indent=1)
print("[manifest] entourage-14 complete — members refreshed, waiver retired", flush=True)
print("=== ENTOURAGE-14 DONE ===", flush=True)
