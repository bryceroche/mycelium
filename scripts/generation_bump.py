"""generation_bump.py — THE ATOMIC GENERATION BUMP, v1 (2026-07-10).
One transaction: mine -> precompute -> specialist -> centroids -> mouth
(recalibrate + gradient read) -> evals -> MANIFEST WRITE (sole commit point).
Every artifact built ALONGSIDE the prior generation; a failure at any stage
leaves the system provably at gen-N (nothing overwritten, manifest untouched).
--inject-fail=K aborts deliberately after stage K (the abort-path witness).

GEN-5 SPECIFICS: parser = the bilingual head (promoted — the register
experiment's byproduct out-benched production; 2nd instrument-side-effect-
becomes-headline this month). GRADIENT HYPOTHESIS (relay, registered before
the number): mostly-LOCAL — MATH-500 distance closes 10-20% under the new
calibration, not half; generator-verbose is our skeletons in narrative
clothes, and style lives in distributional properties the generator doesn't
imitate yet. >20% closure = the staircase is transitive and cheap.
"""
import argparse, json, os, subprocess, sys
sys.path.insert(0, "."); sys.path.insert(0, "scripts")

ENV = {"DEV": "AMD", "ALG2": "1", "ALG_FTYPES": "6"}
PARSER = ".cache/phase1_bilingual_head.safetensors"
SPEC = ".cache/phase1_bilingual_nack.safetensors"

def sh(cmd, extra=None, tail=3):
    env = dict(os.environ); env.update(ENV); env.update(extra or {})
    r = subprocess.run(cmd, shell=True, env=env, capture_output=True, text=True)
    out = (r.stdout + r.stderr).strip().splitlines()
    for l in out[-tail:]:
        print(f"    {l}")
    if r.returncode != 0:
        raise RuntimeError(f"stage command failed: {cmd[:80]}")
    return out

def stage1_mine():
    print("[bump 1/6] fresh mining corpora (4 registers)")
    sh(".venv/bin/python3 scripts/algebra_nl_gen.py --n 800 --seed 111 --out .cache/g5r_v1.jsonl --teeth 0.8", tail=1)
    sh(".venv/bin/python3 scripts/algebra2_nl_gen.py --n 800 --seed 112 --out .cache/g5r_v2.jsonl --teeth 0.8 --token-budget 250", tail=1)
    sh(".venv/bin/python3 scripts/algebra3_nl_gen.py --n 800 --seed 113 --out .cache/g5r_v3.jsonl --teeth 0.8 --token-budget 250", tail=1)
    sh(".venv/bin/python3 scripts/algebra_verbose_gen.py 600 114 .cache/g5r_vb", tail=1)
    sh("cat .cache/g5r_v1.jsonl .cache/g5r_v2.jsonl .cache/g5r_v3.jsonl .cache/g5r_vb_verbose.jsonl > .cache/gen5_repair.jsonl && wc -l .cache/gen5_repair.jsonl", tail=1)

def stage2_precompute():
    print("[bump 2/6] precompute (repair states)")
    sh(".venv/bin/python3 scripts/phase1_algebra_head.py --precompute",
       {"ALG_TRAIN": ".cache/gen5_repair.jsonl", "ALG_TRAIN_NAME": "gen5repair",
        "PRECOMPUTE_ONLY": "gen5repair"}, tail=1)

def stage3_specialist():
    print("[bump 3/6] specialist mine+train (in-generation)")
    e = {"ALG_TRAIN": ".cache/gen5_repair.jsonl", "ALG_TRAIN_NAME": "gen5repair",
         "ALG_CKPT": PARSER, "NACK_CKPT": SPEC, "NACK_SPLIT": "train"}
    sh(".venv/bin/python3 scripts/phase1_algebra_nack.py --prep", e, tail=2)
    e.update({"STEPS": "6000", "LR": "3e-4", "BATCH": "8"})
    sh(".venv/bin/python3 scripts/phase1_algebra_nack.py --train", e, tail=2)

def stage4_centroids():
    print("[bump 4/6] monitor centroids (bilingual fst space)")
    code = '''
import sys, os, json; sys.path.insert(0,"."); sys.path.insert(0,"scripts")
os.environ["ALG2"]="1"; os.environ["ALG_FTYPES"]="6"
import numpy as np
from phase1_algebra_head import L_FAC, build_params
from waist_abstention_probe import compute_fst
from monitor_rebuild_drift import head_kinds
from tinygrad.nn.state import safe_load
p = build_params(0); sd = safe_load("%s")
for k in p: p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
z = np.load(".cache/phase1_alg_states_mvtrain.npz")
idx = list(range(0, z["states"].shape[0], 3))[:3000]
fst = compute_fst(p, z["states"], z["tokmask"], z["sent"], idx)
by = {}
for r in range(len(idx)):
    for j, kd in enumerate(head_kinds(p, None, fst[r], True)):
        if kd: by.setdefault(kd, []).append(fst[r, j])
lib = {k: (lambda c: c/np.linalg.norm(c))(np.mean(v,0)) for k,v in by.items()}
np.savez(".cache/monitor_centroids_gen5.npz", **lib)
print("centroids:", sorted(lib))
''' % PARSER
    open("/tmp/claude-1000/-home-bryce-mycelium/d57518e2-8afb-4987-989f-292859d8ca53/scratchpad/_s4.py", "w").write(code)
    sh(".venv/bin/python3 /tmp/claude-1000/-home-bryce-mycelium/d57518e2-8afb-4987-989f-292859d8ca53/scratchpad/_s4.py", tail=1)

def stage5_mouth():
    print("[bump 5/6] mouth recalibration + THE GRADIENT READ")
    code = '''
import sys, os, json; sys.path.insert(0,"."); sys.path.insert(0,"scripts")
import numpy as np
from phase1_algebra_head import T_ALG, TOKENIZER_JSON
from beacon_closing_arm import recompute_states
from tokenizers import Tokenizer
def pooled(path):
    z = np.load(path)
    st, tk = z["states"].astype(np.float32), z["tokmask"].astype(np.float32)
    v = (st*tk[:,:,None]).sum(1)/np.maximum(tk.sum(1)[:,None],1)
    return v/np.linalg.norm(v,axis=1,keepdims=True)
fam = pooled(".cache/phase1_alg_states_mvtrain.npz")
rng = np.random.RandomState(0)
bank = fam[rng.choice(len(fam), 2000, replace=False)]
native = np.vstack([pooled(".cache/phase1_alg_states_vtest.npz"),
                    pooled(".cache/phase1_alg_states_alg4test.npz"),
                    pooled(".cache/phase1_alg_states_bigtest.npz")])
def knn(V):
    d = 1.0 - V @ bank.T
    return np.sort(d, axis=1)[:, :8].mean(1)
nk = knn(native); thr = float(np.percentile(nk, 99))
rows = [json.loads(l) for l in open(".cache/math500_test.jsonl")]
tok = Tokenizer.from_file(TOKENIZER_JSON)
ids = np.zeros((len(rows), T_ALG), np.int32); msk = np.zeros((len(rows), T_ALG), np.float32)
kept = []
for i, r in enumerate(rows):
    e = tok.encode(r["problem"])
    if len(e.ids) > T_ALG: continue
    kept.append(i); ids[i,:len(e.ids)] = e.ids; msk[i,:len(e.ids)] = 1.0
st = recompute_states(ids)
fv = (st.astype(np.float32)*msk[:,:,None]).sum(1)/np.maximum(msk.sum(1)[:,None],1)
fv = fv/np.maximum(np.linalg.norm(fv,axis=1,keepdims=True),1e-9)
fk = knn(fv[kept])
np.savez(".cache/recognition_mouth_gen5.npz", bank=bank, thr_knn=thr)
old_m, old_t = 0.2531, 0.0439   # gen-4 calibration: MATH-500 mean, native thr
new_m = float(fk.mean())
print(f"gen-5 mouth: native thr {thr:.4f} | MATH-500 mean kNN {new_m:.4f} "
      f"(gen-4: {old_m} over thr {old_t}) | foreign refused {float((fk>thr).mean()):.1%}")
gap_old = old_m - old_t; gap_new = new_m - thr
print(f"GRADIENT READ: distance-over-threshold {gap_old:.4f} -> {gap_new:.4f} "
      f"= {100*(1-gap_new/gap_old):.0f}%% closed (hypothesis: 10-20%% local / >20%% transitive)")
'''
    open("/tmp/claude-1000/-home-bryce-mycelium/d57518e2-8afb-4987-989f-292859d8ca53/scratchpad/_s5.py", "w").write(code)
    sh(".venv/bin/python3 /tmp/claude-1000/-home-bryce-mycelium/d57518e2-8afb-4987-989f-292859d8ca53/scratchpad/_s5.py", {"DEV": "AMD"}, tail=3)

def stage6_evals_and_manifest():
    print("[bump 6/6] evals + MANIFEST (the commit point)")
    bars = {}
    for nm, (jp, sp) in {"bigtest": (".cache/algebra_nl_bigtest.jsonl", "bigtest"),
                         "alg2test": (".cache/algebra2_nl_test.jsonl", "alg2test"),
                         "alg4test": (".cache/algebra4_nl_test.jsonl", "alg4test"),
                         "vtest": (".cache/algv_test_verbose.jsonl", "vtest")}.items():
        out = sh(".venv/bin/python3 scripts/phase1_algebra_head.py --eval",
                 {"ALG_CKPT": PARSER, "ALG_TEST": jp, "ALG_TEST_NAME": sp,
                  "ALG_TRAIN": ".cache/algebra_mixedv_train.jsonl",
                  "ALG_TRAIN_NAME": "mvtrain"}, tail=1)
        tot = [l for l in out if "TOTAL" in l][-1]
        bars[nm] = int(tot.split(",")[1].strip().split("/")[0].replace("ANSWER", "").strip())
    print(f"    bars: {bars}")
    import hashlib
    def fh(p):
        h = hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]
        return h
    m = {"gen_id": 5, "date": "2026-07-10",
         "env": {"ALG2": "1", "ALG_FTYPES": "6"},
         "parser_ckpt": PARSER, "specialist_ckpt": SPEC,
         "monitor_centroids": ".cache/monitor_centroids_gen5.npz",
         "mouth": ".cache/recognition_mouth_gen5.npz",
         "corpora": {"train": ".cache/algebra_mixedv_train.jsonl",
                     "test": ".cache/algebra4_nl_test.jsonl",
                     "register_test": ".cache/algv_test_verbose.jsonl",
                     "repair": ".cache/gen5_repair.jsonl"},
         "regression_bars": bars,
         "notes": "gen-5: bilingual parser promoted (register experiment's "
                  "byproduct out-benched production); first SCRIPTED bump; "
                  "mouth recalibrated post-verbose; gradient datapoint logged."}
    m["hashes"] = {k: fh(v) for k, v in
                   [("parser", PARSER), ("specialist", SPEC),
                    ("centroids", m["monitor_centroids"]), ("mouth", m["mouth"])]
                   + list(m["corpora"].items())}
    json.dump(m, open(".cache/GENERATION.json", "w"), indent=1)
    print(f"    [gen] COMMITTED gen-5 ({len(m['hashes'])} artifacts pinned)")

STAGES = [stage1_mine, stage2_precompute, stage3_specialist, stage4_centroids,
          stage5_mouth, stage6_evals_and_manifest]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inject-fail", type=int, default=0)
    a = ap.parse_args()
    for i, st in enumerate(STAGES, 1):
        st()
        if a.inject_fail == i:
            raise RuntimeError(f"INJECTED FAILURE after stage {i} — "
                               f"manifest must still read the prior generation")
    print("[bump] TRANSACTION COMPLETE")
