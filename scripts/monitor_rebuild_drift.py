"""monitor_rebuild_drift.py — rebuild the waist monitor in the tranche head's
fst space + the CENTROID-DRIFT read (2026-07-10): mechanism-(b) evidence for
the +68. PREDICTION (hypothesis b, registered): old-kind separations WIDEN in
the new space (reorganization-toward-separation; neural-collapse frame) —
measured as pairwise centroid cosine + within/between ratio on the SAME old
corpus, per-space (basis-free geometry, not cross-space cosines).
Also: the rebuilt library's AUC on the CURRENT stack's committed-wrongs.
USAGE: DEV=AMD .venv/bin/python3 scripts/monitor_rebuild_drift.py
"""
import json, os, sys
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
os.environ["ALG2"] = "1"
from phase1_algebra_head import L_FAC, build_params
from waist_abstention_probe import compute_fst
from survivor_multiplicity import midrank_auc
from tinygrad.nn.state import safe_load

def load_np(path):
    z = np.load(path)
    return z["states"], z["tokmask"], z["sent"]

def head_kinds(p, hd_keys, fst_rows, four_way):
    g = lambda k: p[k].detach().numpy()
    hp, hpb = g("h_pres"), g("h_pres_b")
    hf, hfb = g("h_ftype"), g("h_ftype_b")
    ho, hob = g("h_op"), g("h_op_b")
    out = []
    for v in fst_rows:
        if v @ hp[:, 0] + hpb[0] <= 0:
            out.append(None); continue
        ft = int(np.argmax(v @ hf + hfb))
        if four_way:
            out.append(("given", "rel_add", "rel_mul", "mod", "sel")[
                1 + int(np.argmax(v @ ho + hob)) if ft == 0 else
                (0 if ft == 1 else (3 if ft == 2 else 4))])
        else:
            out.append("given" if ft == 1 else
                       ("rel_add" if np.argmax(v @ ho + hob) == 0 else "rel_mul"))
    return out

def space_stats(name, ckpt, four_way, states, tokmask, sent, n):
    os.environ["ALG_BREATH"] = "1"
    p = build_params(0)
    sd = safe_load(ckpt)
    # legacy ckpt lacks ALG2 keys: build under matching env
    for k in p:
        p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
    fst = compute_fst(p, states, tokmask, sent, list(range(n)))
    by = {}
    for i in range(n):
        kinds = head_kinds(p, None, fst[i], four_way)
        for j, kd in enumerate(kinds):
            if kd:
                by.setdefault(kd, []).append(fst[i, j])
    cent = {k: np.mean(v, 0) for k, v in by.items()}
    cn = {k: c / np.linalg.norm(c) for k, c in cent.items()}
    old3 = [k for k in ("given", "rel_add", "rel_mul") if k in cn]
    pair = {f"{a}~{b}": float(cn[a] @ cn[b])
            for x, a in enumerate(old3) for b in old3[x + 1:]}
    within = {}
    for k in old3:
        V = np.stack(by[k]); Vn = V / np.linalg.norm(V, axis=1, keepdims=True)
        within[k] = float((Vn @ cn[k]).mean())
    print(f"  [{name}] pairwise centroid cos (old kinds): " +
          " ".join(f"{k}={v:.3f}" for k, v in pair.items()))
    print(f"  [{name}] within-kind cos-to-centroid: " +
          " ".join(f"{k}={v:.3f}" for k, v in within.items()))
    return cn, by, p

# OLD corpus, both spaces (drift read on identical inputs)
samples_old = [json.loads(l) for l in open(".cache/algebra_nl_train.jsonl")]
st, tk, se = load_np(".cache/phase1_alg_states_train.npz")
n_old = len(samples_old)
print("=== DRIFT READ (same old-corpus slots, per-space geometry) ===")
os.environ["ALG2"] = "0"
space_stats("legacy", ".cache/phase1_algebra_head.safetensors", False,
            st, tk, se, n_old)
os.environ["ALG2"] = "1"
cn_new, _, p_new = space_stats("tranche", ".cache/phase1_algebra2_head.safetensors",
                               True, st, tk, se, n_old)

# FULL library in tranche space (mixed corpus) + AUC on current stack wrongs
st2, tk2, se2 = load_np(".cache/phase1_alg_states_alg2train.npz")
n2 = st2.shape[0]
fst2 = compute_fst(p_new, st2, tk2, se2, list(range(n2)))
by2 = {}
for i in range(n2):
    for j, kd in enumerate(head_kinds(p_new, None, fst2[i], True)):
        if kd:
            by2.setdefault(kd, []).append(fst2[i, j])
lib = {k: (lambda c: c / np.linalg.norm(c))(np.mean(v, 0))
       for k, v in by2.items()}
np.savez(".cache/monitor_centroids_alg2.npz",
         **{k: v for k, v in lib.items()})
print(f"\n[library] rebuilt in tranche space: {sorted(lib)} -> "
      f".cache/monitor_centroids_alg2.npz")

aud = np.load(".cache/deploy_audit_alg2test.npz")
correct = {int(i): int(c) for i, c in zip(aud["idx"], aud["correct"])}
samples_t = [json.loads(l) for l in open(".cache/algebra2_nl_test.jsonl")]
st3, tk3, se3 = load_np(".cache/phase1_alg_states_alg2test.npz")
idx = sorted(correct)
fst3 = compute_fst(p_new, st3, tk3, se3, idx)
sc, lb = [], []
for r, i in enumerate(idx):
    w = 1.0
    for j, kd in enumerate(head_kinds(p_new, None, fst3[r], True)):
        if kd and kd in lib:
            v = fst3[r, j]
            w = min(w, float((v / max(np.linalg.norm(v), 1e-9)) @ lib[kd]))
    sc.append(1 - w); lb.append(1 - correct[i])
sc, lb = np.array(sc), np.array(lb)
print(f"[monitor v2] AUC on current-stack committed-wrongs "
      f"({int(lb.sum())}/{len(lb)}): "
      f"{midrank_auc(sc[lb == 1], sc[lb == 0]):.3f} (v1 was 0.728)")
