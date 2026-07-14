"""latent_audit.py — THE LATENT-SPACE AUDIT (2026-07-13, gut #15).
Three probes, no training:
 (A) DRIFT: orthogonal Procrustes between gen-5 and gen-9b centroid
     constellations + per-kind alignment before/after.
 (B) STRATIFICATION: per-kind fst NORM longitudinal across the bench
     (gen-6..gen-11 heads, one fixed m9btrain sample — head-side only;
     the mouth is immune by construction, frozen trunk).
 (C) SEPARATION: per book-1 pair, pooled-TRUNK cosine raw<->dialect;
     the chronic [45] family vs its banked siblings.
"""
import json, sys, os
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
os.environ.setdefault("ALG2", "1"); os.environ.setdefault("ALG_FTYPES", "6")

# ---------- (A) DRIFT ----------
g5 = np.load(".cache/monitor_centroids_gen5.npz")
g9 = np.load(".cache/monitor_centroids_gen9b.npz")
kinds = sorted(set(g5.files) & set(g9.files))
A = np.stack([g5[k] for k in kinds])
B = np.stack([g9[k] for k in kinds])
U, _, Vt = np.linalg.svd(A.T @ B)
R = U @ Vt
A_rot = A @ R
print("=== (A) DRIFT: Procrustes gen-5 -> gen-9b centroids ===")
print(f"  kinds: {kinds}")
raw_cos = [float(a @ b) for a, b in zip(A, B)]
rot_cos = [float(a @ b) for a, b in zip(A_rot, B)]
for k, rc, ac in zip(kinds, raw_cos, rot_cos):
    print(f"  {k:8s}: raw cos {rc:+.3f} -> aligned {ac:+.3f}")
res = float(np.linalg.norm(A_rot - B) / np.linalg.norm(B))
print(f"  mean raw cos {np.mean(raw_cos):+.3f} | aligned {np.mean(rot_cos):+.3f} "
      f"| Procrustes residual {res:.3f}")
print("  (high aligned cos + low residual = rotation/translation only;")
print("   low aligned cos = the constellation itself reorganized)")

# ---------- (B) STRATIFICATION ----------
from phase1_algebra_head import build_params, STATES_NPZ, STATES_NPY, L_FAC
from waist_abstention_probe import compute_fst
from tinygrad.nn.state import safe_load

z = np.load(STATES_NPZ.format(split="m9btrain"))
st = np.load(STATES_NPY.format(split="m9btrain"), mmap_mode="r")
idx = list(range(0, st.shape[0], st.shape[0] // 1500))[:1500]

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

BENCH = [("gen6", ".cache/phase1_gen6_head.safetensors", "0"),
         ("gen7b", ".cache/phase1_gen7b_head.safetensors", "0"),
         ("gen8", ".cache/phase1_gen8_head.safetensors", "0"),
         ("gen9b", ".cache/phase1_gen9b_head.safetensors", "1"),
         ("gen10", ".cache/phase1_gen10_head.safetensors", "1"),
         ("gen11", ".cache/phase1_gen11_head.safetensors", "1")]
print("\n=== (B) STRATIFICATION: per-kind mean fst NORM across the bench ===")
all_kinds = ("given", "rel_add", "rel_mul", "mod", "sel", "pct", "fdiv")
print("  gen    " + "".join(f"{k:>9s}" for k in all_kinds))
for name, ckpt, dup in BENCH:
    os.environ["ALG_DUP"] = dup
    p = build_params(0)
    sd = safe_load(ckpt)
    assert set(sd.keys()) == set(p.keys()), name
    for k in p:
        p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
    fst = compute_fst(p, st, z["tokmask"], z["sent"], idx)
    norms = {k: [] for k in all_kinds}
    for r in range(len(idx)):
        for j, kd in enumerate(head_kinds(p, fst[r])):
            if kd:
                norms[kd].append(float(np.linalg.norm(fst[r, j])))
    print(f"  {name:6s}" + "".join(
        f"{np.mean(norms[k]):9.2f}" if norms[k] else f"{'—':>9s}"
        for k in all_kinds))

# ---------- (C) SEPARATION ----------
os.environ["ALG_DUP"] = "1"
from phase1_algebra_head import T_ALG, TOKENIZER_JSON
from beacon_closing_arm import recompute_states
from tokenizers import Tokenizer
tok = Tokenizer.from_file(TOKENIZER_JSON)

def pooled(texts):
    out = []
    for s0 in range(0, len(texts), 8):
        chunk = texts[s0:s0+8]
        ids = np.zeros((8, T_ALG), np.int32); msk = np.zeros((8, T_ALG), np.float32)
        for i, t in enumerate(chunk):
            e = tok.encode(t)
            L = min(len(e.ids), T_ALG)
            ids[i, :L] = e.ids[:L]; msk[i, :L] = 1.0
        s = recompute_states(ids).astype(np.float32)
        v = (s * msk[:, :, None]).sum(1) / np.maximum(msk.sum(1)[:, None], 1)
        out.append(v[:len(chunk)])
    v = np.concatenate(out)
    return v / np.linalg.norm(v, axis=1, keepdims=True)

book = [json.loads(l) for l in open(".cache/book1.jsonl") if not json.loads(l).get("residual")]
raws = pooled([b["raw"] for b in book])
dias = pooled([b["dialect"] for b in book])
sims = [(b["idx"], float(r @ d)) for b, r, d in zip(book, raws, dias)]
sims.sort(key=lambda t: t[1])
print("\n=== (C) SEPARATION: pooled-TRUNK raw<->dialect cosine per book pair ===")
CHRONIC = {45}
MIXED = {45, 51, 54}     # historical mixed/3-vote items
for idx_, s in sims:
    tag = " <-- CHRONIC" if idx_ in CHRONIC else (" (mixed-vote history)" if idx_ in MIXED else "")
    print(f"  [{idx_:2d}] raw<->dialect cos {s:.4f}{tag}")
mu = np.mean([s for _, s in sims]); sd_ = np.std([s for _, s in sims])
s45 = dict(sims).get(45)
if s45 is not None:
    print(f"  panel mean {mu:.4f} +- {sd_:.4f}; [45] z-score {(s45-mu)/max(sd_,1e-9):+.2f}")
