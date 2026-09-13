"""THE WAIST PROBE (2026-09-13; Bryce: "is 2048 -> 512 losing what wild needs?"):
a token-level tagging probe — is this token part of a VALUE mention, a VARIABLE
mention, or neither (the dish line's first stage) — fit by ridge (one-vs-rest,
closed form) on half the rows and scored on the other half, from three feature
sets: the raw trunk L3 state (2048), the checkpoint's WAIST (512: gelu(trunk@W+b)
+ sent_emb), and a random 512-d projection of the trunk (the dimensionality-only
control). If waist ~= trunk, the waist is not losing it; if trunk >> waist, the
waist is; if all three are poor on wild and good on mint, the information is
not in L0-L3 for wild at all. Env: family env, WP_CKPT, ALG_TEST(_NAME), WP_N."""
import os, sys, numpy as np
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
from phase1_algebra_head import build_params, load_alg, L_FAC, K_VARS
from tinygrad.nn.state import safe_load
import re
vs, vst, vtk, vg, vse = load_alg(os.environ.get("WP_SPLIT", "test"))
_pat = re.compile(r"\b(Let|Consider|Suppose|Take)\b.*\b[a-z]\s*(,|and|be)")
if os.environ.get("WP_FILTER", "") == "prose_mentions":   # the wild REGISTER with span gold: the books' prose rows
    pool = np.array([i for i, r in enumerate(vs) if r.get("mentions") and not _pat.search((r.get("text") or "")[:200])])
else:
    pool = np.arange(len(vs))
N = min(int(os.environ.get("WP_N", "600")), len(pool)); rng = np.random.RandomState(0); idx = pool[rng.permutation(len(pool))[:N]]
print(f"[waist-probe] split={os.environ.get('WP_SPLIT','test')} filter={os.environ.get('WP_FILTER','none')}: pool {len(pool)} rows")
sd = safe_load(os.environ["WP_CKPT"]); W = sd["waist_w"].numpy().astype(np.float32); b = sd["waist_b"].numpy().astype(np.float32); SE = sd["sent_emb"].numpy().astype(np.float32)
X_tr, X_w, X_r, Y, G = [], [], [], [], []
P = np.random.RandomState(1).randn(2048, 512).astype(np.float32) / np.sqrt(2048)
for r, i in enumerate(idx):
    i = int(i); tk = vtk[i] > 0.5; T = int(tk.sum())
    s = vst[i][:T].astype(np.float32)                              # (T, 2048) trunk L3
    w = np.maximum(s @ W + b, 0) * 0 + (lambda z: 0.5 * z * (1 + np.tanh(0.7978845608 * (z + 0.044715 * z ** 3))))(s @ W + b) + SE[vse[i][:T]]   # gelu(trunk@W+b) + sent_emb
    y = np.zeros(T, np.int64)
    for j in range(L_FAC):
        if vg["presence"][i, j] < 0.5: continue
        y[vg["fspan"][i, j][:T] > 0.5] = 1                         # value / factor mention
    for k in range(K_VARS):
        y[vg["vspan"][i, k][:T] > 0.5] = 2                         # variable mention
    X_tr.append(s); X_w.append(w); X_r.append(s @ P); Y.append(y); G.append(np.full(T, r))
X_tr, X_w, X_r, Y, G = map(np.concatenate, (X_tr, X_w, X_r, Y, G))
half = N // 2; tr = G < half; te = ~tr
def ridge_ovr(Xa, ya, Xb, lam=1.0):
    mu = Xa.mean(0); sdv = Xa.std(0) + 1e-6; A = (Xa - mu) / sdv; B = (Xb - mu) / sdv
    Yoh = np.eye(3)[ya]; Wc = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ (Yoh - Yoh.mean(0)))
    return (B @ Wc).argmax(1)
name = os.path.basename(os.environ["WP_CKPT"]).replace("sharp_", "").replace(".safetensors", "")
print(f"[waist-probe] {name} on {os.environ.get('ALG_TEST_NAME','?')}: N={N} rows, {len(Y)} tokens; tag base rates none/value/variable = {np.bincount(Y, minlength=3) / len(Y)}")
for lab, Xa in (("trunk L3 (2048)", X_tr), ("waist (512)", X_w), ("random proj (512)", X_r)):
    for lam in (1.0, 10.0):
        pred = ridge_ovr(Xa[tr], Y[tr], Xa[te], lam)
        acc = (pred == Y[te]).mean(); f1 = []
        for c in (1, 2):
            tp = ((pred == c) & (Y[te] == c)).sum(); fp = ((pred == c) & (Y[te] != c)).sum(); fn = ((pred != c) & (Y[te] == c)).sum()
            f1.append(2 * tp / max(2 * tp + fp + fn, 1))
        print(f"  {lab:20s} lam={lam:4.0f}: token acc {acc:.3f} | F1 value {f1[0]:.3f} | F1 variable {f1[1]:.3f}")
