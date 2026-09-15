"""waist_lexicon_probe.py — does the waist (2048 -> 512, linear + GELU) keep
what the trunk carries? The lexicon probe repeated on WAIST states computed
from a checkpoint's waist_w / waist_b (the sentence embedding omitted: it
is per-sentence, orthogonal to value): (1) the ridge numeral -> log-value
map on waist states (held-out R^2, within x2); (2) the number words'
cosine rank of their numeral among 1..100 in waist space. CPU; the diet's
precomputed states. usage: waist_lexicon_probe.py CKPT [N_ROWS]"""
import json, re, sys, numpy as np
from tokenizers import Tokenizer
from tinygrad.nn.state import safe_load
ck = sys.argv[1]; N_ROWS = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
sd = safe_load(ck); Ww = sd["waist_w"].numpy().astype(np.float64); Wb = sd["waist_b"].numpy().astype(np.float64)
def gelu(x): return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))
proj = lambda X: gelu(np.asarray(X, np.float64) @ Ww + Wb)
rng = np.random.default_rng(0)
rows = [json.loads(l) for l in open(".cache/form_mix12_aligned.jsonl")]
S = np.load(".cache/phase1_alg_states_form12al_states.npy", mmap_mode="r"); T = 256
tok = Tokenizer.from_file(".cache/llama-3.2-1b-weights/tokenizer.json")
WORDS = {"dozen": 12, "twice": 2, "double": 2, "pair": 2, "half": 0.5, "quarter": 0.25}
pick = set(rng.choice(len(rows), N_ROWS, replace=False).tolist()); word_rows = [i for i, r in enumerate(rows) if any(re.search(r"\b" + w + r"\b", r["text"].lower()) for w in WORDS)]
num_x, num_y, word_x = [], [], {w: [] for w in WORDS}
for i in sorted(pick | set(word_rows)):
    e = tok.encode(rows[i]["text"]); ids = e.ids[:T]; toks = [tok.decode([t]).strip().lower() for t in ids]; st = None
    for t_i, s in enumerate(toks):
        if re.fullmatch(r"\d{1,3}", s) and i in pick and 1 <= int(s) <= 999:
            st = S[i] if st is None else st; num_x.append(np.asarray(st[t_i], np.float32)); num_y.append(int(s))
        elif s in WORDS:
            st = S[i] if st is None else st; word_x[s].append(np.asarray(st[t_i], np.float32))
X2 = np.stack(num_x); y = np.log10(np.array(num_y, np.float64)); n = len(y); perm = rng.permutation(n); tr, te = perm[: int(0.8 * n)], perm[int(0.8 * n):]
vals = np.array(sorted(set(num_y)))
for space, X in (("TRUNK 2048d", X2.astype(np.float64)), ("WAIST 512d", proj(X2))):
    mu = X[tr].mean(0); Xc = X - mu; lam = 10.0
    w = np.linalg.solve(Xc[tr].T @ Xc[tr] + lam * np.eye(X.shape[1]), Xc[tr].T @ (y[tr] - y[tr].mean())); b = y[tr].mean()
    pred = Xc[te] @ w + b; r2 = 1 - ((pred - y[te]) ** 2).sum() / ((y[te] - y[te].mean()) ** 2).sum()
    near = vals[np.abs(np.log10(vals)[None] - pred[:, None]).argmin(1)]
    by_v = {}
    for xi, v in zip(X, num_y): by_v.setdefault(v, []).append(xi)
    cand = sorted(v for v in by_v if len(by_v[v]) >= 5 and v <= 100); M = np.stack([np.mean(by_v[v], 0) for v in cand]); Mn = (M - mu) / (np.linalg.norm(M - mu, axis=1, keepdims=True) + 1e-9)
    print(f"[waist-probe] {space}: numeral R^2 {r2:.3f}, exact {np.mean(near == np.array(num_y)[te]):.3f}, within x2 {np.mean(np.abs(pred - y[te]) < np.log10(2)):.3f}; cosine candidates {len(cand)}")
    for wd, v in WORDS.items():
        xs = word_x[wd]
        if not xs or v < 1 or round(v) not in cand: continue
        Xw = (np.stack(xs).astype(np.float64) if space.startswith("TRUNK") else proj(np.stack(xs))) - mu
        cos = (Xw / (np.linalg.norm(Xw, axis=1, keepdims=True) + 1e-9)) @ Mn.T; rank = [int((c > c[cand.index(round(v))]).sum()) + 1 for c in cos]
        print(f"    {wd:7s} -> {v}: cosine rank median {np.median(rank):.0f} of {len(cand)}, top-1 {np.mean(np.array(rank) == 1):.3f} (n={len(xs)})")
