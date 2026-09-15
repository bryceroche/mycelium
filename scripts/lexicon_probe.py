"""lexicon_probe.py — THE LEXICON PROBE (2026-09-14, Bryce's question: does
the frozen trunk already carry the VALUE of a number word, or only the
word?). Zero training of the machine; CPU; the precomputed L0-L3 states.
  1. On numeral tokens (single tokens 1..999 in ~PROBE_N rows) fit a ridge
     map state(2048) -> log10(value); report held-out numeral R^2 and the
     nearest-numeral accuracy (the probe works on numerals or it is void).
  2. Apply the SAME map to number-word tokens (dozen, twice, double, pair,
     triple/thrice, half, quarter, hundred): the predicted value vs the
     entry; the entry's rank among the candidate values.
  3. The cosine read: cos(mean state(word), mean state(numeral v)) — the
     rank of the entry's numeral among 1..100 by cosine.
If the entries land (rank ~1), the head can learn them from few rows; if
they land nowhere, the lexicon road must INJECT the value (a token-born
certificate through the bridge)."""
import json, re, sys, numpy as np
from tokenizers import Tokenizer
rng = np.random.default_rng(0)
N_ROWS = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
rows = [json.loads(l) for l in open(".cache/form_mix12_aligned.jsonl")]
S = np.load(".cache/phase1_alg_states_form12al_states.npy", mmap_mode="r")
tok = Tokenizer.from_file(".cache/llama-3.2-1b-weights/tokenizer.json"); T = 256
WORDS = {"dozen": 12, "twice": 2, "double": 2, "doubled": 2, "pair": 2, "triple": 3, "thrice": 3, "tripled": 3, "half": 0.5, "quarter": 0.25, "hundred": 100}
num_x, num_y, word_x = [], [], {w: [] for w in WORDS}
pick = set(rng.choice(len(rows), N_ROWS, replace=False).tolist())
word_rows = [i for i, r in enumerate(rows) if any(re.search(r"\b" + w + r"\b", r["text"].lower()) for w in WORDS)]
for i in sorted(pick | set(word_rows)):
    e = tok.encode(rows[i]["text"]); ids = e.ids[:T]; toks = [tok.decode([t]).strip().lower() for t in ids]
    st = None
    for t_i, s in enumerate(toks):
        if re.fullmatch(r"\d{1,3}", s) and i in pick and 1 <= int(s) <= 999:
            st = S[i] if st is None else st
            num_x.append(np.asarray(st[t_i], np.float32)); num_y.append(int(s))
        elif s in WORDS:
            st = S[i] if st is None else st
            word_x[s].append(np.asarray(st[t_i], np.float32))
X = np.stack(num_x); y = np.log10(np.array(num_y, np.float64)); n = len(y)
print(f"[lexicon-probe] numeral tokens {n} from {N_ROWS} rows; number-word tokens " + ", ".join(f"{w}:{len(v)}" for w, v in word_x.items() if v))
perm = rng.permutation(n); tr, te = perm[: int(0.8 * n)], perm[int(0.8 * n):]
mu = X[tr].mean(0); Xc = X - mu
lam = 10.0; A = Xc[tr].T @ Xc[tr] + lam * np.eye(X.shape[1]); w = np.linalg.solve(A, Xc[tr].T @ (y[tr] - y[tr].mean())); b = y[tr].mean()
pred = Xc[te] @ w + b; r2 = 1 - ((pred - y[te]) ** 2).sum() / ((y[te] - y[te].mean()) ** 2).sum()
vals = np.array(sorted(set(num_y))); near = vals[np.abs(np.log10(vals)[None] - pred[:, None]).argmin(1)]
print(f"[lexicon-probe] the numeral probe: held-out R^2 {r2:.3f}; nearest-numeral exact {np.mean(near == np.array(num_y)[te]):.3f}; within x2 {np.mean(np.abs(pred - y[te]) < np.log10(2)):.3f}  (n_test={len(te)})")
# per-numeral mean states for the cosine read
by_v = {}
for xi, v in zip(X, num_y): by_v.setdefault(v, []).append(xi)
cand = sorted(v for v in by_v if len(by_v[v]) >= 5 and v <= 100); M = np.stack([np.mean(by_v[v], 0) for v in cand]); Mn = (M - mu) / (np.linalg.norm(M - mu, axis=1, keepdims=True) + 1e-9)
print(f"[lexicon-probe] cosine candidates: {len(cand)} numerals with >= 5 tokens (1..100)")
for wd, v in WORDS.items():
    xs = word_x[wd]
    if not xs: continue
    Xw = np.stack(xs) - mu; p = Xw @ w + b; pv = 10 ** p
    pn = vals[np.abs(np.log10(vals)[None] - p[:, None]).argmin(1)]
    cos = ((Xw / (np.linalg.norm(Xw, axis=1, keepdims=True) + 1e-9)) @ Mn.T)     # (n_w, n_cand)
    rank = [int((c > c[cand.index(round(v))]).sum()) + 1 for c in cos] if round(v) in cand and v >= 1 else None
    print(f"  {wd:8s} (entry {v:>5}): n={len(xs):4d}  probe median {np.median(pv):7.2f}  nearest-numeral == entry {np.mean(pn == round(v)) if v >= 1 else float('nan'):.3f}  "
          + (f"cosine rank of {round(v)} among {len(cand)}: median {np.median(rank):.0f}, top-1 {np.mean(np.array(rank) == 1):.3f}" if rank else "(entry < 1: no numeral to rank)"))
