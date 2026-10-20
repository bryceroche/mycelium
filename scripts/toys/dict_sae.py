"""dict_sae.py -- THE DICTIONARY READ (2026-10-20; Bryce's map-reduce reverse index = dictionary learning; Opus: D4 + D7).
A TopK sparse autoencoder per tap on .cache/dict_states_D{4,7}.npy (dict_collect.py), then the pinned read: do Llama's
features separate SURFACE categories (numerals, op-cue words) and do they separate ROLES (owner, recipient) once the
word itself is controlled? The binding theorem predicts surface yes, roles no. CPU (torch).

SPLITS by row: 80% train / 10% SELECT / 10% MEASURE. Every 'best feature' is chosen on SELECT and scored on MEASURE (the
winner's curse over 8192 features stays on the select split). BOS tokens excluded (Llama's massive-norm position).
SAE: x' = (x - mu) / s (train mean; s = the mean train norm / sqrt(2048)); z = TopK_k(relu(W_e (x' - b_d) + b_e));
x^ = W_d z + b_d, W_d columns unit-norm after every step. Adam, batch 4096.
GATES (the dictionary is fit to read): MEASURE fraction of variance explained >= 0.70; dead features (never active on
MEASURE) < 20%.
THE READ:
  NUMERAL   best single feature (active > 0) F1 for numeral tokens.
  OP-CUE    per cue word with >= 50 MEASURE instances (each, per, every, times, twice, half, more, less, fewer, total,
            altogether, left, remaining, double, percent): best-feature F1; the MEDIAN over words.
  OWNER / RECIPIENT  lexically controlled: tokens whose id occurs >= 5 times with the role and >= 5 without (on
            SELECT and on MEASURE separately); each feature's activation minus its mean within the token id; AUROC of
            role vs not. PLACEBO: the same pipeline with the role labels shuffled WITHIN each token id.
Usage: .venv/bin/python3 scripts/toys/dict_sae.py --tap 4 [--steps N] [--smoke]  -> .cache/dict_sae_D{tap}{tag}.{txt,json,pt}
"""
import argparse, json, math, os, sys, time
sys.path.insert(0, ".")
import numpy as np
import torch
import torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--tap", type=int, required=True)
ap.add_argument("--n_feat", type=int, default=8192)
ap.add_argument("--k", type=int, default=32)
ap.add_argument("--steps", type=int, default=8000)
ap.add_argument("--lr", type=float, default=4e-4)
ap.add_argument("--smoke", action="store_true")
args = ap.parse_args()
torch.set_num_threads(int(os.environ.get("TOY_THREADS", "16"))); torch.manual_seed(0)
tag = "_smoke" if args.smoke else ""
OUT = f".cache/dict_sae_D{args.tap}{tag}"
assert args.smoke or not os.path.exists(OUT + ".txt"), f"refusing to overwrite {OUT}.txt"
log = open(OUT + ".txt", "w")


def say(s):
    print(s, flush=True); log.write(s + "\n"); log.flush()


from tokenizers import Tokenizer  # noqa: E402
tok = Tokenizer.from_file(".cache/llama-3.2-1b-weights/tokenizer.json")
ix = np.load(".cache/dict_index.npz")
X = np.load(f".cache/dict_states_D{args.tap}.npy", mmap_mode="r")
row, tid = ix["row"], ix["tok"]
keep = tid != tok.token_to_id("<|begin_of_text|>")
n_rows = int(row.max()) + 1
perm = np.random.default_rng(7).permutation(n_rows)
part = np.zeros(n_rows, np.int8); part[perm[int(0.8 * n_rows):int(0.9 * n_rows)]] = 1; part[perm[int(0.9 * n_rows):]] = 2
if args.smoke:
    part[perm[2000:int(0.8 * n_rows)]] = 3                                    # smoke: 2000 train rows only
sp = part[row]
I = {name: np.where(keep & (sp == v))[0] for name, v in (("train", 0), ("select", 1), ("measure", 2))}
say(f"D{args.tap}: tokens train {len(I['train'])} select {len(I['select'])} measure {len(I['measure'])} (BOS excluded)")
Xtr = torch.from_numpy(np.asarray(X[I["train"]], np.float32))
mu = Xtr.mean(0); s = (Xtr - mu).norm(dim=1).mean() / math.sqrt(Xtr.shape[1])
prep = lambda a: (torch.from_numpy(np.asarray(a, np.float32)) - mu) / s
Xtr = (Xtr - mu) / s

D, NF = Xtr.shape[1], args.n_feat
We = torch.nn.Parameter(torch.randn(NF, D) / math.sqrt(D)); be = torch.nn.Parameter(torch.zeros(NF))
Wd = torch.nn.Parameter(We.detach().T.clone()); bd = torch.nn.Parameter(Xtr.median(0).values.clone())
with torch.no_grad(): Wd /= Wd.norm(dim=0, keepdim=True)
opt = torch.optim.Adam([We, be, Wd, bd], lr=args.lr)


def encode(x):
    pre = F.relu((x - bd) @ We.T + be)
    v, i = pre.topk(args.k, -1)
    return torch.zeros_like(pre).scatter_(-1, i, v)


steps = 300 if args.smoke else args.steps; t0 = time.time(); g = torch.Generator().manual_seed(1)
for step in range(steps):
    xb = Xtr[torch.randint(len(Xtr), (4096,), generator=g)]
    loss = ((encode(xb) @ Wd.T + bd - xb) ** 2).sum(-1).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad(): Wd /= Wd.norm(dim=0, keepdim=True)
    if step % 1000 == 0 or step == steps - 1:
        say(f"step {step} loss {loss.item():.2f} (var {xb.var(0).sum().item():.2f})  [{time.time() - t0:.0f}s]")
del Xtr


@torch.no_grad()
def acts(idx):
    out, err, var = [], 0.0, 0.0
    for a in range(0, len(idx), 8192):
        x = prep(X[idx[a:a + 8192]]); z = encode(x)
        err += ((z @ Wd.T + bd - x) ** 2).sum().item(); var += ((x - x.mean(0)) ** 2).sum().item()
        out.append(z.to_sparse())
    return torch.cat(out).to_dense().numpy(), 1 - err / var


Zs, _ = acts(I["select"]); Zm, fve = acts(I["measure"])
dead = float(((Zm > 0).sum(0) == 0).mean())
gate = fve >= 0.70 and dead < 0.20
say(f"GATE: measure FVE {fve:.4f} (>= 0.70), dead {dead:.3f} (< 0.20) -> {'PASS' if gate else 'FAIL (the read is reported, not claimed)'}")
res = {"tap": args.tap, "fve": fve, "dead": dead, "gate": gate}


def f1_best(ys, ym, Zs_, Zm_):
    """Choose the feature with the best F1 on SELECT; return its MEASURE F1."""
    def f1(y, Z):
        a = Z > 0; tp = (a & y[:, None]).sum(0); return 2 * tp / (a.sum(0) + y.sum() + 1e-9)
    j = int(np.argmax(f1(ys, Zs_))); return float(f1(ym, Zm_[:, [j]])[0]), j


tsel, tmea = tid[I["select"]], tid[I["measure"]]
ys, ym = ix["numeral"][I["select"]] > 0, ix["numeral"][I["measure"]] > 0
res["numeral_f1"], j = f1_best(ys, ym, Zs, Zm); say(f"NUMERAL best-feature F1 (measure) {res['numeral_f1']:.3f} (feature {j})")
CUES = ["each", "per", "every", "times", "twice", "half", "more", "less", "fewer", "total", "altogether", "left",
        "remaining", "double", "percent"]
word = lambda ids: np.array([tok.decode([int(t)]).strip().lower() for t in ids])
ws, wm = word(tsel), word(tmea); cue = {}
for w in CUES:
    if (wm == w).sum() >= 50 and (ws == w).sum() >= 20:
        cue[w] = f1_best(ws == w, wm == w, Zs, Zm)[0]
res["cue_f1"] = cue; res["cue_median"] = float(np.median(list(cue.values()))) if cue else float("nan")
say(f"OP-CUE best-feature F1 per word: {json.dumps({k: round(v, 3) for k, v in cue.items()})}; MEDIAN {res['cue_median']:.3f}")


def auroc_cols(y, A):
    """AUROC of each column of A for binary y (ties averaged), vectorized by ranks."""
    from scipy.stats import rankdata
    r = rankdata(A, axis=0)
    n1 = y.sum(); n0 = len(y) - n1
    return (r[y].sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0)


def role_read(name, shuffle=False):
    out = {}
    for split, idx, t, Z in (("select", I["select"], tsel, Zs), ("measure", I["measure"], tmea, Zm)):
        y = ix[name][idx] > 0
        ids, cnt1 = np.unique(t[y], return_counts=True)
        ok = [i for i, c in zip(ids, cnt1) if c >= 5 and (~y & (t == i)).sum() >= 5]
        m = np.isin(t, ok)
        yy, tt, ZZ = y[m].copy(), t[m], Z[m].astype(np.float32)
        if shuffle:
            rng = np.random.default_rng(11)
            for i in ok:
                w = np.where(tt == i)[0]; yy[w] = yy[rng.permutation(w)]
        for i in ok:                                                          # the lexical control
            w = tt == i; ZZ[w] -= ZZ[w].mean(0)
        out[split] = (yy, ZZ, len(ok))
    (ys_, Zs_, ns), (ym_, Zm_, nm) = out["select"], out["measure"]
    live = np.where((Zs_ != 0).any(0))[0]
    raw = auroc_cols(ys_, Zs_[:, live]); jj = int(np.argmax(np.maximum(raw, 1 - raw)))   # a feature may mark by silence
    j = live[jj]; am = float(auroc_cols(ym_, Zm_[:, [j]])[0])
    if raw[jj] < 0.5: am = 1 - am                                             # the orientation is chosen on SELECT
    return am, int(j), int(ym_.sum()), int(len(ym_)), nm


for name in ("owner", "recipient"):
    am, j, npos, n, nw = role_read(name); pl = role_read(name, shuffle=True)[0]
    res[f"{name}_auroc"], res[f"{name}_placebo"] = am, pl
    say(f"{name.upper()} lexically-controlled best-feature AUROC (measure) {am:.3f} (feature {j}; {npos} role tokens of {n}, "
        f"{nw} word types) | PLACEBO (labels shuffled within word) {pl:.3f}")
json.dump(res, open(OUT + ".json", "w"), indent=1)
torch.save({"We": We.detach(), "be": be.detach(), "Wd": Wd.detach(), "bd": bd.detach(), "mu": mu, "s": s}, OUT + ".pt")
say(f"DONE in {time.time() - t0:.0f}s")
