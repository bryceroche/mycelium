"""impulse_boundary_read.py -- R3 THE GEOLOGIST'S IMPULSE (2026-10-18; bars pinned in the ledger 11:25 before running).
Bryce's geophones: kick one token, listen everywhere. The impulse = a fixed-seed random direction of 0.1 x |e_t| added to
token t's input embedding (t = the left token of a word gap); the geophones = the relative state change
r_l(u) = |dh_l(u)| / |h_l(u)| at every position u after each trunk layer l. Coupling across the gap = r_l(t+1) / median
over u not in {0, t} of r_l(u); EDGE score = -coupling (a unit moves as one block; an edge should damp the wave).
Fixture, labels, baseline and gate are the JSD read's exactly (scripts/picker/jsd_boundary_read.py): 300 verbatim GSM8K
rows of form_mix_real (seed 0), 150 DEV (picks the layer) / 150 TEST (the bar); noun-chunk EDGE vs INSIDE word gaps;
POS-change baseline; the CPU real-prefix BP trunk gated against the banked realBP states. The JSD read's BOS fix does
not apply here (gaps touching BOS were never labeled). Out: .cache/impulse_boundary_read.txt"""
import os, sys
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
os.environ.setdefault("DEV", "CPU")
import numpy as np  # noqa: E402
import spacy  # noqa: E402
from tinygrad import Tensor, dtypes  # noqa: E402
from phase1_algebra_head import tokenize, STATES_NPY  # noqa: E402
from mycelium.llama_loader import attach_llama_layers, load_llama_weights, LLAMA_3_2_1B_CFG, _rms_norm  # noqa: E402

N, SPLIT, JSONL, EPS = 300, "realBP", ".cache/form_mix_real.jsonl", 0.1
OUT = []


def say(s=""):
    print(s, flush=True); OUT.append(s)


samples, ids, mask, offsets = tokenize(JSONL)
pick = [i for i, r in enumerate(samples) if (r.get("gen") or {}).get("src") == "gsm8k" and not r["gen"].get("resample")]
pick = sorted(np.random.default_rng(0).choice(pick, N, replace=False).tolist())
S_banked = np.load(STATES_NPY.format(split=SPLIT), mmap_mode="r")


class _H: pass


host = _H()
sd = load_llama_weights(".cache/llama-3.2-1b-weights/model.safetensors")
attach_llama_layers(host, n_layers=4, sd=sd, cfg=LLAMA_3_2_1B_CFG); del sd
nlp = spacy.load("en_core_web_sm")
rng = np.random.default_rng(1)
DIR = rng.standard_normal(host.llama_embed.shape[1]).astype(np.float32); DIR /= np.linalg.norm(DIR)


def auc(pos, neg):
    s = np.concatenate([pos, neg]); y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    o = np.argsort(s, kind="mergesort"); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    for v in np.unique(s):
        w = s == v; r[w] = r[w].mean()
    return (r[y == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def gaps_of(i, nt):
    """the JSD read's labeled word gaps: (t, label, pos_change) with label 1 = EDGE of a >= 2-word noun chunk."""
    text = samples[i]["text"]; doc = nlp(text)
    wid = np.full(nt, -1)
    for t in range(1, nt):  # BOS (t=0) never belongs to a word
        cs, ce = offsets[i][t]
        for wj, tok in enumerate(doc):
            if tok.idx <= cs < tok.idx + len(tok) or (cs < tok.idx < ce):
                wid[t] = wj; break
    chunk = {}
    for k, ch in enumerate(doc.noun_chunks):
        for tok in ch:
            chunk[tok.i] = (k, len(ch))
    out = []
    for t in range(1, nt - 1):
        if wid[t] < 0 or wid[t + 1] < 0 or wid[t] == wid[t + 1]:
            continue
        w1, w2 = doc[int(wid[t])], doc[int(wid[t + 1])]
        if w1.is_punct or w2.is_punct or w2.i != w1.i + 1:
            continue
        c1, c2 = chunk.get(w1.i), chunk.get(w2.i)
        if c1 and c2 and c1[0] == c2[0] and c1[1] >= 2:
            lab = 0
        elif (c1 and c1[1] >= 2) or (c2 and c2[1] >= 2):
            lab = 1
        else:
            continue
        out.append((t, lab, float(w1.pos_ != w2.pos_)))
    return out


rec = {"dev": [], "test": []}
cosg = []
for n, i in enumerate(pick):
    nt = int(mask[i].sum())
    G = gaps_of(i, nt)
    ts = sorted({g[0] for g in G})
    e = host.llama_embed[Tensor(ids[i:i + 1, :nt], dtype=dtypes.int)].cast(dtypes.float).numpy()  # (1, nt, H)
    X = np.repeat(e, 1 + len(ts), 0)
    for b, t in enumerate(ts):
        X[1 + b, t] += EPS * np.linalg.norm(e[0, t]) * DIR
    x = Tensor(X).cast(host.llama_embed.dtype)
    R = []
    for layer in host.llama_layers:
        x = layer(x, host.llama_rope_cos, host.llama_rope_sin, attn_mask=None)
        h = x.cast(dtypes.float).numpy()
        R.append(np.linalg.norm(h[1:] - h[:1], axis=-1) / (np.linalg.norm(h[:1], axis=-1) + 1e-8))  # (len(ts), nt)
    st = _rms_norm(x[:1], host.llama_layers[-1].ffn_norm, host.llama_cfg.rms_norm_eps).cast(dtypes.float).numpy()[0]
    bk = np.asarray(S_banked[i, :nt], np.float32)
    cosg.append(((st * bk).sum(-1) / (np.linalg.norm(st, axis=-1) * np.linalg.norm(bk, axis=-1) + 1e-8)).mean())
    bi = {t: b for b, t in enumerate(ts)}
    for t, lab, pc in G:
        b = bi[t]; sc = []
        for l in range(4):
            r = R[l][b]; others = np.delete(r, [0, t])
            sc.append(-float(r[t + 1] / (np.median(others) + 1e-12)))
        rec["dev" if n < N // 2 else "test"].append((lab, *sc, float(np.mean(sc)), pc))
    if n % 25 == 0:
        print(f"  row {n}/{N} gaps {len(G)} gate cos {np.mean(cosg):.5f}", flush=True)
g = float(np.mean(cosg))
say(f"GATE: unperturbed CPU BP trunk vs banked realBP states, mean cos {g:.5f} (min row {min(cosg):.5f}) -> {'PASS' if g >= 0.999 else 'VOID'}")
names = ["L0", "L1", "L2", "L3", "mean"]
res = {}
for part in ("dev", "test"):
    A = np.array(rec[part]); y = A[:, 0] == 1
    res[part] = {nm: auc(A[y, 1 + k], A[~y, 1 + k]) for k, nm in enumerate(names)}
    res[part]["POS-change"] = auc(A[y, 6], A[~y, 6])
    say(f"{part.upper()}: gaps {len(A)} (edge {int(y.sum())} / inside {int((~y).sum())}) | " + " | ".join(f"{k} {v:.3f}" for k, v in res[part].items()))
best = max(names, key=lambda k: res["dev"][k])
t_auc, t_pos = res["test"][best], res["test"]["POS-change"]
say(f"LAYER chosen on DEV: {best} (dev {res['dev'][best]:.3f}) -> TEST AUC {t_auc:.3f} vs POS-change {t_pos:.3f}")
v = ("VOID" if g < 0.999 else "IMPULSE SEGMENTER LIVE" if (t_auc >= 0.75 and t_auc >= t_pos + 0.05)
     else "DEAD" if (t_auc < 0.65 or t_auc <= t_pos) else "WEAK (between the bars)")
say(f"VERDICT (pinned frame): {v}")
open(".cache/impulse_boundary_read.txt", "w").write("\n".join(OUT) + "\n")
