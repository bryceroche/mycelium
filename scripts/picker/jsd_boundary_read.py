"""jsd_boundary_read.py -- THE JSD BOUNDARY READ (2026-10-17; word given; bars pinned in the ledger before running).
Does the Jensen-Shannon divergence between adjacent tokens' trunk attention mark the edges of multi-word chunks (the
Panama-hats case) better than a grammar baseline? Zero GPU: the deployed BP trunk (L0-L3, bidirectional, pad keys
masked) re-run on CPU through llama_loader.forward_return_weights; gated against the banked realBP states.
Fixture: 300 verbatim GSM8K rows of form_mix_real (seed 0), 150 DEV (picks the layer) / 150 TEST (the bar).
Out: .cache/jsd_boundary_read.txt"""
import json, os, sys
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
os.environ.setdefault("DEV", "CPU")
import numpy as np  # noqa: E402
import spacy  # noqa: E402
from tinygrad import Tensor, dtypes  # noqa: E402
from phase1_algebra_head import tokenize, T_ALG, STATES_NPY  # noqa: E402
from mycelium.llama_loader import attach_llama_layers, load_llama_weights, LLAMA_3_2_1B_CFG, _rms_norm  # noqa: E402

N, SPLIT, JSONL = 300, "realBP", ".cache/form_mix_real.jsonl"
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


def jsd(p, q):
    m = 0.5 * (p + q)
    kl = lambda a, b: np.where(a > 0, a * (np.log(np.maximum(a, 1e-30)) - np.log(np.maximum(b, 1e-30))), 0).sum(-1)
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def auc(pos, neg):
    s = np.concatenate([pos, neg]); y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    o = np.argsort(s, kind="mergesort"); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    for v in np.unique(s):  # average ranks on ties
        w = s == v; r[w] = r[w].mean()
    return (r[y == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


rec = {"dev": [], "test": []}
cosg, subw, wordg = [], [], []
for n, i in enumerate(pick):
    m = mask[i]; nt = int(m.sum())
    # pad KEYS are masked to exact zero weight in the BP regime and the real tokens are a prefix, so running the real
    # prefix alone is the same computation for every real token (the gate below checks it against the banked states)
    am = None
    x = host.llama_embed[Tensor(ids[i:i + 1, :nt], dtype=dtypes.int)]
    A = []
    for layer in host.llama_layers:
        x, w = layer.forward_return_weights(x, host.llama_rope_cos, host.llama_rope_sin, attn_mask=am)
        A.append(w.numpy()[0, :, :nt, :nt].astype(np.float64))   # (heads, nt, nt)
    st = _rms_norm(x, host.llama_layers[-1].ffn_norm, host.llama_cfg.rms_norm_eps).cast(dtypes.float).numpy()[0, :nt]
    bk = np.asarray(S_banked[i, :nt], np.float32)
    c = (st * bk).sum(-1) / (np.linalg.norm(st, axis=-1) * np.linalg.norm(bk, axis=-1) + 1e-8); cosg.append(c.mean())
    text = samples[i]["text"]; doc = nlp(text)
    wid = np.full(nt, -1)
    for t in range(nt):
        cs, ce = offsets[i][t]
        for wj, tok in enumerate(doc):
            if tok.idx <= cs < tok.idx + len(tok) or (cs < tok.idx < ce):
                wid[t] = wj; break
    chunk = {}
    for k, ch in enumerate(doc.noun_chunks):
        for tok in ch:
            chunk[tok.i] = (k, len(ch))
    for t in range(nt - 1):
        if wid[t] < 0 or wid[t + 1] < 0:
            continue
        sc = [jsd(A[l][:, t], A[l][:, t + 1]).mean() for l in range(4)]
        if wid[t] == wid[t + 1]:
            subw.append(np.mean(sc)); continue
        w1, w2 = doc[int(wid[t])], doc[int(wid[t + 1])]
        if w1.is_punct or w2.is_punct or w2.i != w1.i + 1:
            continue
        c1, c2 = chunk.get(w1.i), chunk.get(w2.i)
        wordg.append(np.mean(sc))
        if c1 and c2 and c1[0] == c2[0] and c1[1] >= 2:
            lab = 0
        elif (c1 and c1[1] >= 2) or (c2 and c2[1] >= 2):
            lab = 1
        else:
            continue
        sdist = 1 - float((bk[t] * bk[t + 1]).sum() / (np.linalg.norm(bk[t]) * np.linalg.norm(bk[t + 1]) + 1e-8))
        rec["dev" if n < N // 2 else "test"].append((lab, *sc, float(np.mean(sc)), float(w1.pos_ != w2.pos_), sdist))
    if n % 25 == 0:
        print(f"  row {n}/{N} gate cos {np.mean(cosg):.5f}", flush=True)
g = float(np.mean(cosg))
say(f"GATE: re-run BP trunk vs banked realBP states, mean cos {g:.5f} (min row {min(cosg):.5f}) -> {'PASS' if g >= 0.999 else 'VOID'}")
names = ["L0", "L1", "L2", "L3", "mean"]
res = {}
for part in ("dev", "test"):
    R = np.array(rec[part]); y = R[:, 0] == 1
    res[part] = {nm: auc(R[y, 1 + k], R[~y, 1 + k]) for k, nm in enumerate(names)}
    res[part]["POS-change"] = auc(R[y, 6], R[~y, 6]); res[part]["state-dist"] = auc(R[y, 7], R[~y, 7])
    say(f"{part.upper()}: gaps {len(R)} (edge {int(y.sum())} / inside {int((~y).sum())}) | " + " | ".join(f"{k} {v:.3f}" for k, v in res[part].items()))
best = max(names, key=lambda k: res["dev"][k])
t_auc, t_pos = res["test"][best], res["test"]["POS-change"]
say(f"LAYER chosen on DEV: {best} (dev {res['dev'][best]:.3f}) -> TEST AUC {t_auc:.3f} vs POS-change {t_pos:.3f} vs state-dist {res['test']['state-dist']:.3f}")
say(f"sanity: mean JSD subword gaps {np.mean(subw):.4f} (n {len(subw)}) vs word gaps {np.mean(wordg):.4f} (n {len(wordg)})")
v = ("VOID" if g < 0.999 else "JSD SEGMENTER LIVE" if (t_auc >= 0.75 and t_auc >= t_pos + 0.05)
     else "DEAD" if (t_auc < 0.65 or t_auc <= t_pos) else "WEAK (between the bars)")
say(f"VERDICT (pinned frame): {v}")
open(".cache/jsd_boundary_read.txt", "w").write("\n".join(OUT) + "\n")
