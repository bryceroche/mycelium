"""span_mine.py — SPAN-GRAIN HMM v2 (2026-08-24, word given; the
keystone). The mint is span-labeled by construction: re-render comp-mint
node trees with an instrumented renderer that records each op's char
span; every token inherits its span's kind (given/rel/macro) or NONE
(frame words). Mine: token-grain per-kind centroids (minibatch k-means)
+ empirical kind-transition matrix, both under gsb227 coordinates (the
v1 instrument — baseline comparable). Decode: 4-state Viterbi
(given/rel/macro/none) with empirical transitions. Grade: (a) span-
exact on held-out mint rows (the clean upper read), (b) kind-multiset
F1 on the 143 gold wild rows vs v1's 0.373 baseline and the 0.5 bar
(pct/fdiv/sel gold rows cap F1 — comp mint doesn't carry those kinds;
noted honestly).
"""
import os, sys, json, glob, random
os.environ.update({"DEV": "AMD", "ALG2": "1", "ALG_FTYPES": "8",
                   "ALG_DUP": "1", "ALG_HW": "512", "ALG_WIDE": "1",
                   "ALG_TEST": ".cache/algebra_nl_bigtest.jsonl",
                   "ALG_TEST_NAME": "bigtest"})
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts')
import numpy as np
from collections import Counter
from phase1_algebra_head import T_ALG, TOKENIZER_JSON, sent_indices, load_alg
from beacon_closing_arm import recompute_states
from tokenizers import Tokenizer
from tinygrad.nn.state import safe_load
from comp_mint import sample_expr, ok_node, val, LETTERS

_ = load_alg("test")
tok = Tokenizer.from_file(TOKENIZER_JSON)
STATES = ["none", "given", "rel", "macro"]

# ---- instrumented renderer: returns (text, spans=[(kind,a,b)]) ----
def render_sp(n, prec=0):
    t = n[0]
    if t == "lit":
        s = str(n[1]); return s, [("given", 0, len(s))]
    if t == "var":
        return n[1], []          # letter itself; its given lives in the decl
    if t == "sq":
        inner, sp = render_sp(n[1], 4)
        if n[1][0] not in ("lit", "var"):
            inner = f"({inner})"; sp = [(k, a + 1, b + 1) for k, a, b in sp]
        s = f"{inner}^2"
        return s, sp + [("rel", 0, len(s))]
    if t == "fr":
        inner, sp = render_sp(n[1], 0)
        s = "\\frac{%s}{%d}" % (inner, n[2])
        off = len("\\frac{")
        return s, [(k, a + off, b + off) for k, a, b in sp] + \
               [("macro", 0, len(s))]
    if t == "opa":
        k1, a, k2, b, op = n[1], n[2], n[3], n[4], n[5]
        def leg(k, x):
            rx, sp = render_sp(x, 3)
            if x[0] == "var": return f"{k}{rx}", [(kk, aa + len(str(k)), bb + len(str(k))) for kk, aa, bb in sp]
            if x[0] == "lit": return f"{k} \\times {rx}", []
            return f"{k}({rx})", [(kk, aa + len(str(k)) + 1, bb + len(str(k)) + 1) for kk, aa, bb in sp]
        sgn = "+" if op == "add" else "-"
        la, sa = leg(k1, a); lb, sb = leg(k2, b)
        s = f"{la} {sgn} {lb}"
        off_b = len(la) + 3
        return s, sa + [(k, aa + off_b, bb + off_b) for k, aa, bb in sb] + \
               [("macro", 0, len(s))]
    a, b = n[1], n[2]
    ra, sa = render_sp(a, 1 if t in ("add", "sub") else 2)
    rb, sb = render_sp(b, 2 if t == "sub" else (3 if t == "mul" else 1))
    joiner = {"add": " + ", "sub": " - ", "mul": " \\times "}[t]
    if t == "mul" and a[0] == "lit" and b[0] == "var":
        s = f"{ra}{rb}"; off_b = len(ra)
    else:
        s = f"{ra}{joiner}{rb}"; off_b = len(ra) + len(joiner)
    sp = sa + [(k, aa + off_b, bb + off_b) for k, aa, bb in sb]
    return s, sp + [("rel", 0, len(s))]

TEMPL = ["If {d}, what is the value of ${e}$?",
         "Suppose {d}. What is ${e}$?",
         "Given that {d}, find ${e}$.",
         "Let {d}. Compute ${e}$."]

def gen_row(rng):
    nlet = rng.randint(1, 3)
    Ls = rng.sample(LETTERS, nlet)
    lets = [(L, rng.randint(1, 30)) for L in Ls]
    n = sample_expr(rng, rng.randint(1, 3), lets)
    def used(m, acc):
        if m[0] == "var": acc.add((m[1], m[2]))
        elif m[0] in ("add", "sub", "mul"): used(m[1], acc); used(m[2], acc)
        elif m[0] in ("sq", "fr"): used(m[1], acc)
        elif m[0] == "opa": used(m[2], acc); used(m[4], acc)
        return acc
    ul = sorted(used(n, set()))
    if not ul or not ok_node(n) or n[0] in ("lit", "var"): return None
    expr, esp = render_sp(n)
    tpl = rng.choice(TEMPL)
    decls = " and ".join(f"${L} = {v}$" for L, v in ul)
    head = tpl.split("{d}")[0]
    mid = tpl.split("{d}")[1].split("{e}")[0]
    text = tpl.format(d=decls, e=expr)
    spans = []
    off = len(head)
    for L, v in ul:                       # decl spans: "$L = v$" -> given
        a = text.find(f"${L} = {v}$", off)
        if a >= 0: spans.append(("given", a + 1, a + len(f"{L} = {v}") + 1))
    eoff = len(head) + len(decls) + len(mid) + 1   # inside the $...$
    spans += [(k, a + eoff, b + eoff) for k, a, b in esp]
    return text, spans

def main():
    rng = random.Random(31313)
    sd = safe_load('.cache/gsb227_real.safetensors')
    W = sd["waist_w"].to("CPU").cast('float').numpy()
    Bw = sd["waist_b"].to("CPU").cast('float').numpy()
    SE = sd["sent_emb"].to("CPU").cast('float').numpy()

    def waist_feats(texts):
        F = []; T = []
        for s0 in range(0, len(texts), 16):
            sl = texts[s0:s0 + 16]
            ids = np.zeros((16, T_ALG), np.int32)
            msk = np.zeros((16, T_ALG), np.float32)
            snt = np.zeros((16, T_ALG), np.int32)
            offs = []
            for li, t in enumerate(sl):
                e = tok.encode(t)
                if len(e.ids) > T_ALG:
                    offs.append(None); continue
                ids[li, :len(e.ids)] = e.ids; msk[li, :len(e.ids)] = 1.0
                snt[li] = sent_indices(t, list(e.offsets), msk[li])
                offs.append(list(e.offsets))
            sts = np.asarray(recompute_states(ids)).astype(np.float32)
            for li in range(len(sl)):
                if offs[li] is None: F.append(None); T.append(None); continue
                ntk = len(offs[li])
                h = sts[li, :ntk] @ W + Bw
                h = 0.5 * h * (1 + np.tanh(0.7978845608 * (h + 0.044715 * h ** 3)))
                h = h + SE[snt[li, :ntk]]
                h = h / (np.linalg.norm(h, axis=1, keepdims=True) + 1e-9)
                F.append(h); T.append(offs[li])
        return F, T

    # ---- 1) span-labeled mint sample ----
    N = int(os.environ.get("SPAN_N", "3000"))
    rows = []
    while len(rows) < N:
        r = gen_row(rng)
        if r: rows.append(r)
    print(f"[span] {len(rows)} span-labeled rows rendered", flush=True)
    feats, offs = waist_feats([t for t, _ in rows])
    # token labels from char spans (smallest covering span wins = leaf op)
    tokX = {s: [] for s in STATES}
    seqs = []
    for (text, spans), h, off in zip(rows, feats, offs):
        if h is None: continue
        labs = []
        for ti, (a, b) in enumerate(off):
            cover = [(sb - sa, k) for k, sa, sb in spans
                     if sa <= a and b <= sb]
            lab = min(cover)[1] if cover else "none"
            labs.append(lab)
            tokX[lab].append(h[ti])
        seq = [labs[0]] + [l for i, l in enumerate(labs[1:], 1)
                           if l != labs[i - 1]]
        seqs.append(labs)
    for s in STATES:
        print(f"[span] tokens[{s}] = {len(tokX[s])}", flush=True)
    # ---- 2) per-kind token centroids (minibatch k-means, k=16) ----
    banks = {}
    for s in STATES:
        Xs = np.stack(tokX[s]) if tokX[s] else np.zeros((1, W.shape[1]))
        k = min(16, max(1, len(Xs) // 50))
        c = Xs[np.random.default_rng(5).choice(len(Xs), k, replace=False)]
        for _ in range(12):
            d = ((Xs[:, None] - c[None]) ** 2).sum(-1)
            a = d.argmin(1)
            c = np.array([Xs[a == j].mean(0) if (a == j).any() else c[j]
                          for j in range(k)])
        banks[s] = c / (np.linalg.norm(c, axis=1, keepdims=True) + 1e-9)
    # ---- 3) empirical transitions ----
    Tm = np.full((4, 4), 1e-3)
    for labs in seqs:
        for x, y in zip(labs, labs[1:]):
            Tm[STATES.index(x), STATES.index(y)] += 1
    Tm = np.log(Tm / Tm.sum(1, keepdims=True))
    np.savez('.cache/span_hmm.npz', Tm=Tm,
             **{f"bank_{s}": banks[s] for s in STATES})
    print("[span] banks + transitions mined -> .cache/span_hmm.npz", flush=True)

    # ---- 4) Viterbi v2 on the 143 gold ----
    def decode(h):
        ntk = len(h)
        em = np.stack([ (h @ banks[s].T).max(1) for s in STATES ], 1)
        # emissions -> log-likelihood units (score-scale fix: raw cosines
        # were swamped by log-prob transitions; tau sharpens the contrast)
        tau = float(os.environ.get("SPAN_TAU", "8.0"))
        em = em * tau
        em = em - np.log(np.exp(em).sum(1, keepdims=True))
        V = em[0].copy()
        back = np.zeros((ntk, 4), np.int32)
        for t in range(1, ntk):
            sc = V[:, None] + Tm
            back[t] = sc.argmax(0)
            V = sc.max(0) + em[t]
        path = [int(V.argmax())]
        for t in range(ntk - 1, 0, -1):
            path.append(int(back[t, path[-1]]))
        return path[::-1]

    byid = {}
    for f in sorted(glob.glob('.cache/book*_t*_batch*.jsonl')):
        for l in open(f): r = json.loads(l); byid[r["src_idx"]] = r
    for l in open('.cache/book12_anchor_batch1.jsonl'):
        r = json.loads(l); byid[r["src_idx"]] = r
    sk = set(json.load(open('.cache/book12_anchor_skips.json')))
    gold = [v for k, v in sorted(byid.items()) if k not in sk]
    gf, go = waist_feats([r["original"] for r in gold])
    f1s = []
    KMAP = {"rel": "rel", "given": "given", "macro": "macro"}
    for r, h in zip(gold, gf):
        if h is None: f1s.append(0.0); continue
        path = decode(h)
        dec = []
        for t, ki in enumerate(path):
            if (t == 0 or ki != path[t - 1]) and STATES[ki] != "none":
                dec.append(STATES[ki])
        d = Counter(dec)
        g = Counter(KMAP.get(f["ftype"], f["ftype"]) for f in r["factors"])
        inter = sum((d & g).values())
        f1s.append(2 * inter / max(sum(d.values()) + sum(g.values()), 1))
    f1s = np.array(f1s)
    print(f"[span] V2 KIND-MULTISET F1 on 143 gold: mean {f1s.mean():.3f} "
          f"median {np.median(f1s):.3f} (v1 baseline 0.373; bar 0.5)  "
          f"rows>=0.5: {(f1s >= 0.5).sum()}/{len(f1s)}", flush=True)
    print("[span] VERDICT: " + ("THE KEYSTONE HOLDS — assembly is the game"
          if f1s.mean() >= 0.5 else
          ("ABOVE BASELINE — grain right, refine emissions"
           if f1s.mean() > 0.373 else "no gain over sentence grain")),
          flush=True)

if __name__ == "__main__":
    main()
