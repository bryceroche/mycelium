"""brickp_split_bars.py — THE REGISTERED VERDICT on Brick-P (2026-07-10).

THREE VERDICTS, sentences ready (registered before measuring):
  A. COLLISIONS drop, LONE misbindings stand — the clean split: breathing and
     the pointer law each hold jurisdiction; the wall's core renamed
     (internally-consistent misbinding, contestable by nothing in the field).
  B. BOTH drop — breathing exceeds theory; the pointer law gets its first
     boundary condition (NEIGHBORHOOD PRESSURE moves pointers where
     conditioning couldn't — 6th sighting resolving differently).
  C. NEITHER drops (fac carries it) — capacity in a costume; dies honestly.

COUNTERS (per head x per corpus, given-slot errors vs gold):
  SWAP      — mutual value exchange (collision)
  DUP       — two given slots claim the same var (collision)
  LONE-MISB — wrong value that exists elsewhere in gold, NO complementary
              error (the pointer-law population)
  LONE-HALL — wrong value not in gold anywhere
  INVISIBLE — one-shot forced-wrong (solve2 != gold)

RIDER (relay): the parser's FIRST per-breath belief-movement field — per-slot
JSD between breath-0 and breath-1 head distributions vs slot wrongness
(midrank AUC). Portfolio classification on arrival; the customer (lone
misbindings, if they stand) is already waiting.

USAGE: DEV=AMD .venv/bin/python3 scripts/brickp_split_bars.py
"""
from __future__ import annotations

import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

import numpy as np

os.environ["ALG2"] = "1"
from phase1_algebra_head import (  # noqa: E402
    L_FAC, build_params, forward, decode, build_slot_masks,
)
from tta_alg2_dials import solve2  # noqa: E402
from survivor_multiplicity import midrank_auc  # noqa: E402

HEADS = {"incumbent": (".cache/phase1_algebra2_head.safetensors", 1),
         "breath-K2": (".cache/phase1_breath_head.safetensors", 2)}
CORPORA = {"alg2test": (".cache/algebra2_nl_test.jsonl",
                        ".cache/phase1_alg_states_alg2test.npz"),
           "bigtest": (".cache/algebra_nl_bigtest.jsonl",
                       ".cache/phase1_alg_states_bigtest.npz")}


def softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def jsd(p, q):
    m = 0.5 * (p + q)

    def kl(a, b):
        return float((a * (np.log(a + 1e-12) - np.log(b + 1e-12))).sum(-1)
                     .mean())
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def run(ckpt, k_breath, samples, states, tokmask, sent):
    from tinygrad import Tensor, dtypes
    from tinygrad.nn.state import safe_load
    os.environ["ALG_BREATH"] = str(k_breath)
    p = build_params(0)
    sd = safe_load(ckpt)
    assert set(sd.keys()) == set(p.keys()), f"eval load incomplete: {ckpt}"
    for k in p:
        p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
    n = len(samples)
    parses, answers, movement = {}, {}, {}
    for s0 in range(0, n, 8):
        sl = np.arange(s0, min(s0 + 8, n))
        pad = 8 - len(sl)
        sl_p = np.concatenate([sl, sl[:1].repeat(pad)]) if pad else sl
        t_tr = Tensor(states[sl_p].astype(np.float32), dtype=dtypes.float)
        t_tk = Tensor(tokmask[sl_p].astype(np.float32), dtype=dtypes.float)
        t_se = Tensor(sent[sl_p].astype(np.int32), dtype=dtypes.int)
        out = forward(p, t_tr, t_tk, t_se)
        if k_breath > 1:
            o0 = {k: out[k].realize().numpy() for k in ("fat", "args", "res")}
            mk = build_slot_masks(o0, sent[sl_p])
            out = forward(p, t_tr, t_tk, t_se,
                          slot_mask=Tensor(mk, dtype=dtypes.float))
        keys = ("pres", "ftype", "op", "islit", "dig", "args", "res",
                "query") + (("sel",) if "sel" in out else ())
        o = {k: out[k].realize().numpy() for k in keys}
        br = None
        if k_breath > 1 and "breaths" in out:
            br = [{k: b[k].realize().numpy() for k in ("ftype", "args", "dig")}
                  for b in out["breaths"]]
        for bi, i in enumerate(sl):
            i = int(i)
            facs, q = decode({k: o[k][bi] for k in o})
            parses[i] = (facs, q)
            answers[i] = solve2(facs, q, samples[i])
            if br is not None:
                mv = np.zeros(L_FAC)
                for j in range(L_FAC):
                    mv[j] = (jsd(softmax(br[0]["ftype"][bi, j][None]),
                                 softmax(br[1]["ftype"][bi, j][None]))
                             + jsd(softmax(br[0]["args"][bi, j][None]),
                                   softmax(br[1]["args"][bi, j][None]))
                             + jsd(softmax(br[0]["dig"][bi, j]),
                                   softmax(br[1]["dig"][bi, j])))
                movement[i] = mv
    return parses, answers, movement


def classify(samples, parses, answers):
    c = {"swap": 0, "dup": 0, "lone_misb": 0, "lone_hall": 0, "invisible": 0,
         "wrong_givens": 0}
    for i, smp in enumerate(samples):
        facs, q = parses[i]
        gold_g = {f["var"]: f["value"] for f in smp["factors"]
                  if f["ftype"] == "given"}
        gold_vals = set(gold_g.values())
        seen, em = {}, {}
        for f in facs:
            if f["ftype"] != "given":
                continue
            if f["var"] in seen:
                c["dup"] += 1
            seen[f["var"]] = True
            em.setdefault(f["var"], f["value"])
        wrongs = {v: em[v] for v in em
                  if v in gold_g and em[v] != gold_g[v]}
        c["wrong_givens"] += len(wrongs)
        in_swap = set()
        wl = list(wrongs)
        for x in range(len(wl)):
            for y in range(x + 1, len(wl)):
                u, v = wl[x], wl[y]
                if wrongs[u] == gold_g[v] and wrongs[v] == gold_g[u]:
                    c["swap"] += 1
                    in_swap |= {u, v}
        for v in wl:
            if v in in_swap:
                continue
            if wrongs[v] in gold_vals:
                c["lone_misb"] += 1
            else:
                c["lone_hall"] += 1
        gold_ans = smp["solution"][smp["query_var"]]
        if answers[i] is not None and answers[i] != gold_ans:
            c["invisible"] += 1
    return c


def main():
    for cname, (jpath, spath) in CORPORA.items():
        samples = [json.loads(l) for l in open(jpath)]
        z = np.load(spath)
        states, tokmask, sent = z["states"], z["tokmask"], z["sent"]
        gold = {k[2:]: z[k] for k in z.files if k.startswith("g_")}
        print(f"\n=== {cname} (n={len(samples)}) ===")
        print(f"  head       | wrongG | SWAP | DUP | LONE-MISB | LONE-HALL |"
              f" INVISIBLE")
        mv_store = None
        for hname, (ckpt, kb) in HEADS.items():
            parses, answers, movement = run(ckpt, kb, samples, states,
                                            tokmask, sent)
            c = classify(samples, parses, answers)
            print(f"  {hname:10s} | {c['wrong_givens']:5d}  | {c['swap']:4d} |"
                  f" {c['dup']:3d} | {c['lone_misb']:8d}  |"
                  f" {c['lone_hall']:8d}  | {c['invisible']:5d}")
            if movement:
                mv_store = (movement, parses)
        # RIDER: belief-movement vs slot wrongness (breath head only).
        # Slot alignment is order-approximate (decode's j-th present factor
        # vs span-sorted gold's j-th) — noted, symmetric noise.
        if mv_store:
            from tta_alg2_dials import gkey
            movement, parses = mv_store
            mw, mr = [], []
            for i, smp in enumerate(samples):
                gf = sorted(smp["factors"],
                            key=lambda f: min(s for s, _ in f["spans"]))
                facs = parses[i][0]
                for j in range(min(len(gf), L_FAC)):
                    wrong = (j >= len(facs)
                             or gkey(facs[j]) != gkey(gf[j]))
                    (mw if wrong else mr).append(movement[i][j])
            auc = midrank_auc(np.array(mw), np.array(mr))
            print(f"  RIDER — belief-movement vs slot-wrongness AUC "
                  f"(midrank): {auc:.3f}  (n_wrong={len(mw)}, "
                  f"n_right={len(mr)})")
    print(f"\n  VERDICTS: A) collisions drop, lone stands = clean split;"
          f" B) both drop = pointer-law boundary condition (6th sighting"
          f" differs); C) neither = capacity, dies honestly.")


if __name__ == "__main__":
    main()
