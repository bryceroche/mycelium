"""survivor_depth_probe.py — WHICH WALL: routing vs depth vs content (spec
registration, 2026-07-08 night — the relay's pre-build discriminator, sharpened
to three arms).

THE QUESTION: the 396 encode-side survivors fail even with oracle flags. Three
walls could produce that, and they dictate DIFFERENT builds:
  ROUTING — the gold value IS decodable from the current L4 states at its gold
    span; the trained head deterministically mis-routes (the §6 attention-
    bootstrap ghost: pointer pathways don't move without direct supervision).
    -> marker-v0 works as an attention BEACON; deeper prefix retired.
  DEPTH — not decodable at L4, decodable at L8: the information entered but
    shallow encoding buried it. -> deeper prefix is the cheap lever; re-render
    overkill.
  CONTENT — decodable at neither: never written into the states. -> only
    re-encoding (second-view re-render) can fix it.

PROTOCOL: a fresh probe (2048 -> 512 -> N_DIG x 10 MLP, identical arch/steps
per depth) trained to read the GIVEN VALUE from mean-pooled gold-span trunk
states. Train on givens from samples OUTSIDE the 703-failure set; evaluate
digit-exact on: (a) held-out clean-sample givens (instrument baseline),
(b) wrong-parsed givens on recovered-population samples, (c) wrong-parsed
givens on the 396 (the target).

REGISTERED BARS (pinned before measuring):
  INSTRUMENT: baseline (a) must exceed 0.70 at L4, else the probe is broken —
    no verdict is read.
  ROUTING: group (c) at L4 within 10 pts of baseline (a).
  DEPTH:   (c) at L4 >=20 pts below (a) AND L8 closes >=50% of that gap.
  CONTENT: (c) at L4 >=20 pts below (a) AND L8 closes <50% of the gap.
  (10-20-pt middle ground -> mixed; report honestly, partition by kind.)

Requires .cache/oracle_recovered_bigtest.npz (run survivor_oracle_ceiling.py
first) + the survivor profile npz. L8 states cached to
.cache/algebra_bigtest_L8_states.npy on first run.

USAGE: DEV=AMD ALG_TEST=.cache/algebra_nl_bigtest.jsonl ALG_TEST_NAME=bigtest \
           .venv/bin/python3 scripts/survivor_depth_probe.py
"""
from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

import numpy as np

from phase1_algebra_head import (  # noqa: E402
    T_ALG, H_TRUNK, N_DIG, ALG_TEST, build_params, forward, load_alg, decode,
    tokenize, ALG_CKPT, _spans_to_tokmask,
)

L8_NPY = ".cache/algebra_bigtest_L8_states.npy"
N_TRUNK_DEEP = 8


def compute_l8(ids):
    from tinygrad import Tensor, dtypes
    from mycelium.llama_loader import (
        attach_llama_layers, load_llama_weights, LLAMA_3_2_1B_CFG, _rms_norm)

    class _H:
        pass
    host = _H()
    sd = load_llama_weights(os.path.join(
        _ROOT, ".cache/llama-3.2-1b-weights/model.safetensors"))
    attach_llama_layers(host, n_layers=N_TRUNK_DEEP, sd=sd, cfg=LLAMA_3_2_1B_CFG)
    del sd
    n = len(ids)
    states = np.zeros((n, T_ALG, H_TRUNK), np.float16)
    for s0 in range(0, n, 8):
        sl = slice(s0, min(s0 + 8, n))
        x = host.llama_embed[Tensor(ids[sl], dtype=dtypes.int)]
        for layer in host.llama_layers:
            x = layer(x, host.llama_rope_cos, host.llama_rope_sin)
        x = _rms_norm(x, host.llama_layers[-1].ffn_norm,
                      host.llama_cfg.rms_norm_eps)
        c = x.cast(dtypes.float).realize().numpy()
        assert np.isfinite(c).all()
        states[sl] = c.astype(np.float16)
    np.save(L8_NPY, states)
    print(f"[depth] L8 states computed + cached {states.shape}")
    return states


def train_probe(Xtr, Ytr, seed):
    """Identical probe per depth: 2048 -> 512 -> N_DIG*10, Adam, fixed steps."""
    from tinygrad import Tensor, dtypes
    from tinygrad.nn.optim import Adam
    Tensor.training = True
    rng = np.random.RandomState(seed)

    def t(a):
        x = Tensor(a.astype(np.float32), dtype=dtypes.float,
                   requires_grad=True).contiguous().realize()
        x.requires_grad = True
        return x

    p = {"w1": t(rng.randn(H_TRUNK, 512) * 0.02), "b1": t(np.zeros(512)),
         "w2": t(rng.randn(512, N_DIG * 10) * 0.02),
         "b2": t(np.zeros(N_DIG * 10))}
    opt = Adam(list(p.values()), lr=1e-3)

    def fwd(xb):
        h = (xb @ p["w1"] + p["b1"]).gelu()
        return (h @ p["w2"] + p["b2"]).reshape(-1, N_DIG, 10)

    n = len(Xtr)
    from tinygrad import Tensor as T_
    for step in range(1500):
        sl = rng.randint(0, n, 256)
        xb = T_(Xtr[sl], dtype=dtypes.float)
        yb = T_(Ytr[sl].astype(np.int32), dtype=dtypes.int)
        logits = fwd(xb)
        loss = logits.sparse_categorical_crossentropy(yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return p, fwd


def eval_probe(fwd, X, Y):
    from tinygrad import Tensor, dtypes
    Tensor.training = False
    if not len(X):
        return float("nan")
    preds = []
    for s0 in range(0, len(X), 512):
        lg = fwd(Tensor(X[s0:s0 + 512], dtype=dtypes.float))
        preds.append(lg.realize().numpy().argmax(-1))
    pred = np.concatenate(preds)
    return float((pred == Y).all(axis=1).mean())


def main():
    from tinygrad import Tensor, dtypes
    from tinygrad.nn.state import safe_load

    prof = np.load(".cache/survivor_profile_bigtest.npz")
    failure_set = set(int(i) for i in prof["idx"])
    surv460 = set(int(i) for i, s in zip(prof["idx"], prof["status"]) if s == 2)
    orc = np.load(".cache/oracle_recovered_bigtest.npz")
    hard396 = surv460 - set(int(i) for i in orc["recovered"])

    samples, states4, tokmask, gold, sent = load_alg("test")
    _, ids, _, offsets = tokenize(ALG_TEST)
    states8 = (np.load(L8_NPY) if os.path.exists(L8_NPY) else compute_l8(ids))

    # blank parse -> which gold givens were parsed wrong
    p = build_params(0)
    sd = safe_load(ALG_CKPT)
    for k in p:
        p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
    n = len(samples)
    emitted = {}
    for s0 in range(0, n, 8):
        sl = np.arange(s0, min(s0 + 8, n))
        pad = 8 - len(sl)
        sl_p = np.concatenate([sl, sl[:1].repeat(pad)]) if pad else sl
        out = forward(p, Tensor(states4[sl_p].astype(np.float32),
                                dtype=dtypes.float),
                      Tensor(tokmask[sl_p].astype(np.float32),
                             dtype=dtypes.float),
                      Tensor(sent[sl_p].astype(np.int32), dtype=dtypes.int))
        o = {k: out[k].realize().numpy() for k in
             ("pres", "ftype", "op", "islit", "dig", "args", "res", "query")}
        for bi, i in enumerate(sl):
            facs, _ = decode({k: o[k][bi] for k in o})
            eg = {}
            for f in facs:
                if f["ftype"] == "given" and f["var"] not in eg:
                    eg[f["var"]] = f["value"]
            emitted[int(i)] = eg

    # collect (pooled4, pooled8, digits, group) per gold given
    def digits_of(v):
        return [(v // 10 ** (N_DIG - 1 - d)) % 10 for d in range(N_DIG)]

    rows = {"train": [], "base": [], "wrong_rec": [], "wrong_hard": []}
    rng = np.random.RandomState(0)
    clean = sorted(set(range(n)) - failure_set)
    heldout = set(rng.choice(clean, size=len(clean) // 10, replace=False)
                  .tolist())
    for i in range(n):
        for f in samples[i]["factors"]:
            if f["ftype"] != "given":
                continue
            m = np.zeros((T_ALG,), np.float32)
            _spans_to_tokmask(f["spans"], offsets[i], m)
            if m.sum() == 0:
                continue
            m /= m.sum()
            v4 = (states4[i].astype(np.float32) * m[:, None]).sum(0)
            v8 = (states8[i].astype(np.float32) * m[:, None]).sum(0)
            wrong = emitted[i].get(f["var"]) != f["value"]
            if i in hard396 and wrong:
                grp = "wrong_hard"
            elif i in failure_set and wrong:
                grp = "wrong_rec"
            elif i not in failure_set:
                grp = "base" if i in heldout else "train"
            else:
                continue  # correctly-parsed givens on failure samples: unused
            rows[grp].append((v4, v8, digits_of(int(f["value"]))))

    print(f"[depth] spans: train={len(rows['train'])} base={len(rows['base'])} "
          f"wrong_rec={len(rows['wrong_rec'])} wrong_hard={len(rows['wrong_hard'])}")

    results = {}
    for di, depth in ((0, "L4"), (1, "L8")):
        Xtr = np.stack([r[di] for r in rows["train"]])
        Ytr = np.stack([r[2] for r in rows["train"]])
        mu, sd_ = Xtr.mean(0), Xtr.std(0) + 1e-6
        _, fwd = train_probe((Xtr - mu) / sd_, Ytr, seed=1)
        for grp in ("base", "wrong_rec", "wrong_hard"):
            X = np.stack([r[di] for r in rows[grp]]) if rows[grp] else \
                np.zeros((0, H_TRUNK))
            Y = np.stack([r[2] for r in rows[grp]]) if rows[grp] else \
                np.zeros((0, N_DIG))
            results[(depth, grp)] = eval_probe(
                fwd, (X - mu) / sd_ if len(X) else X, Y)

    print(f"\n  digit-exact value decodability (probe, identical arch/steps)")
    print(f"  group       |   L4    |   L8")
    for grp in ("base", "wrong_rec", "wrong_hard"):
        print(f"  {grp:11s} |  {results[('L4', grp)]:.3f}  |  "
              f"{results[('L8', grp)]:.3f}")

    b4, h4 = results[("L4", "base")], results[("L4", "wrong_hard")]
    h8 = results[("L8", "wrong_hard")]
    print(f"\n  REGISTERED VERDICT (instrument bar: base L4 > 0.70):")
    if b4 <= 0.70:
        print(f"  INSTRUMENT BROKEN (base {b4:.3f}) — no verdict.")
    elif b4 - h4 <= 0.10:
        print(f"  ROUTING wall: gold decodable at L4 on the 396 "
              f"({h4:.3f} vs base {b4:.3f}) — marker-v0 as attention beacon; "
              f"deeper prefix retired.")
    elif b4 - h4 >= 0.20 and (h8 - h4) >= 0.5 * (b4 - h4):
        print(f"  DEPTH wall: L8 closes >=50% of the gap "
              f"({h4:.3f} -> {h8:.3f}, base {b4:.3f}) — deeper prefix is the "
              f"cheap lever.")
    elif b4 - h4 >= 0.20:
        print(f"  CONTENT wall: gap {b4 - h4:.3f} at L4, L8 closes "
              f"{(h8 - h4) / max(b4 - h4, 1e-9):.0%} — only re-encoding can "
              f"fix what was never written; marker-v0 re-render mandated.")
    else:
        print(f"  MIXED (gap {b4 - h4:.3f} in the 10-20pt band) — partition by "
              f"kind before the build call.")


if __name__ == "__main__":
    main()
