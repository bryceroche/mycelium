"""phase1_brick_a.py — BRICK-A: the zero-LoRA conditioning null, measured.

THE HYPOTHESIS (the last pure plane-ride hypothesis — CLAUDE.md §8.4, §8.7 #1):
same FROZEN trunk, different INPUT conditioning (notebook/NACK) -> appropriately
different parse behavior, with NO per-cycle weight mutation. Design registered in
docs/phase1_skeleton_spec.md §9 BEFORE this build:

  * TRUNK-LEVEL conditioning: a shared encoder maps NACK features -> K_PREFIX prefix
    embeddings prepended to the token embeddings, flowing through the frozen
    bidirectional Llama L0-L3. (Head-level injection = the weaker comparison arm.)
  * ZERO-LORA BOUNDARY: trainable = the emission head + ONE shared conditioning
    encoder (the interface). Nothing varies per cycle except the notebook CONTENT.
  * ORACLE-NACK ARM SEPARATION: conditioning uses gold-derived wrong-SENTENCE flags
    on the plateaued parser's ORGANIC failures. Brick-A tests "can the frozen trunk
    USE a correct NACK"; generating one gold-free is Brick-C's question.
  * THE MEASUREMENT IS A FIELD: blank vs NACK-conditioned pass through the SAME
    graph; does the parse delta (emission-change mass, attention-delta mass)
    CONCENTRATE on flagged slots/sentences? Plus flagged-slot FIX RATE vs a
    SHUFFLED-FLAGS control (same count, innocent sentences).
  * KILL: fix_rate(true) ~= fix_rate(shuffled) => the null FAILS toward the LoRA
    ladder (§8.4). Say so plainly.

MECHANICS: all passes (blank included) run the SAME prefixed graph — blank = zero
NACK features -> encoder -> its baseline prefix — so differentials are well-defined.
Note the prefix shifts text RoPE positions vs the banked unprefixed precompute, so
Brick-A retrains head+encoder jointly (warm-started from the plateaued head) with
LIVE trunk forwards; the frozen trunk itself is byte-untouched.

USAGE:
  CPU selftest:        .venv/bin/python3 scripts/phase1_brick_a.py --selftest
  Prep NACK dataset:   DEV=AMD .venv/bin/python3 scripts/phase1_brick_a.py --prep [--prep-n 8000]
  Train (warm-start):  DEV=AMD STEPS=3000 .venv/bin/python3 scripts/phase1_brick_a.py --train
  The measurement:     DEV=AMD .venv/bin/python3 scripts/phase1_brick_a.py --eval
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

_THIS_FILE = os.path.abspath(__file__)
_ROOT = os.path.dirname(os.path.dirname(_THIS_FILE))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.dirname(_THIS_FILE))

import numpy as np

from phase1_delta_head import (  # noqa: E402  (reused verbatim — the trained object)
    T_WINDOW, H_TRUNK, H_WAIST, L_SLOTS, S_CELLS, N_DIGITS, SENT_MAX,
    NL_TRAIN, NL_TEST, CKPT_PATH, TOKENIZER_JSON,
    build_head_params, head_forward, head_loss, load_split, tokenize_corpus,
    decode_slots, OPS, TYPES,
)

K_PREFIX = 8
COND_DIM = SENT_MAX + 1          # per-sentence flags + global-fail bit
BRICK_A_CKPT = ".cache/phase1_brick_a.safetensors"
NACK_NPZ = ".cache/phase1_brick_a_nack_{split}.npz"


# ===========================================================================
# TRUNK (frozen, live forwards, prefix-conditioned)
# ===========================================================================

def build_trunk():
    from mycelium.llama_loader import (
        attach_llama_layers, load_llama_weights, LLAMA_3_2_1B_CFG)

    class _H:
        pass
    host = _H()
    sd = load_llama_weights(os.path.join(
        _ROOT, ".cache/llama-3.2-1b-weights/model.safetensors"))
    attach_llama_layers(host, n_layers=4, sd=sd, cfg=LLAMA_3_2_1B_CFG)
    del sd
    return host


def trunk_forward_prefixed(host, tok_ids, prefix):
    """tok_ids (B,T) int · prefix (B,K_PREFIX,H_TRUNK). Returns token-aligned states
    (B,T,H_TRUNK) — the prefix positions are stripped after the frozen layers."""
    from tinygrad import Tensor, dtypes
    from mycelium.llama_loader import _rms_norm
    cfg = host.llama_cfg
    x_tok = host.llama_embed[tok_ids].cast(dtypes.float)         # (B,T,H)
    x = prefix.cat(x_tok, dim=1)                                  # (B,K+T,H)
    for layer in host.llama_layers:
        x = layer(x, host.llama_rope_cos, host.llama_rope_sin)
    x = _rms_norm(x, host.llama_layers[-1].ffn_norm, cfg.rms_norm_eps)
    return x[:, K_PREFIX:]                                        # (B,T,H)


# ===========================================================================
# CONDITIONING ENCODER (the ONE shared, cycle-invariant interface — sanctioned)
# ===========================================================================

def build_encoder_params(seed: int = 1) -> dict:
    from tinygrad import Tensor, dtypes
    rng = np.random.RandomState(seed)

    def t(a):
        x = Tensor(a.astype(np.float32), dtype=dtypes.float,
                   requires_grad=True).contiguous().realize()
        x.requires_grad = True
        return x

    return {
        "enc_w1": t(rng.randn(COND_DIM, 256) / math.sqrt(COND_DIM)),
        "enc_b1": t(np.zeros((256,))),
        # ZERO-INIT output: blank/any conditioning -> prefix == 0 at step 0, so the
        # warm-started head starts from a consistent (if shifted-RoPE) graph.
        "enc_w2": t(np.zeros((256, K_PREFIX * H_TRUNK))),
        "enc_b2": t(np.zeros((K_PREFIX * H_TRUNK,))),
    }


def encode_cond(enc: dict, cond):
    """cond (B, COND_DIM) -> prefix (B, K_PREFIX, H_TRUNK)."""
    h = (cond @ enc["enc_w1"] + enc["enc_b1"]).gelu()
    p = h @ enc["enc_w2"] + enc["enc_b2"]
    return p.reshape(cond.shape[0], K_PREFIX, H_TRUNK)


# ===========================================================================
# NACK FLAGS (oracle arm): gold-wrong slots -> their SENTENCES -> flag vector
# ===========================================================================

def wrong_slot_mask(o_np: dict, gold: dict, i: int, bi: int) -> np.ndarray:
    """Per-slot bool: slot is present-in-gold AND any field wrong (the oracle)."""
    wrong = np.zeros((L_SLOTS,), bool)
    for j in range(L_SLOTS):
        pg = gold["presence"][i, j] > 0.5
        pp = o_np["pres"][bi, j] > 0.0
        if pg != pp:
            wrong[j] = True
            continue
        if not pg:
            continue
        bad = int(o_np["type"][bi, j].argmax()) != gold["type"][i, j]
        bad |= not bool(((o_np["mem"][bi, j] > 0.0) == (gold["members"][i, j] > 0.5)).all())
        if gold["is_cage"][i, j] > 0.5:
            bad |= int(o_np["op"][bi, j].argmax()) != gold["op"][i, j]
            bad |= not bool((o_np["dig"][bi, j].argmax(-1) == gold["digits"][i, j]).all())
        wrong[j] = bad
    return wrong


def slots_to_sent_flags(wrong, gold_span_i, sent_i) -> np.ndarray:
    """wrong (L,) bool + span (L,T) + sent (T,) -> cond vector (COND_DIM,)."""
    flags = np.zeros((COND_DIM,), np.float32)
    for j in np.where(wrong)[0]:
        toks = np.where(gold_span_i[j] > 0)[0]
        for s in np.unique(sent_i[toks]) if len(toks) else []:
            flags[int(s)] = 1.0
    if wrong.any():
        flags[-1] = 1.0                       # the global-fail bit
    return flags


def shuffle_flags(flags: np.ndarray, rng) -> np.ndarray:
    """The CONTROL: same number of flagged sentences, moved to innocent ones."""
    out = np.zeros_like(flags)
    on = np.where(flags[:-1] > 0)[0]
    n_sent = int(flags[:-1].sum())
    innocent = [s for s in range(SENT_MAX) if flags[s] == 0]
    pick = rng.choice(innocent, size=min(len(on), len(innocent)), replace=False)
    out[pick] = 1.0
    out[-1] = flags[-1]
    return out


# ===========================================================================
# PREP (--prep): the plateaued parser's ORGANIC failures + oracle flags
# ===========================================================================

def do_prep(prep_n: int) -> None:
    from tinygrad import Tensor, dtypes
    from tinygrad.nn.state import safe_load

    for split, cap in (("train", prep_n), ("test", 10 ** 9)):
        samples, states, tokmask, gold, sent = load_split(split)
        n = min(len(samples), cap)
        p = build_head_params(0)
        sd = safe_load(CKPT_PATH)
        for k in p:
            p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
        flags = np.zeros((n, COND_DIM), np.float32)
        has_fail = np.zeros((n,), bool)
        for s0 in range(0, n, 8):
            sl = slice(s0, min(s0 + 8, n))
            out = head_forward(
                p, Tensor(np.asarray(states[sl], dtype=np.float32), dtype=dtypes.float),
                Tensor(tokmask[sl].astype(np.float32), dtype=dtypes.float),
                Tensor(np.ones((1, 1, H_WAIST), np.float32), dtype=dtypes.float),
                Tensor(sent[sl].astype(np.int32), dtype=dtypes.int))
            o = {k: out[k].realize().numpy() for k in ("pres", "type", "op", "dig", "mem")}
            for bi, i in enumerate(range(s0, min(s0 + 8, n))):
                wrong = wrong_slot_mask(o, gold, i, bi)
                has_fail[i] = wrong.any()
                flags[i] = slots_to_sent_flags(
                    wrong, gold["span"][i].astype(np.float32), sent[i])
            if s0 % 2048 == 0:
                print(f"  [{split}] {s0}/{n}", flush=True)
        np.savez_compressed(NACK_NPZ.format(split=split),
                            flags=flags, has_fail=has_fail, n=n)
        print(f"[prep] {split}: {int(has_fail.sum())}/{n} organic failures -> "
              f"{NACK_NPZ.format(split=split)}", flush=True)


# ===========================================================================
# TRAIN (--train): head+encoder jointly, trunk FROZEN, live prefixed forwards
# ===========================================================================

def do_train(steps: int, lr: float, batch: int, seed: int) -> None:
    from tinygrad import Tensor, dtypes
    from tinygrad.engine.jit import TinyJit
    from tinygrad.nn.optim import AdamW
    from tinygrad.nn.state import safe_load, safe_save

    samples, _states, tokmask, gold, sent = load_split("train")
    _s2, ids, _m2, _off = tokenize_corpus(NL_TRAIN)
    z = np.load(NACK_NPZ.format(split="train"))
    flags_all, has_fail, n = z["flags"], z["has_fail"], int(z["n"])
    fail_idx = np.where(has_fail[:n])[0]
    print(f"[train] {len(fail_idx)}/{n} organic failures available", flush=True)

    host = build_trunk()
    p = build_head_params(seed)
    sd = safe_load(CKPT_PATH)                 # warm-start the plateaued head
    for k in p:
        p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
    enc = build_encoder_params(seed + 1)
    params = list(p.values()) + list(enc.values())
    opt = AdamW(params, lr=lr, weight_decay=0.01)
    rng = np.random.RandomState(seed)

    def fix(a, dt):
        return Tensor(a, dtype=dt).contiguous().realize()
    b_ids = fix(np.zeros((batch, T_WINDOW), np.int32), dtypes.int)
    b_tok = fix(np.zeros((batch, T_WINDOW), np.float32), dtypes.float)
    b_sent = fix(np.zeros((batch, T_WINDOW), np.int32), dtypes.int)
    b_cond = fix(np.zeros((batch, COND_DIM), np.float32), dtypes.float)
    b_wm = fix(np.ones((1, 1, H_WAIST), np.float32), dtypes.float)
    b_gold = {
        k: fix(np.zeros((batch, L_SLOTS) + tail, np.float32), dtypes.float)
        for k, tail in (("presence", ()), ("is_cage", ()),
                        ("members", (S_CELLS,)), ("span", (T_WINDOW,)))
    }
    for k, tail in (("type", ()), ("op", ()), ("digits", (N_DIGITS,))):
        b_gold[k] = fix(np.zeros((batch, L_SLOTS) + tail, np.int32), dtypes.int)

    @TinyJit
    def train_step():
        Tensor.training = True
        prefix = encode_cond(enc, b_cond)
        trunk = trunk_forward_prefixed(host, b_ids, prefix)
        out = head_forward(p, trunk, b_tok, b_wm, b_sent)
        total, parts = head_loss(out, b_gold)
        opt.zero_grad()
        total.backward()
        opt.step()
        return total.realize(), parts["mem"].realize()

    t0 = time.time()
    for step in range(steps):
        idx = rng.choice(fail_idx, batch, replace=False)
        cond = flags_all[idx].copy()
        blank = rng.rand(batch) < 0.25        # 25% blank-conditioned mix
        cond[blank] = 0.0
        b_ids.assign(Tensor(ids[idx], dtype=dtypes.int).contiguous()).realize()
        b_tok.assign(Tensor(tokmask[idx].astype(np.float32), dtype=dtypes.float).contiguous()).realize()
        b_sent.assign(Tensor(sent[idx].astype(np.int32), dtype=dtypes.int).contiguous()).realize()
        b_cond.assign(Tensor(cond, dtype=dtypes.float).contiguous()).realize()
        for kk in ("presence", "is_cage", "members", "span"):
            b_gold[kk].assign(Tensor(gold[kk][idx].astype(np.float32), dtype=dtypes.float).contiguous()).realize()
        for kk in ("type", "op", "digits"):
            b_gold[kk].assign(Tensor(gold[kk][idx].astype(np.int32), dtype=dtypes.int).contiguous()).realize()
        tot, lmem = train_step()
        if step % 100 == 0 or step == steps - 1:
            tv = float(tot.numpy())
            assert np.isfinite(tv), f"non-finite loss at step {step}"
            print(f"  step {step:4d} loss={tv:.4f} mem={float(lmem.numpy()):.4f} "
                  f"({(time.time()-t0)/(step+1):.2f}s/step)", flush=True)

    safe_save({**p, **enc}, BRICK_A_CKPT)
    print(f"[train] saved {BRICK_A_CKPT}", flush=True)


# ===========================================================================
# THE MEASUREMENT (--eval): blank vs true-NACK vs shuffled-NACK, same graph
# ===========================================================================

def do_eval(seed: int = 0) -> None:
    from tinygrad import Tensor, dtypes
    from tinygrad.nn.state import safe_load

    samples, _st, tokmask, gold, sent = load_split("test")
    _s2, ids, _m2, _off = tokenize_corpus(NL_TEST)
    z = np.load(NACK_NPZ.format(split="test"))
    flags_all, has_fail = z["flags"], z["has_fail"]
    n = len(samples)

    host = build_trunk()
    p = build_head_params(0)
    enc = build_encoder_params(1)
    sd = safe_load(BRICK_A_CKPT)
    for d in (p, enc):
        for k in d:
            d[k].assign(sd[k].to(d[k].device).cast(d[k].dtype)).realize()
    rng = np.random.RandomState(seed)

    def run_pass(idx, cond_np):
        outs = {}
        for s0 in range(0, len(idx), 8):
            sl = idx[s0:s0 + 8]
            pad = 8 - len(sl)
            sl_p = np.concatenate([sl, sl[:1].repeat(pad)]) if pad else sl
            cond_b = cond_np[s0:s0 + 8]
            if pad:
                cond_b = np.concatenate([cond_b, cond_b[:1].repeat(pad, 0)])
            prefix = encode_cond(enc, Tensor(cond_b, dtype=dtypes.float))
            trunk = trunk_forward_prefixed(
                host, Tensor(ids[sl_p], dtype=dtypes.int), prefix)
            out = head_forward(
                p, trunk,
                Tensor(tokmask[sl_p].astype(np.float32), dtype=dtypes.float),
                Tensor(np.ones((1, 1, H_WAIST), np.float32), dtype=dtypes.float),
                Tensor(sent[sl_p].astype(np.int32), dtype=dtypes.int))
            o = {k: out[k].realize().numpy() for k in
                 ("pres", "type", "op", "dig", "mem", "attn_mean")}
            for bi, i in enumerate(sl):
                outs[int(i)] = {k: o[k][bi] for k in o}
        return outs

    fail_i = np.array([i for i in range(n) if has_fail[i]])
    cond_true = flags_all[fail_i]
    cond_blank = np.zeros_like(cond_true)
    cond_shuf = np.stack([shuffle_flags(f, rng) for f in cond_true])

    blank = run_pass(fail_i, cond_blank)
    true_p = run_pass(fail_i, cond_true)
    shuf = run_pass(fail_i, cond_shuf)

    def slot_delta_mass(a, b):
        m = np.zeros((L_SLOTS,))
        for j in range(L_SLOTS):
            for k in ("pres", "type", "op", "mem"):
                m[j] += np.abs(a[k][j] - b[k][j]).sum()
            m[j] += np.abs(a["dig"][j] - b["dig"][j]).sum()
        return m

    stats = {"fix_true": [0, 0], "fix_shuf": [0, 0], "pres_true": [0, 0],
             "pres_shuf": [0, 0], "conc": [], "conc_shuf": [], "solve": [0, 0, 0]}
    from phase1_delta_head import _gold_grid_cache
    from mycelium.csp_domains import problem_from_kenken
    from mycelium.csp_core import solve_symbolic

    def solves(o_np, smp):
        N = int(smp["N"])
        pred = decode_slots(o_np, N)
        cages, clues = [], []
        for f in pred:
            if f["ftype"] != "cage":
                continue
            rc = [[m // 7, m % 7] for m in f["members_flat"]]
            if not rc or any(r >= N or c >= N for (r, c) in rc):
                return 0
            cages.append(rc); clues.append([f["op"], f["target"]])
        try:
            res = solve_symbolic(problem_from_kenken(N, cages, clues),
                                 budget=100_000, seed=0)
        except Exception:
            return 0
        if res["status"] != "solved":
            return 0
        grid = [[int(res["assignment"][r * N + c]) for c in range(N)] for r in range(N)]
        return int(grid == _gold_grid_cache(smp)[0])

    for i in fail_i:
        i = int(i)
        wrong_b = wrong_slot_mask({k: v[None] for k, v in blank[i].items()}, gold, i, 0)
        wrong_t = wrong_slot_mask({k: v[None] for k, v in true_p[i].items()}, gold, i, 0)
        wrong_s = wrong_slot_mask({k: v[None] for k, v in shuf[i].items()}, gold, i, 0)
        flagged = np.where(wrong_b)[0]
        ok_slots = np.where(~wrong_b & (gold["presence"][i] > 0.5))[0]
        stats["fix_true"][0] += int((~wrong_t[flagged]).sum()); stats["fix_true"][1] += len(flagged)
        stats["fix_shuf"][0] += int((~wrong_s[flagged]).sum()); stats["fix_shuf"][1] += len(flagged)
        stats["pres_true"][0] += int((~wrong_t[ok_slots]).sum()); stats["pres_true"][1] += len(ok_slots)
        stats["pres_shuf"][0] += int((~wrong_s[ok_slots]).sum()); stats["pres_shuf"][1] += len(ok_slots)
        for key, arms in (("conc", true_p), ("conc_shuf", shuf)):
            dm = slot_delta_mass(arms[i], blank[i])
            if dm.sum() > 1e-9 and len(flagged):
                share = len(flagged) / max((gold["presence"][i] > 0.5).sum(), 1)
                stats[key].append((dm[flagged].sum() / dm.sum()) / max(share, 1e-9))
        smp = samples[i]
        stats["solve"][0] += solves(blank[i], smp)
        stats["solve"][1] += solves(true_p[i], smp)
        stats["solve"][2] += solves(shuf[i], smp)

    ft = stats["fix_true"]; fs = stats["fix_shuf"]
    pt = stats["pres_true"]; ps = stats["pres_shuf"]
    print(f"\n[brick-A] organic test failures evaluated: {len(fail_i)}")
    print(f"  flagged-slot FIX rate : true-NACK {ft[0]}/{ft[1]} = {ft[0]/max(ft[1],1):.3f}"
          f"   shuffled {fs[0]}/{fs[1]} = {fs[0]/max(fs[1],1):.3f}")
    print(f"  correct-slot PRESERVE : true-NACK {pt[0]/max(pt[1],1):.3f}"
          f"   shuffled {ps[0]/max(ps[1],1):.3f}")
    print(f"  delta CONCENTRATION (mass-on-flagged / flagged-share; 1.0 = uniform):")
    print(f"    true-NACK {np.mean(stats['conc']):.2f}   shuffled {np.mean(stats['conc_shuf']):.2f}")
    print(f"  SOLVES on these failures: blank {stats['solve'][0]}  "
          f"true-NACK {stats['solve'][1]}  shuffled {stats['solve'][2]}")
    print(f"  KILL CHECK: null LIVES iff fix(true) >> fix(shuffled) AND concentration"
          f" > 1; otherwise the LoRA ladder (§8.4).")


# ===========================================================================
# SELFTEST (CPU)
# ===========================================================================

def selftest() -> None:
    os.environ.setdefault("DEV", "CPU")
    from tinygrad import Tensor, dtypes

    enc = build_encoder_params(1)
    cond = Tensor(np.random.RandomState(0).rand(2, COND_DIM).astype(np.float32))
    pref = encode_cond(enc, cond)
    assert pref.shape == (2, K_PREFIX, H_TRUNK)
    assert float(pref.abs().max().numpy()) == 0.0, "zero-init output must give zero prefix"
    print("  [OK] encoder shapes + zero-init prefix")

    # flags: wrong slot -> its sentences flagged, global bit set
    wrong = np.zeros((L_SLOTS,), bool); wrong[3] = True
    span = np.zeros((L_SLOTS, T_WINDOW), np.float32); span[3, 10:14] = 1.0
    sent = np.zeros((T_WINDOW,), np.int32); sent[10:14] = 5
    f = slots_to_sent_flags(wrong, span, sent)
    assert f[5] == 1.0 and f[-1] == 1.0 and f[:5].sum() == 0, f
    print("  [OK] wrong-slot -> sentence flags")

    # shuffled control: same count, different sentences
    rng = np.random.RandomState(0)
    sh = shuffle_flags(f, rng)
    assert sh[:-1].sum() == f[:-1].sum() and sh[5] == 0.0 and sh[-1] == 1.0
    print("  [OK] shuffled-flags control moves flags to innocent sentences")
    print("[selftest] PASS")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--prep", action="store_true")
    ap.add_argument("--prep-n", type=int, default=8000)
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--eval", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        selftest()
    elif args.prep:
        do_prep(args.prep_n)
    elif args.train:
        do_train(steps=int(os.environ.get("STEPS", "3000")),
                 lr=float(os.environ.get("LR", "1e-4")),
                 batch=int(os.environ.get("BATCH", "8")),
                 seed=int(os.environ.get("SEED", "0")))
    elif args.eval:
        do_eval()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
