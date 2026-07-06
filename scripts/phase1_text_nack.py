"""phase1_text_nack.py — the TEXT-RENDERED NACK arm: trunk-level conditioning with
ZERO new gradient paths (spec §9 registration, 2026-07-06).

THE MOVE: render the NACK as literal text prepended to the problem —
    "NOTE: check statement 3, statement 7 and statement 12. <original problem>"
— and run the ordinary frozen forward. The flags now change how the text is READ
(trunk activations differ), which is exactly what the head-level arm structurally
could not do; and the training graph is the 68k-step-proven head class (backward
stops at the head input). No prefix params, no backward-through-trunk, no driver
exposure. The C1-A counter-prior is registered: "telling" predicts a MODEST effect.

ISOLATION: the head-level suspect/fail embeddings are NOT used here — this arm
isolates the trunk channel. Training = the same FLAG-DEPENDENT objective proven in
the head arm (flagged slots -> gold fix; unflagged + blank-mix -> copy-previous), so
the two arms differ ONLY in where conditioning enters.

METRIC (pre-registered after the third instrument sighting): the localization
instrument is the FLAGGED-vs-UNFLAGGED FLIP-RATE RATIO at the decision level
(a flip = any field decision change vs the blank pass), alongside fix rate,
preservation, and solves.

STATEMENT NUMBERING: 1-based over the original text's sentences (the note itself
becomes sentence 0 of the conditioned text; original sentence k -> conditioned
sentence k+1 — handled naturally by re-running the tokenize/gold pipeline on the
conditioned sample). Token budget: notes cite at most MAX_CITED statements; samples
that still exceed T=512 are DROPPED WITH A COUNT (the truncation guard's law).

USAGE:
  Selftest (CPU):    .venv/bin/python3 scripts/phase1_text_nack.py --selftest
  Precompute:        DEV=AMD .venv/bin/python3 scripts/phase1_text_nack.py --precompute
  Train:             DEV=AMD STEPS=3000 .venv/bin/python3 scripts/phase1_text_nack.py --train
  Measurement:       DEV=AMD .venv/bin/python3 scripts/phase1_text_nack.py --eval
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

from phase1_delta_head import (  # noqa: E402
    T_WINDOW, H_TRUNK, H_WAIST, L_SLOTS, S_CELLS, N_DIGITS, SENT_MAX,
    NL_TRAIN, NL_TEST, CKPT_PATH, TOKENIZER_JSON,
    build_head_params, head_forward, head_loss, load_split,
    build_gold, sentence_indices, decode_slots, _solve_rate_one,
)
from phase1_brick_a import (  # noqa: E402
    COND_DIM, NACK_NPZ, wrong_slot_mask, shuffle_flags,
)

MAX_CITED = 12
TEXT_CKPT = ".cache/phase1_text_nack.safetensors"
COND_NPZ = ".cache/phase1_text_nack_{name}.npz"     # conditioned states + gold + sent


# ===========================================================================
# CONDITIONED SAMPLE CONSTRUCTION (note + shifted gold, pipeline-reused)
# ===========================================================================

def render_note(flags: np.ndarray) -> str:
    """flags (COND_DIM,) -> the NACK note text. Cites 1-based statement numbers."""
    sents = [int(s) + 1 for s in np.where(flags[:SENT_MAX] > 0)[0]][:MAX_CITED]
    if not sents:
        return ""
    if len(sents) == 1:
        body = f"statement {sents[0]}"
    else:
        body = ", ".join(f"statement {s}" for s in sents[:-1]) + f" and statement {sents[-1]}"
    return f"NOTE: check {body}."


def conditioned_sample(smp: dict, flags: np.ndarray) -> dict:
    """Prepend the note; shift every factor's char spans by the note length."""
    note = render_note(flags)
    if not note:
        return smp
    off = len(note) + 1
    out = dict(smp)
    out["text"] = note + " " + smp["text"]
    out["factors"] = [
        {**f, "spans": [[s + off, e + off] for (s, e) in f["spans"]]}
        for f in smp["factors"]
    ]
    return out


def tokenize_samples(samples: list):
    """tokenize_corpus's body for in-memory samples (same guard, DROP over-budget)."""
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(TOKENIZER_JSON)
    ids = np.zeros((len(samples), T_WINDOW), np.int32)
    mask = np.zeros((len(samples), T_WINDOW), np.float32)
    offsets, keep = [], []
    for i, s in enumerate(samples):
        e = tok.encode(s["text"])
        if len(e.ids) > T_WINDOW:
            offsets.append([])
            continue
        ids[i, : len(e.ids)] = e.ids
        mask[i, : len(e.ids)] = 1.0
        offsets.append(list(e.offsets))
        keep.append(i)
    return ids, mask, offsets, np.array(keep, dtype=np.int64)


def build_conditioned_pack(samples, flags_rows):
    """samples + per-sample flags -> (ids, tokmask, gold, sent, keep)."""
    cond = [conditioned_sample(s, f) for s, f in zip(samples, flags_rows)]
    ids, mask, offsets, keep = tokenize_samples(cond)
    gold = build_gold([cond[i] for i in keep], [offsets[i] for i in keep])
    sent = np.stack([sentence_indices(cond[i]["text"], offsets[i], mask[i])
                     for i in keep])
    return ids[keep], mask[keep], gold, sent, keep


# ===========================================================================
# PRECOMPUTE (--precompute): conditioned trunk states, frozen forward only
# ===========================================================================

def _run_trunk(host, ids: np.ndarray) -> np.ndarray:
    from tinygrad import Tensor, dtypes
    from mycelium.llama_loader import _rms_norm
    cfg = host.llama_cfg
    n = ids.shape[0]
    out = np.zeros((n, T_WINDOW, H_TRUNK), np.float16)
    for s0 in range(0, n, 8):
        sl = slice(s0, min(s0 + 8, n))
        x = host.llama_embed[Tensor(ids[sl], dtype=dtypes.int)]
        for layer in host.llama_layers:
            x = layer(x, host.llama_rope_cos, host.llama_rope_sin)
        x = _rms_norm(x, host.llama_layers[-1].ffn_norm, cfg.rms_norm_eps)
        chunk = x.cast(dtypes.float).realize().numpy()
        assert np.isfinite(chunk).all()
        out[sl] = chunk.astype(np.float16)
    return out


def do_precompute() -> None:
    from phase1_brick_a import build_trunk
    host = build_trunk()
    rng = np.random.RandomState(0)

    jobs = []
    # train failures, TRUE flags (the training corpus)
    s_tr, _st, _tm, _g, _se = load_split("train")
    z = np.load(NACK_NPZ.format(split="train"))
    fail_tr = np.where(z["has_fail"][: int(z["n"])])[0]
    jobs.append(("train_true", [s_tr[i] for i in fail_tr], z["flags"][fail_tr], fail_tr))
    # test failures, TRUE + SHUFFLED flags (the measurement arms)
    s_te, _st2, _tm2, _g2, _se2 = load_split("test")
    zt = np.load(NACK_NPZ.format(split="test"))
    fail_te = np.where(zt["has_fail"])[0]
    f_true = zt["flags"][fail_te]
    f_shuf = np.stack([shuffle_flags(f, rng) for f in f_true])
    jobs.append(("test_true", [s_te[i] for i in fail_te], f_true, fail_te))
    jobs.append(("test_shuf", [s_te[i] for i in fail_te], f_shuf, fail_te))

    for name, samples, flags_rows, orig_idx in jobs:
        ids, mask, gold, sent, keep = build_conditioned_pack(samples, flags_rows)
        dropped = len(samples) - len(keep)
        states = _run_trunk(host, ids)
        np.savez_compressed(
            COND_NPZ.format(name=name),
            states=states.reshape(len(keep), -1),   # flat -> npz-friendly
            shape=np.array(states.shape),
            tokmask=mask.astype(np.uint8), sent=sent.astype(np.int8),
            orig_idx=orig_idx[keep], flags=flags_rows[keep],
            **{f"g_{k}": (v.astype(np.uint8) if v.dtype == np.float32 else v.astype(np.int8))
               for k, v in gold.items() if k != "N"},
            g_N=gold["N"])
        print(f"[precompute] {name}: {states.shape} (dropped {dropped} over-budget) "
              f"-> {COND_NPZ.format(name=name)}", flush=True)


def load_pack(name: str):
    z = np.load(COND_NPZ.format(name=name))
    shp = tuple(z["shape"])
    states = z["states"].reshape(shp)
    gold = {k[2:]: z[k] for k in z.files if k.startswith("g_")}
    gold["N"] = gold["N"].astype(np.int32)
    return states, z["tokmask"], gold, z["sent"], z["orig_idx"], z["flags"]


# ===========================================================================
# TRAIN (--train): flag-dependent objective on conditioned states; proven graph
# ===========================================================================

def do_train(steps: int, lr: float, batch: int, seed: int) -> None:
    from tinygrad import Tensor, dtypes
    from tinygrad.engine.jit import TinyJit
    from tinygrad.nn.optim import AdamW
    from tinygrad.nn.state import safe_load, safe_save

    c_states, c_tok, c_gold, c_sent, c_orig, _f = load_pack("train_true")
    b_samples, b_states, b_tok_all, b_gold_all, b_sent_all = load_split("train")
    z = np.load(NACK_NPZ.format(split="train"))
    wrong_slots = z["wrong_slots"]
    prev = {k[5:]: z[k] for k in z.files if k.startswith("prev_")}
    nc = c_states.shape[0]
    print(f"[train] TEXT-NACK arm: {nc} conditioned train failures", flush=True)

    p = build_head_params(seed)
    sd = safe_load(CKPT_PATH)
    for k in p:
        p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
    opt = AdamW(list(p.values()), lr=lr, weight_decay=0.01)
    rng = np.random.RandomState(seed)

    def fix(a, dt):
        return Tensor(a, dtype=dt).contiguous().realize()
    b_trunk = fix(np.zeros((batch, T_WINDOW, H_TRUNK), np.float32), dtypes.float)
    b_tok = fix(np.zeros((batch, T_WINDOW), np.float32), dtypes.float)
    b_sent = fix(np.zeros((batch, T_WINDOW), np.int32), dtypes.int)
    b_wm = fix(np.ones((1, 1, H_WAIST), np.float32), dtypes.float)
    b_gold = {k: fix(np.zeros((batch, L_SLOTS) + tail, np.float32), dtypes.float)
              for k, tail in (("presence", ()), ("is_cage", ()),
                              ("members", (S_CELLS,)), ("span", (T_WINDOW,)))}
    for k, tail in (("type", ()), ("op", ()), ("digits", (N_DIGITS,))):
        b_gold[k] = fix(np.zeros((batch, L_SLOTS) + tail, np.int32), dtypes.int)

    @TinyJit
    def step_fn():
        Tensor.training = True
        out = head_forward(p, b_trunk, b_tok, b_wm, b_sent)
        total, parts = head_loss(out, b_gold)
        opt.zero_grad()
        total.backward()
        opt.step()
        return total.realize(), parts["mem"].realize()

    t0 = time.time()
    for step in range(steps):
        use_blank = rng.rand() < 0.25
        if use_blank:
            # blank-mix: ORIGINAL text/states, targets = copy-previous on all slots
            oi = rng.choice(np.where(z["has_fail"][: int(z["n"])])[0], batch, replace=False)
            trunk_np = np.asarray(b_states[oi], dtype=np.float32)
            tok_np, sent_np = b_tok_all[oi], b_sent_all[oi]
            g_src, use_fix = b_gold_all, np.zeros((batch, L_SLOTS), bool)
            gi = oi
        else:
            ci = rng.choice(nc, batch, replace=False)
            trunk_np = c_states[ci].astype(np.float32)
            tok_np, sent_np = c_tok[ci], c_sent[ci]
            g_src, use_fix = c_gold, wrong_slots[c_orig[ci]].astype(bool)
            gi = ci
        tg = {}
        pv = c_orig[ci] if not use_blank else oi
        tg["presence"] = np.where(use_fix, g_src["presence"][gi] > 0.5,
                                  prev["pres"][pv] > 0.5).astype(np.float32)
        tg["is_cage"] = (g_src["is_cage"][gi] > 0.5).astype(np.float32)
        tg["span"] = g_src["span"][gi].astype(np.float32)
        tg["members"] = np.where(use_fix[:, :, None], g_src["members"][gi] > 0.5,
                                 prev["mem"][pv] > 0.5).astype(np.float32)
        tg["type"] = np.where(use_fix, g_src["type"][gi], prev["type"][pv]).astype(np.int32)
        tg["op"] = np.where(use_fix, g_src["op"][gi], prev["op"][pv]).astype(np.int32)
        tg["digits"] = np.where(use_fix[:, :, None], g_src["digits"][gi],
                                prev["dig"][pv]).astype(np.int32)
        b_trunk.assign(Tensor(trunk_np, dtype=dtypes.float).contiguous()).realize()
        b_tok.assign(Tensor(tok_np.astype(np.float32), dtype=dtypes.float).contiguous()).realize()
        b_sent.assign(Tensor(sent_np.astype(np.int32), dtype=dtypes.int).contiguous()).realize()
        for kk in ("presence", "is_cage", "members", "span"):
            b_gold[kk].assign(Tensor(tg[kk], dtype=dtypes.float).contiguous()).realize()
        for kk in ("type", "op", "digits"):
            b_gold[kk].assign(Tensor(tg[kk], dtype=dtypes.int).contiguous()).realize()
        tot, lmem = step_fn()
        if step % 500 == 0 or step == steps - 1:
            print(f"  step {step:4d} loss={float(tot.numpy()):.4f} "
                  f"mem={float(lmem.numpy()):.4f} ({(time.time()-t0)/(step+1):.2f}s/step)",
                  flush=True)
    safe_save(p, TEXT_CKPT)
    print(f"[train] saved {TEXT_CKPT}", flush=True)


# ===========================================================================
# MEASUREMENT (--eval): blank vs true-note vs shuffled-note; flip-rate metric
# ===========================================================================

def do_eval() -> None:
    from tinygrad import Tensor, dtypes
    from tinygrad.nn.state import safe_load

    samples, bl_states, bl_tok, bl_gold, bl_sent = load_split("test")
    zt = np.load(NACK_NPZ.format(split="test"))
    p = build_head_params(0)
    sd = safe_load(TEXT_CKPT)
    for k in p:
        p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()

    def run(states, tok, sent, idx_list):
        outs = {}
        n = len(idx_list)
        for s0 in range(0, n, 8):
            sl = np.arange(s0, min(s0 + 8, n))
            pad = 8 - len(sl)
            sl_p = np.concatenate([sl, sl[:1].repeat(pad)]) if pad else sl
            out = head_forward(
                p, Tensor(np.asarray(states[sl_p], dtype=np.float32), dtype=dtypes.float),
                Tensor(tok[sl_p].astype(np.float32), dtype=dtypes.float),
                Tensor(np.ones((1, 1, H_WAIST), np.float32), dtype=dtypes.float),
                Tensor(sent[sl_p].astype(np.int32), dtype=dtypes.int))
            o = {k: out[k].realize().numpy() for k in ("pres", "type", "op", "dig", "mem")}
            for bi, j in enumerate(sl):
                outs[int(idx_list[j])] = {k: o[k][bi] for k in o}
        return outs

    t_states, t_tok, t_gold, t_sent, t_orig, _ = load_pack("test_true")
    s_states, s_tok, s_gold, s_sent, s_orig, _ = load_pack("test_shuf")
    common = sorted(set(t_orig.tolist()) & set(s_orig.tolist()))
    blank = run(bl_states, bl_tok, bl_sent, list(range(len(samples))))
    true_p = run(t_states, t_tok, t_sent, t_orig.tolist())
    shuf = run(s_states, s_tok, s_sent, s_orig.tolist())

    def decisions(o):
        return (o["pres"] > 0, o["type"].argmax(-1), o["op"].argmax(-1),
                o["dig"].argmax(-1), o["mem"] > 0)

    def slot_flips(a, b):
        da, db = decisions(a), decisions(b)
        fl = np.zeros((L_SLOTS,), bool)
        for j in range(L_SLOTS):
            fl[j] = any(not np.array_equal(x[j], y[j]) for x, y in zip(da, db))
        return fl

    stats = {"fix_t": [0, 0], "fix_s": [0, 0], "pres_t": [0, 0], "pres_s": [0, 0],
             "flip_flag": [0, 0], "flip_unflag": [0, 0],
             "solve": [0, 0, 0]}
    for i in common:
        i = int(i)
        wrong_b = wrong_slot_mask({k: v[None] for k, v in blank[i].items()}, bl_gold, i, 0)
        wrong_t = wrong_slot_mask({k: v[None] for k, v in true_p[i].items()}, bl_gold, i, 0)
        wrong_s = wrong_slot_mask({k: v[None] for k, v in shuf[i].items()}, bl_gold, i, 0)
        flagged = np.where(wrong_b)[0]
        present = bl_gold["presence"][i] > 0.5
        ok_slots = np.where(~wrong_b & present)[0]
        stats["fix_t"][0] += int((~wrong_t[flagged]).sum()); stats["fix_t"][1] += len(flagged)
        stats["fix_s"][0] += int((~wrong_s[flagged]).sum()); stats["fix_s"][1] += len(flagged)
        stats["pres_t"][0] += int((~wrong_t[ok_slots]).sum()); stats["pres_t"][1] += len(ok_slots)
        stats["pres_s"][0] += int((~wrong_s[ok_slots]).sum()); stats["pres_s"][1] += len(ok_slots)
        fl = slot_flips(true_p[i], blank[i])
        stats["flip_flag"][0] += int(fl[flagged].sum()); stats["flip_flag"][1] += len(flagged)
        stats["flip_unflag"][0] += int(fl[ok_slots].sum()); stats["flip_unflag"][1] += len(ok_slots)
        smp = samples[i]
        stats["solve"][0] += _solve_rate_one({k: v[None] for k, v in blank[i].items()}, 0, smp)
        stats["solve"][1] += _solve_rate_one({k: v[None] for k, v in true_p[i].items()}, 0, smp)
        stats["solve"][2] += _solve_rate_one({k: v[None] for k, v in shuf[i].items()}, 0, smp)

    ft, fs = stats["fix_t"], stats["fix_s"]
    ff, fu = stats["flip_flag"], stats["flip_unflag"]
    r_f = ff[0] / max(ff[1], 1)
    r_u = fu[0] / max(fu[1], 1)
    print(f"\n[text-NACK] test failures with both arms: {len(common)}")
    print(f"  flagged-slot FIX rate : true-note {ft[0]}/{ft[1]} = {ft[0]/max(ft[1],1):.3f}"
          f"   shuffled-note {fs[0]}/{fs[1]} = {fs[0]/max(fs[1],1):.3f}")
    print(f"  correct-slot PRESERVE : true {stats['pres_t'][0]/max(stats['pres_t'][1],1):.3f}"
          f"   shuffled {stats['pres_s'][0]/max(stats['pres_s'][1],1):.3f}")
    print(f"  FLIP-RATE (pre-registered): flagged {r_f:.3f} vs unflagged {r_u:.4f}"
          f"   ratio {r_f/max(r_u,1e-9):.0f}:1")
    print(f"  SOLVES: blank {stats['solve'][0]}  true-note {stats['solve'][1]}"
          f"  shuffled-note {stats['solve'][2]}")
    print(f"  (head-level arm baselines: fix 0.438 vs 0.360; solves 7 vs 4)")


# ===========================================================================
# SELFTEST (CPU)
# ===========================================================================

def selftest() -> None:
    f = np.zeros((COND_DIM,), np.float32); f[2] = 1; f[6] = 1; f[-1] = 1
    note = render_note(f)
    assert note == "NOTE: check statement 3 and statement 7.", note
    smp = {"N": 4, "text": "Alpha. Beta.", "factors": [
        {"ftype": "row", "members_flat": [0], "op": None, "target": None,
         "spans": [[0, 6]]}]}
    c = conditioned_sample(smp, f)
    off = len(note) + 1
    assert c["text"].startswith("NOTE: check") and c["text"].endswith("Alpha. Beta.")
    assert c["factors"][0]["spans"] == [[off, off + 6]]
    assert smp["factors"][0]["spans"] == [[0, 6]], "input must be untouched"
    empty = render_note(np.zeros((COND_DIM,), np.float32))
    assert empty == ""
    print("[selftest] PASS")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--precompute", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--eval", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        selftest()
    elif args.precompute:
        do_precompute()
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
