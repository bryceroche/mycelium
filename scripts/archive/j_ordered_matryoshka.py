"""j_ordered_matryoshka.py — sensitivity-ordered waist dims vs variance vs the
trained incumbent (the J-lens borrow, registered in spec §10 with refinements).

REGISTERED PREDICTION: J-ordering moves the prefix-width CLIFF LEFT. The measured
curve is flat 128~=512, so the sweep goes DOWN: widths 8/16/32/64/128/256/512.

FOUR ORDERINGS of the parse-side waist's 512 dims (head_forward's width mask is an
arbitrary (1,1,512) tensor — any ordering is just a different mask; zero forward
changes):
  identity : dims 0..511 (THE INCUMBENT — Matryoshka training kept 0-127 always on,
             so this carries the baked-in trained importance; not a strawman)
  variance : per-dim activation variance over the test set (the confound-prone
             default the 0.755 trap warns about)
  fisher_loss     : diag Fisher of the TRAINING loss — mean over instances of
                    (d head_loss / d dim)^2, per-sample backwards (signed averaging
                    would cancel; squared-then-averaged per registration)
  fisher_decision : diag Fisher of the DECISION target — the summed top-logit
                    margins of every head (what the DECISIONS care about;
                    wrong-but-equivalent proved loss and decision diverge)

Per (ordering, width): factor exact + solve rate on the 60-test set. NULL SCOPE
(registered): a null reads "diagonal sensitivity doesn't beat variance," not
"sensitivity wrong in principle" (diagonal misses jointly-load-bearing sets).

USAGE:  DEV=AMD .venv/bin/python3 scripts/j_ordered_matryoshka.py
"""
from __future__ import annotations

import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

import numpy as np

from phase1_delta_head import (  # noqa: E402
    T_WINDOW, H_TRUNK, H_WAIST, L_SLOTS, S_CELLS, N_DIGITS,
    build_head_params, head_forward, head_loss, load_split, decode_slots,
    _solve_rate_one, CKPT_PATH, TYPES,
)
from phase1_brick_a import wrong_slot_mask  # noqa: E402

WIDTHS = (8, 16, 32, 64, 128, 256, 512)


def gold_tensors(gold, idx, dtypes, Tensor):
    g = {}
    for kk in ("presence", "is_cage", "members", "span"):
        g[kk] = Tensor(gold[kk][idx].astype(np.float32), dtype=dtypes.float)
    for kk in ("type", "op", "digits"):
        g[kk] = Tensor(gold[kk][idx].astype(np.int32), dtype=dtypes.int)
    return g


def compute_orderings(p, states, tokmask, gold, sent, n_probe: int):
    """Returns dict name -> ordering (512,) int array, best-first."""
    from tinygrad import Tensor, dtypes

    # variance of waist activations over probe samples (full width)
    acc_var_sum = np.zeros(H_WAIST)
    acc_var_sq = np.zeros(H_WAIST)
    n_tok_tot = 0
    fisher = {"fisher_loss": np.zeros(H_WAIST), "fisher_decision": np.zeros(H_WAIST)}

    for i in range(n_probe):
        idx = np.array([i])
        trunk = Tensor(np.asarray(states[idx], dtype=np.float32), dtype=dtypes.float)
        tok = Tensor(tokmask[idx].astype(np.float32), dtype=dtypes.float)
        sent_t = Tensor(sent[idx].astype(np.int32), dtype=dtypes.int)
        g = gold_tensors(gold, idx, dtypes, Tensor)

        for target_name in ("fisher_loss", "fisher_decision"):
            # gate-grad trick: a requires_grad ones-mask; d(target)/d(gate_d) is the
            # aggregated per-dim sensitivity for THIS instance (per-sample backward
            # per the registration — no cross-instance sign cancellation).
            gate = Tensor(np.ones((1, 1, H_WAIST), np.float32),
                          dtype=dtypes.float, requires_grad=True)
            gate.requires_grad = True
            out = head_forward(p, trunk, tok, gate, sent_t)
            if target_name == "fisher_loss":
                target, _parts = head_loss(out, g)
            else:
                target = (out["pres"].abs().sum()
                          + out["type"].max(-1).sum() + out["op"].max(-1).sum()
                          + out["dig"].max(-1).sum() + out["mem"].abs().sum())
            target.backward()
            gvec = gate.grad.reshape(-1).numpy()
            fisher[target_name] += gvec.astype(np.float64) ** 2

        # variance pass (reuse the last forward's waist)
        w = out["waist"].detach().numpy()[0]            # (T,512)
        m = tokmask[i] > 0
        acc_var_sum += w[m].sum(0)
        acc_var_sq += (w[m] ** 2).sum(0)
        n_tok_tot += int(m.sum())

    var = acc_var_sq / n_tok_tot - (acc_var_sum / n_tok_tot) ** 2
    orderings = {
        "identity": np.arange(H_WAIST),
        "variance": np.argsort(-var),
        "fisher_loss": np.argsort(-fisher["fisher_loss"]),
        "fisher_decision": np.argsort(-fisher["fisher_decision"]),
    }
    return orderings


def eval_ordering(p, states, tokmask, gold, sent, samples, order, width):
    from tinygrad import Tensor, dtypes
    mask = np.zeros((1, 1, H_WAIST), np.float32)
    mask[..., order[:width]] = 1.0
    n = len(samples)
    fac_ok = fac_tot = solves = 0
    for s0 in range(0, n, 8):
        sl = np.arange(s0, min(s0 + 8, n))
        pad = 8 - len(sl)
        sl_p = np.concatenate([sl, sl[:1].repeat(pad)]) if pad else sl
        out = head_forward(
            p, Tensor(np.asarray(states[sl_p], dtype=np.float32), dtype=dtypes.float),
            Tensor(tokmask[sl_p].astype(np.float32), dtype=dtypes.float),
            Tensor(mask, dtype=dtypes.float),
            Tensor(sent[sl_p].astype(np.int32), dtype=dtypes.int))
        o = {k: out[k].realize().numpy() for k in ("pres", "type", "op", "dig", "mem")}
        for bi, i in enumerate(sl):
            i = int(i)
            wrong = wrong_slot_mask({k: v[bi][None] for k, v in o.items()}, gold, i, 0)
            present = gold["presence"][i] > 0.5
            fac_ok += int((~wrong[present]).sum())
            fac_tot += int(present.sum())
            solves += _solve_rate_one({k: v[bi][None] for k, v in o.items()}, 0,
                                      samples[i])
    return fac_ok / max(fac_tot, 1), solves


def main() -> None:
    from tinygrad import Tensor, dtypes
    from tinygrad.nn.state import safe_load

    samples, states, tokmask, gold, sent = load_split("test")
    p = build_head_params(0)
    sd = safe_load(CKPT_PATH)
    for k in p:
        p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()

    n_probe = min(len(samples), int(os.environ.get("N_PROBE", "60")))
    print(f"[j-matryoshka] computing orderings on {n_probe} probe samples "
          f"(per-sample Fisher backwards x2)...", flush=True)
    orderings = compute_orderings(p, states, tokmask, gold, sent, n_probe)
    for a in ("variance", "fisher_loss", "fisher_decision"):
        overlap = len(set(orderings[a][:128]) & set(np.arange(128))) / 128
        print(f"  {a:16s} top-128 overlap with incumbent prefix: {overlap:.2f}",
              flush=True)

    print(f"\n[j-matryoshka] prefix-width sweep (factor-exact / solves per cell):")
    header = "  width | " + " | ".join(f"{k:>15s}" for k in orderings)
    print(header)
    results = {}
    for w in WIDTHS:
        row = []
        for name, order in orderings.items():
            fe, sv = eval_ordering(p, states, tokmask, gold, sent, samples, order, w)
            results[(name, w)] = (fe, sv)
            row.append(f"{fe:.3f}/{sv:2d}")
        print(f"  {w:5d} | " + " | ".join(f"{c:>15s}" for c in row), flush=True)

    # the registered read: whose cliff sits further LEFT (last width >= 90% of its
    # own 512-width factor-exact)
    print("\n  CLIFF (last width retaining >=90% of own full-width factor-exact):")
    for name in orderings:
        full = results[(name, 512)][0]
        cliff = 512
        for w in reversed(WIDTHS):
            if results[(name, w)][0] >= 0.9 * full:
                cliff = w
        print(f"    {name:16s} cliff at width {cliff}")
    print("  (registered prediction: fisher_decision moves the cliff LEFT of the "
          "incumbent; null scope: diagonal-vs-variance only)")


if __name__ == "__main__":
    main()
