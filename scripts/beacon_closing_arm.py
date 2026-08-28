"""beacon_closing_arm.py — THE BEACON, fired as the ARC'S CLOSING MEASUREMENT
(spec registration, 2026-07-09, relay adjudication: last sentence for the 396,
either ending closes the chapter).

THE POPULATION: the 460 survivors — states correct at 99.6% (depth probe),
pointer deterministically mis-aimed (routing verdict), 86% unfixable under
PERFECT flags (oracle ceiling 13.9%), every conditioning-based repair dead by
measurement. The beacon is the ONLY untried arm: INPUT-level saliency,
mechanistically distinct from all head-side conditioning.

v0 MECHANISM (zero new params, forward-only — no trunk backprop, no AM
hazard): bracket the suspect SENTENCE with reserved vocabulary tokens
(<|reserved_special_token_0|>, id 128002), re-encode L0-L3, re-parse with the
UNCHANGED heads. Suspect sentence = the sentence containing the flagged slot's
attention-argmax token — marking where the pointer LOOKS (the registered
deployable-placement wrinkle). Flag source: the waist monitor's worst slot
(deployable, gold-free).

REGISTERED (both endings pinned):
  - Relay's polarity-flipped prediction STANDS: recovery <= 2% of the 460 ->
    pointers don't re-aim under input conditioning either; the population is
    DETECT-AND-ABSTAIN ONLY under current machinery; chapter closes.
  - Recovery >= 10% -> input marks CAN move what head conditioning can't —
    the week's most interesting result; the beacon graduates to a build.
  - Gold-free acceptance discipline: any forced answer counts as accepted;
    right/wrong split reported (imposter rate is a first-class column now).
  - COMPOSABILITY COLUMN (relay): monitor score on beacon-accepted parses —
    score drops on repairs = detect->beacon->re-score composes into a
    self-contained final tier; repairs without score movement = the monitor
    cannot certify its own fixes (the ratchet lesson, one level up). Caveat
    logged: centroids were built on UNMARKED states; marked-state parses may
    skew anomalous globally — the within-run right/wrong contrast is the
    honest read, not the absolute level.
  - LEDGER LINE (relay, named form for §6 on its third sighting): a selection
    criterion's jurisdiction is which property it SELECTS ON — "survived
    filter X" is evidence about detectability, not repairability.

USAGE: DEV=AMD ALG_TEST=.cache/algebra_nl_bigtest.jsonl ALG_TEST_NAME=bigtest \
           .venv/bin/python3 scripts/beacon_closing_arm.py
"""
from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

import numpy as np

from phase1_algebra_head import (  # noqa: E402
    T_ALG, L_FAC, H_TRUNK, ALG_TEST, build_params, forward, load_alg, decode,
    tokenize, sent_indices, ALG_CKPT,
)
from waist_abstention_probe import compute_fst, np_heads, slot_kind  # noqa: E402
from repair_replace_swap import solve_forced  # noqa: E402

MARK_ID = 128002   # <|reserved_special_token_0|>


_TRUNK_HOST = None       # 2026-07-13 perf: load the 2.4GB weights ONCE per
_TRUNK_JIT = None        # process (was: reload per CALL — thousands of
_TRUNK_BUF = None        # redundant loads across every eval/census/lattice)


def _trunk_host():
    global _TRUNK_HOST
    if _TRUNK_HOST is None:
        from mycelium.llama_loader import (
            attach_llama_layers, load_llama_weights, LLAMA_3_2_1B_CFG)

        class _H:
            pass
        host = _H()
        sd = load_llama_weights(os.path.join(
            _ROOT, ".cache/llama-3.2-1b-weights/model.safetensors"))
        attach_llama_layers(host, n_layers=4, sd=sd, cfg=LLAMA_3_2_1B_CFG)
        del sd
        _TRUNK_HOST = host
    return _TRUNK_HOST


def recompute_states(ids_batch):
    """L0-L3 trunk forward on marked token ids (forward-only), with the
    host CACHED per process (2026-07-13: was reloading the 2.4GB weights
    on every call — thousands of redundant loads across eval/census/
    lattice paths). Eager forward kept: a zero-arg TinyJit here RECAPTURES
    per call with this layer code (13s/batch vs ~0.5s eager, measured) —
    the jitted trunk needs the residency smoke's buffer pattern, deferred."""
    from tinygrad import Tensor, dtypes
    from mycelium.llama_loader import _rms_norm
    host = _trunk_host()
    n = len(ids_batch)
    states = np.zeros((n, T_ALG, H_TRUNK), np.float16)
    for s0 in range(0, n, 8):
        sl = slice(s0, min(s0 + 8, n))
        x = host.llama_embed[Tensor(ids_batch[sl], dtype=dtypes.int)]
        for layer in host.llama_layers:
            x = layer(x, host.llama_rope_cos, host.llama_rope_sin)
        x = _rms_norm(x, host.llama_layers[-1].ffn_norm,
                      host.llama_cfg.rms_norm_eps)
        c = x.cast(dtypes.float).realize().numpy()
        assert np.isfinite(c).all()
        states[sl] = c.astype(np.float16)
    return states


def main():
    from tinygrad import Tensor, dtypes
    from tinygrad.nn.state import safe_load

    prof = np.load(".cache/survivor_profile_bigtest.npz")
    surv = sorted(int(i) for i, s in zip(prof["idx"], prof["status"]) if s == 2)
    orc = set(int(i) for i in
              np.load(".cache/oracle_recovered_bigtest.npz")["recovered"])

    samples, states, tokmask, gold, sent = load_alg("test")
    _, ids, mask_np, offsets = tokenize(ALG_TEST)
    p = build_params(0)
    sd = safe_load(ALG_CKPT)
    for k in p:
        p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
    hd = np_heads(p)

    # centroids (train, unmarked space) for flag source + composability column
    tr_s, tr_states, tr_tok, _tg, tr_sent = load_alg("train")
    fst_tr = compute_fst(p, tr_states, tr_tok, tr_sent,
                         list(range(len(tr_s))))
    by_kind = {}
    for i in range(len(tr_s)):
        for j in range(L_FAC):
            kind = slot_kind(hd, fst_tr[i, j])
            if kind:
                by_kind.setdefault(kind, []).append(fst_tr[i, j])
    cent = {k: (lambda c: c / np.linalg.norm(c))(np.mean(v, axis=0))
            for k, v in by_kind.items()}

    def monitor_score(fst_row):
        w = 1.0
        for j in range(L_FAC):
            kind = slot_kind(hd, fst_row[j])
            if kind is None or kind not in cent:
                continue
            v = fst_row[j]
            w = min(w, float((v / max(np.linalg.norm(v), 1e-9)) @ cent[kind]))
        return 1.0 - w

    # blank parse on survivors: fat (attention) + worst monitor slot
    fst_sv = compute_fst(p, states, tokmask, sent, surv)
    worst = {}
    for r, i in enumerate(surv):
        w, wj = 1.0, -1
        for j in range(L_FAC):
            kind = slot_kind(hd, fst_sv[r, j])
            if kind is None or kind not in cent:
                continue
            v = fst_sv[r, j]
            c = float((v / max(np.linalg.norm(v), 1e-9)) @ cent[kind])
            if c < w:
                w, wj = c, j
        worst[i] = wj
    fat = {}
    for s0 in range(0, len(surv), 8):
        sl = np.array(surv[s0:s0 + 8])
        pad = 8 - len(sl)
        sl_p = np.concatenate([sl, sl[:1].repeat(pad)]) if pad else sl
        out = forward(p, Tensor(states[sl_p].astype(np.float32),
                                dtype=dtypes.float),
                      Tensor(tokmask[sl_p].astype(np.float32),
                             dtype=dtypes.float),
                      Tensor(sent[sl_p].astype(np.int32), dtype=dtypes.int))
        f = out["fat"].realize().numpy()
        for bi, i in enumerate(sl):
            fat[int(i)] = f[bi]

    # build marked sequences: bracket the suspect sentence
    marked_ids = np.zeros((len(surv), T_ALG), np.int32)
    marked_mask = np.zeros((len(surv), T_ALG), np.float32)
    marked_sent = np.zeros((len(surv), T_ALG), np.int32)
    skipped = []
    for r, i in enumerate(surv):
        ntk = int(mask_np[i].sum())
        wj = worst[i]
        att = fat[i][wj][:ntk] if wj >= 0 else fat[i].mean(0)[:ntk]
        t_star = int(np.argmax(att))
        s_id = int(sent[i][t_star])
        span = np.where(sent[i][:ntk] == s_id)[0]
        a, b = int(span[0]), int(span[-1]) + 1
        if ntk + 2 > T_ALG:
            skipped.append(i)
            marked_ids[r, :ntk] = ids[i, :ntk]
            marked_mask[r, :ntk] = 1.0
            marked_sent[r] = sent[i]
            continue
        new = np.concatenate([ids[i, :a], [MARK_ID], ids[i, a:b], [MARK_ID],
                              ids[i, b:ntk]])
        marked_ids[r, :len(new)] = new
        marked_mask[r, :len(new)] = 1.0
        ns = np.concatenate([sent[i, :a], [s_id], sent[i, a:b], [s_id],
                             sent[i, b:ntk]])
        marked_sent[r, :len(ns)] = ns
    if skipped:
        print(f"[beacon] {len(skipped)} sequences at token budget — "
              f"unmarked passthrough (counted, not dropped)")

    m_states = recompute_states(marked_ids)

    # re-parse marked states with the UNCHANGED heads
    rec_right = rec_wrong = unforced = 0
    right_idx, scores_right, scores_wrong = [], [], []
    fst_mk = compute_fst(p, m_states, marked_mask, marked_sent,
                         list(range(len(surv))))
    for s0 in range(0, len(surv), 8):
        sl = np.arange(s0, min(s0 + 8, len(surv)))
        pad = 8 - len(sl)
        sl_p = np.concatenate([sl, sl[:1].repeat(pad)]) if pad else sl
        out = forward(p, Tensor(m_states[sl_p].astype(np.float32),
                                dtype=dtypes.float),
                      Tensor(marked_mask[sl_p].astype(np.float32),
                             dtype=dtypes.float),
                      Tensor(marked_sent[sl_p].astype(np.int32),
                             dtype=dtypes.int))
        o = {k: out[k].realize().numpy() for k in
             ("pres", "ftype", "op", "islit", "dig", "args", "res", "query")}
        for bi, r in enumerate(sl):
            i = surv[int(r)]
            facs, q_pred = decode({k: o[k][bi] for k in o})
            a = solve_forced(facs, q_pred, samples[i])
            if a is None:
                unforced += 1
                continue
            gold_ans = samples[i]["solution"][samples[i]["query_var"]]
            sc = monitor_score(fst_mk[int(r)])
            if a == gold_ans:
                rec_right += 1
                right_idx.append(i)
                scores_right.append(sc)
            else:
                rec_wrong += 1
                scores_wrong.append(sc)

    n = len(surv)
    print(f"\n[beacon] survivors={n} | forced-accepted "
          f"{rec_right + rec_wrong} (RIGHT {rec_right} / WRONG {rec_wrong}) "
          f"| unforced {unforced}")
    print(f"  RECOVERY: {rec_right}/{n} = {rec_right / n:.3f}  "
          f"(oracle NEURAL ceiling 13.9%; bars: <=2% flipped-prediction "
          f"confirmed / >=10% beacon graduates)")
    in396 = sum(1 for i in right_idx if i not in orc)
    print(f"  right recoveries in the hard-396: {in396}")
    if scores_right or scores_wrong:
        mr = np.mean(scores_right) if scores_right else float("nan")
        mw = np.mean(scores_wrong) if scores_wrong else float("nan")
        print(f"  COMPOSABILITY: monitor score on beacon-accepted — right "
              f"{mr:.3f} vs wrong {mw:.3f} (within-run contrast is the honest "
              f"read; centroids are unmarked-space)")
    print(f"\n  ENDINGS: <=2% -> detect-and-abstain-only, chapter CLOSES; "
          f">=10% -> input marks move what conditioning can't.")


if __name__ == "__main__":
    main()
