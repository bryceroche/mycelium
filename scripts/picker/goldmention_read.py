"""goldmention_read.py -- THE GOLD-MENTION CENSUS (pin 1 of THE SEEDED READINGS,
ledger docs/phase1_skeleton_spec.md 2026-10-12 13:10).

Question: on wild's GIVEN slots that are WRONG at the final decode, what attention
mass and rank does the GOLD MENTION (the token span the gold binding points at)
receive at breath 0 (the grounding read, slot->token attention over 256 tokens)?
Registered prediction: gold mention at rank 2, >=5% mass, on >=30% of wrong slots.
Kill (sampling path): <10% of wrong slots have the gold mention in the top 3.

Zero-GPU for the POOLED (head-averaged) numbers: breath 0 is already present in the
banked .cache/membrane_raw_wild_PMS8_241.npz (K_B axis, kb=0) for the full n=311 wild
holdout -- fat_all's docstring (scripts/membrane_scale.py) says this array IS the
per-breath HEAD-MEAN slots->tokens attention (`at.mean(1)` in phase1_algebra_head.py's
bank() closure), so no regeneration is needed for the pooled read.

The per-head (8 x 64-dim) attention is NOT banked anywhere (bank() returns only
`at.mean(1)`; the pre-mean `at` is a local variable, never surfaced) and this task
forbids editing scripts/phase1_algebra_head.py to add a tap. Mode `collect-head`
below gets it WITHOUT touching that file: it monkeypatches tinygrad's Tensor.softmax
from the outside, on a CPU-only re-run of membrane_scale.py's own collect() cycle
(verbatim non-ALT3 two-pass loop: pass 1 unmasked parse -> build_slot_masks +
alt2_fact_buf -> pass 2 masked, the SAME calls, SAME order, SAME env as the banked
PMS8_241 npz), capturing the FIRST softmax call per pass-2 forward() whose shape is
(8, N_HEADS, L_TOT, T_ALG) = (8, 8, 32, 256) -- that shape is unique to the breath-0
slot->token bank() call (`fst, fat = bank(p["fq"], L_TOT, pbias=_pb, flat=...)`,
phase1_algebra_head.py ~9255); the var/query banks have nq=1, and every later breath's
own re-grounding call recurs with the identical shape but is skipped once the
per-forward-call flag is set. Trimmed to L_FAC=24 (scratch rows dropped, matching
_fed_core). Run for n=64 rows only (the task's instructed sample size for this mode).
A parity check re-derives the head-MEAN from the captured per-head tensor and diffs
it against the banked pooled fat_mass for the same 64 rows -- this is the correctness
gate for trusting the capture (same ckpt, same env, same rows must give the same
pooled number two different ways).

Usage:
  DEV=CPU .venv/bin/python3 scripts/picker/goldmention_read.py collect-head --rows 64
      writes .cache/goldmention_perhead_n64.npz (CPU-only forward passes, no GPU lock).
  .venv/bin/python3 scripts/picker/goldmention_read.py report
      pure numpy; writes .cache/goldmention_report.txt and .json.
"""
import os
import sys
import json
import argparse

sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np

CKPT = ".cache/sharp_PMS8_241.safetensors"
TAG = "PMS8_241"
RAW_BANKED = f".cache/membrane_raw_wild_{TAG}.npz"       # pooled (head-mean) breath 0..6, n=311 -- READ ONLY, never overwritten
OK_BANK = f".cache/ps_legal_wild_{TAG}.npz"               # the per-(row,slot) correctness bank (n=2051 GIVEN slots)
PERHEAD_OUT = ".cache/goldmention_perhead_n64.npz"        # NEW artifact (goldmention_ prefix), n=64

_FAM = {
    "DEV": "CPU", "ALG2": "1", "ALG_FTYPES": "9", "ALG_DUP": "1",
    "ALG_HW": "512", "ALG_WIDE": "1", "ALG_BREATH": "7",
    "ALG_NOTEBOOK": "1", "ALG_SIXWAVE": "1", "NB_PERSLOT": "1",
    "ALG_BINDBUS": "7", "ALG_BIND_D": "512",
    "BIND_CODES": ".cache/bindbus_codes512.npz",
    "ALG_BUSGARAGE": "2", "ALG_SHELF_CIRCLE": "2", "ALG_ALTMASK": "1",
    "ALG_ALT21": "1", "ALG_ALT2": "1", "ALG_MASKHEAD": "1",
    "ALG_FED": "1", "ALG_POLAR": "1", "ALG_POLAR_D": "128",
    "ALG_POLAR_EM": "0.1",
    "ALG_POLAR_D_INIT": ".cache/polar_waist_init_d128u.npz",
    "ALG_PRUNE": "pforms,s4,fednl0,lane2", "ALG_SLOT_ALL": "1",
    "ALG_STELLAR": "2", "ALG_CLOCK_CANON": "1", "SC_EVAL": "0",
    "ALG_TEST": ".cache/wild_admitted_holdout.jsonl", "ALG_TEST_NAME": "wildhold",
    "ALG_MINE_BREATHS": "1",   # THE TAP: forward() hands back out["fat_all"] (used here for the parity check)
    # THE CHASSIS (S8, .cache/breathstop_gate.sh): PMS8_241 is ALG_ROUTER=2 + the role
    # pointer -- membrane_rack.py's own _FAM deliberately omits this (its docstring:
    # "$SURF8 ... NOT defaulted here"; its callers export it). Omitting it is a hard
    # ckpt-key mismatch (verified: build_params() lacks W_rk2/W_role/W_rq2/r_gain
    # without it) -- NOT optional for this checkpoint. BIND_CODES is the ROLE bank
    # (512r), not the plain one membrane_rack.py defaults to; S8 pairs the two, per
    # that script's own ALG_PTR_SURF/BIND_CODES audit guard.
    "ALG_ROUTER": "2", "R_GAIN_INIT": "1.0", "ALG_FREEZE": "r_gain", "ALG_ROUTER_PTR": "0.0",
    "ALG_SPAN_ALL": "1", "ALG_SPAN_ARGS": "1", "ALG_SPAN_OP": "1", "ALG_SPAN_RCUE": "1",
    "ALG_SPAN_ARCUE": "1", "ALG_PTR_SURF": "role:add:2.0",
}
_FAM["BIND_CODES"] = ".cache/bindbus_codes512r.npz"   # overrides the plain-bank default above, per S8


def _set_env():
    for k, v in _FAM.items():
        os.environ.setdefault(k, v)
    assert os.environ.get("DEV") == "CPU", f"DEV={os.environ.get('DEV')!r}: this script is CPU-only by brief"


# ===========================================================================
# collect-head: CPU-only re-run, per-head breath-0 attention via a softmax tap
# ===========================================================================

def collect_head(n_rows):
    _set_env()
    from tinygrad import Tensor, dtypes
    from tinygrad.nn.state import safe_load
    import phase1_algebra_head as H
    from phase1_algebra_head import (build_params, forward, load_alg,
                                      build_slot_masks, alt2_fact_buf,
                                      K_VARS, L_FAC, L_TOT, T_ALG, N_HEADS)

    vs, vst, vtk, vg, vse = load_alg("test")
    n = min(len(vs), n_rows)
    K_B = int(os.environ.get("ALG_BREATH", "1"))
    print(f"[goldmention] collect-head fixture=wild n={n}/{len(vs)} K_B={K_B} ckpt={CKPT}", flush=True)

    p = build_params(0)
    sd = safe_load(CKPT)
    assert set(sd.keys()) == set(p.keys()), (sorted(set(sd) - set(p))[:4], sorted(set(p) - set(sd))[:4])
    for k in p:
        p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()

    # THE SOFTMAX TAP -- captures the FIRST (8, N_HEADS, L_TOT, T_ALG)-shaped softmax
    # output per pass-2 forward() call (reset just before that call below). This is
    # external instrumentation (monkeypatch), NOT an edit to phase1_algebra_head.py.
    _orig_softmax = Tensor.softmax
    _tap = {"armed": False, "out": None, "hits": 0}
    _target_shape = (N_HEADS, L_TOT, T_ALG)

    def _patched_softmax(self, axis=-1, dtype=None):
        r = _orig_softmax(self, axis, dtype) if dtype is not None else _orig_softmax(self, axis)
        if _tap["armed"] and tuple(r.shape[1:]) == _target_shape:
            _tap["out"] = r.realize().numpy().astype(np.float32)
            _tap["armed"] = False
            _tap["hits"] += 1
        return r

    Tensor.softmax = _patched_softmax

    perhead = np.zeros((n, N_HEADS, L_FAC, T_ALG), np.float16)
    pooled_check = np.zeros((n, L_FAC, T_ALG), np.float16)

    try:
        for s0 in range(0, n, 8):
            sl = np.arange(s0, min(s0 + 8, n))
            pad = 8 - len(sl)
            sl_p = np.concatenate([sl, sl[:1].repeat(pad)]) if pad else sl
            ts = Tensor(np.ascontiguousarray(vst[sl_p]), dtype=dtypes.half)
            tk = Tensor(vtk[sl_p].astype(np.float32), dtype=dtypes.float)
            se = Tensor(vse[sl_p].astype(np.int32), dtype=dtypes.int)

            # pass 1 (unmasked parse) -- NOT tapped (armed stays False)
            o0 = forward(p, ts, tk, se, tree=None)
            onp0 = {k: o0[k].realize().numpy() for k in ("fat", "args", "res")}
            mk = build_slot_masks(onp0, vse[sl_p].astype(np.int32))
            mk_t = Tensor(mk, dtype=dtypes.float)
            _nv = np.array([vs[int(i)].get("n_vars", K_VARS) for i in sl_p])
            _ma = np.array([vs[int(i)].get("m", 0) for i in sl_p])
            _ka = ("pres", "ftype", "op", "dig") + (("dup",) if "dup" in o0 else ())
            _oa = {**onp0, **{k: o0[k].realize().numpy() for k in _ka}}
            fb = alt2_fact_buf(_oa, vse[sl_p].astype(np.int32), _nv, _ma)
            fact_t = Tensor(fb, dtype=dtypes.float)

            # pass 2 (masked walk) -- the official read. Arm the tap for its FIRST
            # matching-shape softmax call: the breath-0 grounding bank() call.
            _tap["armed"] = True
            o = forward(p, ts, tk, se, slot_mask=mk_t, fact_buf=fact_t, tree=None)
            fat_all0 = o["fat_all"][0].realize().numpy()   # (8, L_FAC, T_ALG) -- the official pooled breath-0 read

            assert _tap["out"] is not None, "softmax tap never fired -- shape assumption wrong, do not trust this run"
            at0 = _tap["out"][:, :, :L_FAC, :]   # (8, N_HEADS, L_TOT, T) -> trim scratch rows to L_FAC (==_fed_core)

            perhead[sl] = at0[:len(sl)].astype(np.float16)
            pooled_check[sl] = fat_all0[:len(sl)].astype(np.float16)

            print(f"[goldmention] {s0 + len(sl)}/{n} rows (tap hits so far: {_tap['hits']})", flush=True)
    finally:
        Tensor.softmax = _orig_softmax

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(H.TOKENIZER_JSON)
    ids_all = np.zeros((n, T_ALG), np.int32)
    for i in range(n):
        enc = tok.encode(vs[int(i)]["text"])
        L = min(len(enc.ids), T_ALG)
        ids_all[i, :L] = enc.ids[:L]

    np.savez(PERHEAD_OUT, perhead=perhead, pooled_check=pooled_check,
             ids=ids_all, tokmask=vtk[:n].astype(np.uint8), sent=vse[:n].astype(np.int8),
             g_presence=vg["presence"][:n], g_ftype=vg["ftype"][:n], g_digits=vg["digits"][:n],
             n_rows=np.array([n]))
    print(f"[goldmention] wrote {PERHEAD_OUT} (n={n}, tap fired {_tap['hits']} times, "
          f"expected {((n + 7) // 8)})", flush=True)


# ===========================================================================
# report: pure numpy -- gold-mention mass + rank, pooled (n=311) and per-head (n=64)
# ===========================================================================

def _load_ok():
    ps = np.load(OK_BANK)
    return {(int(r), int(c)): bool(o) for r, c, o in zip(ps["rows"], ps["slots"], ps["ok"])}


def _digit_runs(tok, ids, T):
    """Verbatim from scripts/membrane_scale.py (reused by import below where possible;
    inlined once here only because this module must not import GPU-touching deps at
    report time -- membrane_scale.py itself is pure-numpy/report-safe, so we DO import
    it below for the band helpers; this copy exists only as a safety net if that import
    is unavailable in some run context). Not used when the import succeeds."""
    runs = []
    i = 0
    while i < T:
        s = tok.decode([int(ids[i])]).strip()
        if s.isdigit() and s != "":
            j = i + 1
            buf = s
            while j < T:
                s2 = tok.decode([int(ids[j])]).strip()
                if s2.isdigit() and s2 != "":
                    buf += s2
                    j += 1
                else:
                    break
            try:
                v = int(buf)
            except ValueError:
                v = None
            if v is not None:
                runs.append((i, j, v))
            i = j
        else:
            i += 1
    return runs


def _gold_mass_and_rank(m_row, real, occ_tok):
    """m_row: (T,) attention mass for one slot (any single head, or the pooled mean).
    real: (T,) bool, the row's real-token mask. occ_tok: sorted list of token indices
    that are the gold mention (every maximal digit-run decoding to the gold value).
    Returns (mass_share, rank) where mass_share is the gold tokens' summed mass over
    total REAL mass (matches membrane_scale's mass_band bi=0 normalization exactly),
    and rank is the 1-indexed rank (among real tokens, ties broken by strict-greater
    count) of the BEST (highest-mass) gold token -- the most favourable reading of
    "where does the gold mention sit in the attention order"."""
    mreal = m_row[real]
    tot = float(mreal.sum())
    if tot <= 0:
        return None, None
    gold_vals = m_row[occ_tok]
    best = float(gold_vals.max())
    mass_share = float(m_row[occ_tok].sum()) / tot
    rank = 1 + int((mreal > best).sum())
    return mass_share, rank


def _quantiles(xs, qs=(0.1, 0.25, 0.5, 0.75, 0.9)):
    if not xs:
        return {f"p{int(q * 100)}": None for q in qs}
    a = np.asarray(xs, dtype=np.float64)
    return {f"p{int(q * 100)}": float(np.quantile(a, q)) for q in qs}


def _collect_cells(ids, tokmask, sent, pres, ftype, gdig, mass_src, ok, n, L_FAC, T_ALG,
                    tok, head_axis=None):
    """mass_src: either (n, L_FAC, T) [pooled] or (n, H, L_FAC, T) [per-head, head_axis
    picked outside the loop per head]. Returns list of dict rows: i,j,key(right/wrong),
    mass,rank,same_sent(bool via breath-0 argmax band), occ_sent set size (diagnostic)."""
    import membrane_scale as MS
    out = []
    n_nomatch = n_notjoined = 0
    for i in range(n):
        dec = MS._decode_tokens(tok, ids[i], T_ALG)
        clause_id = MS._segment_ids(dec, tokmask[i], sent[i], MS.CONJ_CLAUSE)
        mention_id = MS._segment_ids(dec, tokmask[i], sent[i],
                                      MS.CONJ_MENTION_EXTRA | MS.PUNCT_MENTION_EXTRA | MS.VERBISH)
        runs = MS._digit_runs(tok, ids[i], T_ALG)
        real = tokmask[i] > 0
        for j in range(L_FAC):
            if pres[i, j] < 0.5 or int(ftype[i, j]) != 1:
                continue
            cell = ok.get((i, j))
            if cell is None:
                n_notjoined += 1
                continue
            v = int("".join(str(int(x)) for x in gdig[i, j]))
            matches = [(a, b, val) for a, b, val in runs if val == v]
            if not matches:
                n_nomatch += 1
                continue
            occ_tok = sorted({t for a, b, val in matches for t in range(a, b)})
            occ_sent = {int(sent[i, t]) for t in occ_tok}
            key = "right" if cell else "wrong"
            m_row = mass_src[i, j] if head_axis is None else mass_src[i, head_axis, j]
            mass, rank = _gold_mass_and_rank(m_row, real, occ_tok)
            if mass is None:
                continue
            am = int(np.where(real, m_row, -1e30).argmax())
            band = MS._band_arrays(None, occ_tok, mention_id,
                                    {int(mention_id[t]) for t in occ_tok}, clause_id,
                                    {int(clause_id[t]) for t in occ_tok}, sent[i], occ_sent, tokmask[i])
            same_sent = bool(int(band[am]) != 4)   # argmax band 0-3 (own sentence) vs 4 (other sentence)
            out.append(dict(i=i, j=j, key=key, mass=mass, rank=rank, same_sent=same_sent))
    return out, n_nomatch, n_notjoined


def _summarize(cells, key):
    sub = [c for c in cells if c["key"] == key]
    n = len(sub)
    if n == 0:
        return dict(n=0)
    ranks = [c["rank"] for c in sub]
    masses = [c["mass"] for c in sub]
    frac_rank = {
        "rank1": sum(r == 1 for r in ranks) / n,
        "rank2": sum(r == 2 for r in ranks) / n,
        "rank3": sum(r == 3 for r in ranks) / n,
        "rank_gt3": sum(r > 3 for r in ranks) / n,
        "top3": sum(r <= 3 for r in ranks) / n,
    }
    frac_mass_ge5 = sum(m >= 0.05 for m in masses) / n
    return dict(n=n, **frac_rank, frac_mass_ge5=frac_mass_ge5,
                mass_quantiles=_quantiles(masses), mean_mass=float(np.mean(masses)),
                mean_rank=float(np.mean(ranks)))


def report():
    import membrane_scale as MS  # noqa: F401 -- import-time side-effect-free (report-safe module)
    import phase1_algebra_head as H
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(H.TOKENIZER_JSON)
    T_ALG = H.T_ALG
    L_FAC = H.L_FAC

    ok = _load_ok()
    lines = []
    P = lines.append
    out = {}

    # ---- PART 1: pooled (head-averaged), n=311, zero new compute (breath 0 already banked) ----
    raw = np.load(RAW_BANKED)
    n311 = raw["ids"].shape[0]
    fat0_pooled = raw["fat_mass"][:, 0, :, :].astype(np.float32)   # (n311, L_FAC, T) -- breath 0, head-mean (banked)
    cells_pooled, nomatch1, notj1 = _collect_cells(
        raw["ids"], raw["tokmask"], raw["sent"], raw["g_presence"], raw["g_ftype"], raw["g_digits"],
        fat0_pooled, ok, n311, L_FAC, T_ALG, tok)
    P("=" * 78)
    P(f"PART 1 -- POOLED (head-averaged) breath-0 slot->token attention, n={n311} (full wild holdout)")
    P(f"source: {RAW_BANKED} (fat_mass[:,0,:,:] -- already banked, zero new compute)")
    P(f"no-textual-match={nomatch1}  not-in-correctness-bank={notj1}")
    P("-" * 78)
    sw = _summarize(cells_pooled, "wrong"); sr = _summarize(cells_pooled, "right")
    out["pooled_n311"] = dict(wrong=sw, right=sr)
    for key, s in (("WRONG", sw), ("RIGHT", sr)):
        P(f"{key} (n={s['n']}):")
        if s["n"]:
            P(f"  rank:  1={s['rank1']:.3f}  2={s['rank2']:.3f}  3={s['rank3']:.3f}  >3={s['rank_gt3']:.3f}  (top3={s['top3']:.3f})")
            P(f"  mass:  mean={s['mean_mass']:.3f}  >=5%={s['frac_mass_ge5']:.3f}  quantiles={s['mass_quantiles']}")

    wrong_twin = [c for c in cells_pooled if c["key"] == "wrong" and c["same_sent"]]
    wrong_other = [c for c in cells_pooled if c["key"] == "wrong" and not c["same_sent"]]
    P("-" * 78)
    P("BY CLASS (breath-0 argmax band: same-sentence/'twin' vs other-sentence), WRONG slots only:")
    for name, sub in (("same-sentence (twin)", wrong_twin), ("other-sentence", wrong_other)):
        n = len(sub)
        if n == 0:
            P(f"  {name}: n=0"); continue
        ranks = [c["rank"] for c in sub]; masses = [c["mass"] for c in sub]
        top3 = sum(r <= 3 for r in ranks) / n
        P(f"  {name}: n={n}  rank1={sum(r==1 for r in ranks)/n:.3f}  top3={top3:.3f}  "
          f"mean_rank={np.mean(ranks):.2f}  mean_mass={np.mean(masses):.3f}")
        out[f"pooled_n311_class_{'twin' if 'twin' in name else 'other'}"] = dict(
            n=n, rank1=sum(r == 1 for r in ranks) / n, top3=top3,
            mean_rank=float(np.mean(ranks)), mean_mass=float(np.mean(masses)))

    # ---- verdict against the registered prediction / kill ----
    P("=" * 78)
    P("VERDICT (pooled, n=311 -- the authoritative read; per-head below is a secondary n=64 check)")
    pred_ok = sw["n"] > 0 and sw["rank2"] >= 0.0  # placeholder; real check below
    frac_rank2_ge5 = sum(1 for c in cells_pooled if c["key"] == "wrong" and c["rank"] == 2 and c["mass"] >= 0.05) / max(sw["n"], 1)
    P(f"  registered prediction: gold mention at RANK 2 with >=5% mass on >=30% of wrong slots.")
    P(f"  measured: rank==2 AND mass>=5% on {frac_rank2_ge5:.3f} of wrong slots (n={sw['n']}) -> "
      f"{'PREDICTION HOLDS' if frac_rank2_ge5 >= 0.30 else 'PREDICTION REFUTED'}")
    P(f"  kill criterion: <10% of wrong slots have gold mention in top 3.")
    P(f"  measured top3 share = {sw['top3']:.3f} -> {'KILL FIRES' if sw['top3'] < 0.10 else 'kill does not fire'}")
    out["verdict"] = dict(frac_rank2_mass5=frac_rank2_ge5, prediction_holds=bool(frac_rank2_ge5 >= 0.30),
                           top3_share=sw["top3"], kill_fires=bool(sw["top3"] < 0.10))

    # ---- PART 2: per-head, n=64 (if collected) ----
    if os.path.exists(PERHEAD_OUT):
        ph = np.load(PERHEAD_OUT)
        n64 = int(ph["n_rows"][0])
        perhead = ph["perhead"].astype(np.float32)   # (n64, 8, L_FAC, T)
        pooled64 = ph["pooled_check"].astype(np.float32)
        # parity: does averaging the captured per-head tensor reproduce the banked pooled read?
        recon_pooled = perhead.mean(axis=1)
        diff = np.abs(recon_pooled - pooled64)
        banked64 = fat0_pooled[:n64]
        diff_vs_banked = np.abs(recon_pooled - banked64)
        P("=" * 78)
        P(f"PART 2 -- PER-HEAD breath-0 attention, n={n64} (regenerated: {PERHEAD_OUT})")
        P(f"PARITY CHECK: mean(per-head)-vs-this-run's-own-pooled-tap max|diff|={diff.max():.5f} "
          f"mean|diff|={diff.mean():.6f}; mean(per-head)-vs-BANKED-n311-pooled max|diff|={diff_vs_banked.max():.5f} "
          f"mean|diff|={diff_vs_banked.mean():.6f}  (fp16 storage, ~1e-3 expected if the capture is correct)")
        out["parity"] = dict(n=n64, max_diff_self=float(diff.max()), mean_diff_self=float(diff.mean()),
                              max_diff_vs_banked=float(diff_vs_banked.max()), mean_diff_vs_banked=float(diff_vs_banked.mean()))

        cells_pooled64, _, _ = _collect_cells(
            ph["ids"], ph["tokmask"], ph["sent"], ph["g_presence"], ph["g_ftype"], ph["g_digits"],
            pooled64, ok, n64, L_FAC, T_ALG, tok)
        sw64 = _summarize(cells_pooled64, "wrong")
        P(f"  pooled-on-these-64-rows, WRONG (n={sw64.get('n',0)}): "
          f"{ {k: v for k, v in sw64.items() if k != 'mass_quantiles'} if sw64.get('n') else 'n=0'}")
        out["pooled_n64_wrong"] = sw64

        per_head_summaries = []
        for h in range(perhead.shape[1]):
            cells_h, _, _ = _collect_cells(
                ph["ids"], ph["tokmask"], ph["sent"], ph["g_presence"], ph["g_ftype"], ph["g_digits"],
                perhead[:, h], ok, n64, L_FAC, T_ALG, tok)
            per_head_summaries.append(_summarize(cells_h, "wrong"))
        P("-" * 78)
        P("  PER-HEAD (WRONG slots, n=64 rows): rank1 / top3 / mean_mass by head:")
        for h, s in enumerate(per_head_summaries):
            if s.get("n"):
                P(f"    head {h}: n={s['n']}  rank1={s['rank1']:.3f}  top3={s['top3']:.3f}  mean_mass={s['mean_mass']:.3f}")
        out["per_head_wrong"] = per_head_summaries
        P("-" * 78)
        P("  WHICH ONE THE DECODE USES: bank()'s value read `st = (at @ vh)` (phase1_algebra_head.py "
          "~6277) consumes the FULL per-head `at` BEFORE any averaging -- every head's own attention "
          "weights gate its own value slice, concatenated then projected (attn_wo) into the slot state "
          "that args/res/decode are computed from. The pooled `fat`/`fat_all` (`at.mean(1)`) returned "
          "alongside it is a READOUT TAP ONLY (feeds the clock/NL-tap/mask-cooker diagnostics and this "
          "census) -- it is NOT what the decode's binding is computed from. Reported here for honesty, "
          "not because it is the mechanism.")
    else:
        P("=" * 78)
        P(f"PART 2 -- per-head NOT collected this run ({PERHEAD_OUT} absent). Run "
          f"'collect-head --rows 64' first (CPU-only, no GPU lock) if a per-head read is wanted.")
        out["per_head"] = None

    txt = "\n".join(lines)
    print(txt)
    with open(".cache/goldmention_report.txt", "w") as f:
        f.write(txt + "\n")
    with open(".cache/goldmention_report.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print("\n[goldmention] wrote .cache/goldmention_report.txt / .json", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c1 = sub.add_parser("collect-head")
    c1.add_argument("--rows", type=int, default=64)
    sub.add_parser("report")
    args = ap.parse_args()
    if args.cmd == "collect-head":
        collect_head(args.rows)
    else:
        report()
