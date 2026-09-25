"""membrane_scale.py — THE MEMBRANE CENSUS BY SCALE (analyst, 2026-09-24,
ledger "the pyramid staging"). A NEW script; scripts/membrane_census.py and
scripts/phase1_algebra_head.py are untouched.

Question: of the GIVEN slots on wild that are WRONG under the banked masked
read (.cache/ps_legal_wild_PMS8_241.npz, ok==0) and of those that are RIGHT,
where does the head's per-breath slots<-tokens attention land relative to the
gold binding, staged on a token -> mention -> clause -> sentence -> other-
sentence pyramid? Both the argmax location and the attention-MASS
distribution across the pyramid are reported, by breath (final breath is the
headline; the trajectory across all breaths is also shown).

This is scripts/membrane_census.py's method (two-pass loop_val cycle,
ALG_MINE_BREATHS=1 tap for out["fat_all"], gold value location = digit runs
matching the fixture's own slot-keyed g_digits — read that script's docstring
for why: the positional law's two conventions mean factors[j] != slot j on
mint, g_digits sidesteps it; wild has no hand mention-spans either, same as
there) re-run against the PMS8_241 checkpoint/env (not PM35_scratch_241),
plus a NEW pyramid-staging layer that membrane_census.py does not have:
mention/clause segmentation, and full attention-mass capture (not just
argmax) so the report can weigh mass, not only count the arg-max cell.

Two modes (env MS_MODE):
  collect (default) — GPU pass. Runs the checkpoint's own two-pass loop_val
    cycle (pass-1 unmasked parse -> build_slot_masks + alt2_fact_buf ->
    pass-2 masked) exactly as membrane_census.py does, with
    ALG_MINE_BREATHS=1 so forward() hands back out["fat_all"] (per-breath
    head-mean slots<-tokens attention, FED-trimmed to L_FAC). Unlike
    membrane_census.py this ALSO keeps the full per-breath attention
    DISTRIBUTION over tokens (float16; ~26MB for 311 rows x 7 breaths x 24
    slots x 256 tokens), not just the argmax, so mass-weighted bands can be
    computed later. Writes .cache/membrane_raw_wild_PMS8_241.npz — a NEW
    file; .cache/membrane_raw_wild.npz / membrane_raw_mint.npz (the
    PM35_scratch_241 intermediates) are never touched.
  report — pure numpy, no GPU/tinygrad import (only `phase1_algebra_head` is
    imported, for TOKENIZER_JSON/T_ALG, and that module tolerates import
    with no env set). Joins the collected attention against the banked
    per-slot correctness file, builds the pyramid tables, writes
    .cache/membrane_scale_PMS8_241.txt and prints it.

DEFINITIONS CHOSEN FOR THE PYRAMID (no hand mention/clause annotation exists
on wild, same finding as membrane_census.py's 2026-09-19 census):
  - gold binding location for a GIVEN slot j, value v: identical to
    membrane_census.py — every maximal digit-only-token run in the row's OWN
    tokenization decoding to exactly str(v); multiple occurrences unioned.
    Slots with no textual occurrence are NO_MATCH, excluded from the pyramid,
    counted separately.
  - CLAUSE (the prompt's fallback finest scale, used here as the coarser of
    the two sub-sentence scales): a sentence (the fixture's own `sent` token
    index, from tokenize()'s sentence splitter) is cut at every comma (","),
    semicolon (";"), "and", or "but" token, per the prompt's own recipe.
    Each cut starts a new clause segment; the cutting token itself is
    assigned to the segment it OPENS (an arbitrary but immaterial choice —
    gold values are digit tokens, never comma/and/but tokens).
  - MENTION (the finer scale, offered honestly as fragile): the same cut,
    with the boundary set WIDENED to also break on periods, "or", and a
    fixed verb/copula/connective stoplist (is/are/was/were/has/have/costs/
    spent/gives/bought/sold/needs/wants/makes/paid/receives/left/remains/
    earns/uses/takes/adds/gets/found/totals/equals/if/then/so/because/after/
    before/each/per — see VERBISH below). This is a hand-built heuristic
    lexicon, not a parser; it under- and over-segments on sentences whose
    verb is a form not in the list, or whose clause has no verb at all
    (fragment appositions). Read the MENTION numbers as a lower bound on how
    fine a real mention-span annotation would cut, not a ground truth.
    Because MENTION's boundary set is a superset of CLAUSE's (plus period,
    which is already implied by the sentence cut), every mention segment is
    a strict subset of tokens of its enclosing clause segment by
    construction — the pyramid nests: token subset-of mention subset-of
    clause subset-of sentence.
  - Per-breath ARGMAX band = the finest band containing the breath's argmax
    token (token > mention > clause > sentence > other-sentence, with
    sentence distance = min over gold occurrences' sentences).
  - Per-breath MASS-WEIGHTED bands = the full softmax attention distribution
    over real tokens (renormalized over tokmask==1, since fat is already a
    softmax and padding mass is near-zero but not exactly zero at fp16),
    binned into the same 5 bands per token, summed. This is the new number
    membrane_census.py never computed (it only ever stored the argmax).
  - Correctness: WILD masked read only (.cache/ps_legal_wild_PMS8_241.npz,
    LV_LEGAL=num) — the mission's own headline file. No open-read table.
  - "final breath" = fat_all[-1], same convention as membrane_census.py
    (fat_all[0] == out["fat"], the loss-graded tensor).
"""
import os
import sys

sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')

import numpy as np  # module-level: _segment_ids/_band_arrays need it outside report()/collect()

MODE = os.environ.get("MS_MODE", "collect")
CKPT = os.environ.get("MS_CKPT", ".cache/sharp_PMS8_241.safetensors")
RAW_PATH = ".cache/membrane_raw_wild_PMS8_241.npz"
OUT_TXT = ".cache/membrane_scale_PMS8_241.txt"
PS_LEGAL = ".cache/ps_legal_wild_PMS8_241.npz"
WILD_JSONL = ".cache/wild_admitted_holdout.jsonl"

# ---------------------------------------------------------------------
# THE PMS8_241 FAMILY ENV, verbatim from the mission brief (== .cache/
# pms8_chain.sh's $FAM + $SURF8, the exact env its masked wild read
# (.cache/read_legal_wild_PMS8_241.log -> ps_legal_wild_PMS8_241.npz) ran
# under — note $TR (ALG_TRAIN/LR/SEED/...) is NOT part of that read's env
# in the chain script; only $FAM + fixture + $SURF8 + read flags are, so
# ALG_TRAIN is deliberately left at its default here, matching the actual
# read, not membrane_census.py's collect (which sets it — that census was
# for a different checkpoint/lineage, PM35_scratch_241, predating SURF8).
# ---------------------------------------------------------------------
_FAM = {
    "DEV": "PCI+AMD", "ALG2": "1", "ALG_FTYPES": "9", "ALG_DUP": "1",
    "ALG_HW": "512", "ALG_WIDE": "1", "ALG_BREATH": "7",
    "ALG_NOTEBOOK": "1", "ALG_SIXWAVE": "1", "NB_PERSLOT": "1",
    "ALG_BINDBUS": "7", "ALG_BIND_D": "512",
    "BIND_CODES": ".cache/bindbus_codes512r.npz",
    "ALG_BUSGARAGE": "2", "ALG_SHELF_CIRCLE": "2", "ALG_ALTMASK": "1",
    "ALG_ALT21": "1", "ALG_ALT2": "1", "ALG_MASKHEAD": "1",
    "ALG_FED": "1", "ALG_POLAR": "1", "ALG_POLAR_D": "128",
    "ALG_POLAR_EM": "0.1",
    "ALG_POLAR_D_INIT": ".cache/polar_waist_init_d128u.npz",
    "ALG_PRUNE": "pforms,s4,fednl0,lane2", "ALG_SLOT_ALL": "1",
    "ALG_STELLAR": "2", "ALG_CLOCK_CANON": "1", "SC_EVAL": "0",
    "ALG_ROUTER": "2", "R_GAIN_INIT": "1.0", "ALG_FREEZE": "r_gain",
    "ALG_ROUTER_PTR": "0.0", "ALG_SPAN_ALL": "1", "ALG_SPAN_ARGS": "1",
    "ALG_SPAN_OP": "1", "ALG_SPAN_RCUE": "1", "ALG_SPAN_ARCUE": "1",
    "ALG_PTR_SURF": "role:add:2.0",
    "ALG_TEST": WILD_JSONL, "ALG_TEST_NAME": "wildhold",
    # THE TAP: forward() hands back out["fat_all"] (read-only)
    "ALG_MINE_BREATHS": "1",
}


def _set_env():
    for k, v in _FAM.items():
        os.environ.setdefault(k, v)


# =======================================================================
# COLLECT (GPU)
# =======================================================================

def collect():
    _set_env()
    import numpy as np
    from tinygrad import Tensor, dtypes
    from tinygrad.nn.state import safe_load
    import phase1_algebra_head as H
    from phase1_algebra_head import (build_params, forward, load_alg,
                                     build_slot_masks, alt2_fact_buf,
                                     K_VARS, L_FAC, T_ALG)

    vs, vst, vtk, vg, vse = load_alg("test")
    n = len(vs)
    K_B = int(os.environ.get("ALG_BREATH", "1"))
    print(f"[membrane-scale] fixture=wild n={n} K_B={K_B} ckpt={CKPT}",
          flush=True)

    p = build_params(0)
    sd = safe_load(CKPT)
    assert set(sd.keys()) == set(p.keys()), \
        (sorted(set(sd) - set(p))[:4], sorted(set(p) - set(sd))[:4])
    for k in p:
        p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()

    argmax_breath = np.full((n, K_B, L_FAC), -1, np.int32)
    fat_mass = np.zeros((n, K_B, L_FAC, T_ALG), np.float16)  # NEW: full mass, not just argmax
    ids_all = np.zeros((n, T_ALG), np.int32)

    for s0 in range(0, n, 8):
        sl = np.arange(s0, min(s0 + 8, n))
        pad = 8 - len(sl)
        sl_p = np.concatenate([sl, sl[:1].repeat(pad)]) if pad else sl
        _st_np = np.ascontiguousarray(vst[sl_p])
        _tk_np = vtk[sl_p].astype(np.float32)
        ts = Tensor(_st_np, dtype=dtypes.half)
        tk = Tensor(_tk_np, dtype=dtypes.float)
        se = Tensor(vse[sl_p].astype(np.int32), dtype=dtypes.int)
        # pass 1: unmasked parse -> masks + live facts (loop_val's cycle, verbatim)
        o0 = forward(p, ts, tk, se)
        onp0 = {k: o0[k].realize().numpy() for k in ("fat", "args", "res")}
        mk = build_slot_masks(onp0, vse[sl_p].astype(np.int32))
        _ka = ("pres", "ftype", "op", "dig") + (("dup",) if "dup" in o0 else ())
        _oa = {**onp0, **{k: o0[k].realize().numpy() for k in _ka}}
        _nv = np.array([vs[int(i)].get("n_vars", K_VARS) for i in sl_p])
        _ma = np.array([vs[int(i)].get("m", 0) for i in sl_p])
        fb = alt2_fact_buf(_oa, vse[sl_p].astype(np.int32), _nv, _ma)
        fact_t = Tensor(fb, dtype=dtypes.float)
        # pass 2: masked walk; the tap hands back the per-breath attention
        o = forward(p, ts, tk, se, slot_mask=Tensor(mk, dtype=dtypes.float),
                    fact_buf=fact_t)
        fat_all = [t.realize().numpy() for t in o["fat_all"]]  # K_B x (8, L_FAC, T)
        assert len(fat_all) == K_B, (len(fat_all), K_B)
        tkm = _tk_np > 0.5  # (8, T) real-token mask (shared across breaths, batch, slots)
        neg = np.where(tkm, 0.0, -1e9).astype(np.float32)  # additive mask so argmax never lands on padding
        for kb in range(K_B):
            fa = fat_all[kb] + neg[:, None, :]
            argmax_breath[sl, kb, :] = fa[:len(sl)].argmax(-1)
            fat_mass[sl, kb, :, :] = fat_all[kb][:len(sl)].astype(np.float16)
        if (s0 // 8) % 20 == 0:
            print(f"[membrane-scale] wild {s0}/{n}", flush=True)

    # ids/tokmask/sent come straight off load_alg's own tokenized arrays
    # (vst is the state tensor, not token ids — recover ids via the
    # tokenizer exactly as membrane_census.py does, so decode() calls at
    # report time see the SAME ids this collect ran the model on).
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(H.TOKENIZER_JSON)
    for i in range(n):
        enc = tok.encode(vs[int(i)]["text"])
        L = min(len(enc.ids), T_ALG)
        ids_all[int(i), :L] = enc.ids[:L]

    np.savez(RAW_PATH, argmax_breath=argmax_breath, fat_mass=fat_mass,
             ids=ids_all, tokmask=vtk.astype(np.uint8), sent=vse.astype(np.int8),
             g_presence=vg["presence"], g_ftype=vg["ftype"],
             g_digits=vg["digits"])
    print(f"[membrane-scale] wrote {RAW_PATH}", flush=True)


# =======================================================================
# REPORT (pure numpy, no GPU)
# =======================================================================

PUNCT_CLAUSE = {",", ";"}
CONJ_CLAUSE = {"and", "but"}
PUNCT_MENTION_EXTRA = {"."}
CONJ_MENTION_EXTRA = {"or"}
VERBISH = {
    "is", "are", "was", "were", "be", "been", "being", "has", "have", "had",
    "will", "would", "can", "could", "costs", "cost", "spent", "spend",
    "spends", "gives", "gave", "give", "given", "bought", "buy", "buys",
    "sold", "sell", "sells", "needs", "need", "wants", "want", "makes",
    "made", "make", "paid", "pay", "pays", "receives", "receive",
    "received", "left", "leaves", "leave", "remains", "remain", "earns",
    "earn", "earned", "uses", "use", "used", "takes", "take", "took",
    "adds", "add", "added", "gets", "get", "got", "found", "finds", "find",
    "totals", "total", "equals", "equal", "contains", "contain", "starts",
    "start", "started", "ends", "end", "ended", "if", "then", "so",
    "because", "after", "before", "each", "per",
}


def _digit_runs(tok, ids, T):
    """Maximal runs of digit-only-decode tokens -> list of (start, end_excl, value).
    Verbatim from membrane_census.py."""
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


def _decode_tokens(tok, ids, T):
    """Per-token decoded, stripped, lowercased text (empty for out-of-range)."""
    out = []
    for t in range(T):
        out.append(tok.decode([int(ids[t])]).strip().lower())
    return out


def _segment_ids(dec, tokmask, sent, extra_boundary_words):
    """Cut sentence into segments at commas/semicolons + extra_boundary_words
    (also at every sentence-index change). Boundary token itself opens the
    NEW segment. Returns int32 array, -1 at padding."""
    T = len(dec)
    seg = -np.ones(T, dtype=np.int32)
    cur = 0
    prev_sent = None
    for t in range(T):
        if not tokmask[t]:
            continue
        s = int(sent[t])
        if prev_sent is not None and s != prev_sent:
            cur += 1
        if dec[t] in PUNCT_CLAUSE or dec[t] in extra_boundary_words:
            cur += 1
        seg[t] = cur
        prev_sent = s
    return seg


BANDS = ["token", "mention", "clause", "sentence", "other_sent"]


def _band_arrays(argmax_pos, occ_tok, mention_id, occ_mention, clause_id,
                  occ_clause, sent, occ_sent, tokmask):
    """Per-real-token band id (0..4) for a whole row, given one slot's gold
    occurrence sets. Vectorized numpy."""
    T = len(sent)
    band = np.full(T, 4, dtype=np.int8)  # default: other-sentence
    real = tokmask > 0
    if occ_sent:
        band[np.isin(sent, list(occ_sent)) & real] = 3
    if occ_clause:
        band[np.isin(clause_id, list(occ_clause)) & real] = 2
    if occ_mention:
        band[np.isin(mention_id, list(occ_mention)) & real] = 1
    band[np.array(sorted(occ_tok), dtype=np.int64)] = 0
    band[~real] = -1
    return band


def report():
    import json
    import numpy as np
    from tokenizers import Tokenizer
    import phase1_algebra_head as H
    tok = Tokenizer.from_file(H.TOKENIZER_JSON)
    T_ALG = H.T_ALG

    lines = []

    def P(s=""):
        print(s)
        lines.append(s)

    P("=" * 78)
    P("THE MEMBRANE CENSUS BY SCALE — PMS8_241, wild only (2026-09-24)")
    P("=" * 78)

    raw = np.load(RAW_PATH)
    ids = raw["ids"]; tokmask = raw["tokmask"]; sent = raw["sent"]
    pres = raw["g_presence"]; ftype = raw["g_ftype"]; gdig = raw["g_digits"]
    argmax_breath = raw["argmax_breath"]; fat_mass = raw["fat_mass"].astype(np.float32)
    n, K_B, L_FAC = argmax_breath.shape

    ps = np.load(PS_LEGAL)
    ok = {(int(r), int(c)): bool(o) for r, c, o in zip(ps["rows"], ps["slots"], ps["ok"])}

    P("")
    P("DEFINITIONS (see this file's module docstring for the full reasoning):")
    P("  gold location = digit-token runs decoding to the slot's g_digits value,")
    P("  occurrences unioned (no hand mention-spans exist on wild, same as the")
    P("  2026-09-19 census). CLAUSE = sentence cut on ,/;/and/but (the prompt's")
    P("  own recipe). MENTION = the same cut widened with ./or/a verb-ish")
    P("  stoplist (see VERBISH) -- a heuristic lexicon, offered as a LOWER BOUND")
    P("  on a real mention-span annotation, not ground truth; every mention")
    P("  segment nests strictly inside its clause segment by construction.")
    P("  Correctness = the banked MASKED wild read only (ps_legal_wild_PMS8_241,")
    P("  LV_LEGAL=num). 'final breath' = fat_all[-1] (fat_all[0] == out['fat']).")

    n_wrong = n_right = n_nomatch = n_notjoined = 0
    # argmax_band[cell]['wrong'|'right'][breath] -> list of band ints
    argmax_band = {kb: {"wrong": [], "right": []} for kb in range(K_B)}
    # mass_band[breath]['wrong'|'right'] -> list of 5-vectors (mass fraction per band)
    mass_band = {kb: {"wrong": [], "right": []} for kb in range(K_B)}
    other_sent_dist = {kb: {"wrong": [], "right": []} for kb in range(K_B)}

    for i in range(n):
        dec = _decode_tokens(tok, ids[i], T_ALG)
        clause_id = _segment_ids(dec, tokmask[i], sent[i], CONJ_CLAUSE)
        mention_id = _segment_ids(dec, tokmask[i], sent[i],
                                   CONJ_MENTION_EXTRA | PUNCT_MENTION_EXTRA | VERBISH)
        runs = _digit_runs(tok, ids[i], T_ALG)
        for j in range(L_FAC):
            if pres[i, j] < 0.5 or int(ftype[i, j]) != 1:
                continue  # GIVEN slots only
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
            occ_clause = {int(clause_id[t]) for t in occ_tok}
            occ_mention = {int(mention_id[t]) for t in occ_tok}
            key = "right" if cell else "wrong"
            if cell:
                n_right += 1
            else:
                n_wrong += 1
            band = _band_arrays(None, occ_tok, mention_id, occ_mention,
                                 clause_id, occ_clause, sent[i], occ_sent,
                                 tokmask[i])
            real = tokmask[i] > 0
            for kb in range(K_B):
                am = int(argmax_breath[i, kb, j])
                b_am = int(band[am])
                argmax_band[kb][key].append(b_am)
                if b_am == 4:
                    d = min(abs(int(sent[i, am]) - s_) for s_ in occ_sent)
                    other_sent_dist[kb][key].append(d)
                m = fat_mass[i, kb, j]
                mreal = m[real]
                tot = float(mreal.sum())
                if tot <= 0:
                    continue
                mvec = np.zeros(5, dtype=np.float64)
                br = band[real]
                for bcode in range(5):
                    sel = br == bcode
                    if sel.any():
                        mvec[bcode] = float(m[real][sel].sum()) / tot
                mass_band[kb][key].append(mvec)

    P("")
    P(f"n GIVEN slots: right={n_right}  wrong={n_wrong}  "
      f"no-textual-match={n_nomatch}  not-in-correctness-bank={n_notjoined}")

    # -------------------------------------------------------------
    # headline table: final breath, argmax band + mass-weighted band
    # -------------------------------------------------------------
    KF = K_B - 1
    P("")
    P("-" * 78)
    P(f"HEADLINE (final breath = breath {KF}): share of slots by pyramid scale")
    P("-" * 78)
    P("  ARGMAX (finest band containing the attended token):")
    header = "    " + "wrong".rjust(10) + "right".rjust(10)
    P("    band        " + "wrong".rjust(9) + "right".rjust(9))
    for bi, bname in enumerate(BANDS):
        w = argmax_band[KF]["wrong"]; r = argmax_band[KF]["right"]
        fw = np.mean([x == bi for x in w]) if w else float("nan")
        fr = np.mean([x == bi for x in r]) if r else float("nan")
        P(f"    {bname:10s}  {fw:8.3f}  {fr:8.3f}")
    P(f"    (n wrong={len(argmax_band[KF]['wrong'])}, n right={len(argmax_band[KF]['right'])})")
    for key in ("wrong", "right"):
        ds = other_sent_dist[KF][key]
        if ds:
            P(f"    other-sentence mean distance ({key}): {np.mean(ds):.2f} (n={len(ds)})")

    P("")
    P("  MASS-WEIGHTED (share of attention mass, not just the argmax cell):")
    P("    band        " + "wrong".rjust(9) + "right".rjust(9))
    for bi, bname in enumerate(BANDS):
        w = mass_band[KF]["wrong"]; r = mass_band[KF]["right"]
        mw = np.mean([v[bi] for v in w]) if w else float("nan")
        mr = np.mean([v[bi] for v in r]) if r else float("nan")
        P(f"    {bname:10s}  {mw:8.3f}  {mr:8.3f}")
    P(f"    (n wrong={len(mass_band[KF]['wrong'])}, n right={len(mass_band[KF]['right'])})")

    # -------------------------------------------------------------
    # trajectory across breaths, one line per band
    # -------------------------------------------------------------
    P("")
    P("-" * 78)
    P("TRAJECTORY ACROSS BREATHS (argmax band share, WRONG slots)")
    P("-" * 78)
    P("  band        " + "".join(f"b{kb}".rjust(8) for kb in range(K_B)))
    for bi, bname in enumerate(BANDS):
        row = f"  {bname:10s}"
        for kb in range(K_B):
            w = argmax_band[kb]["wrong"]
            fw = np.mean([x == bi for x in w]) if w else float("nan")
            row += f"{fw:8.3f}"
        P(row)

    P("")
    P("-" * 78)
    P("TRAJECTORY ACROSS BREATHS (argmax band share, RIGHT slots)")
    P("-" * 78)
    P("  band        " + "".join(f"b{kb}".rjust(8) for kb in range(K_B)))
    for bi, bname in enumerate(BANDS):
        row = f"  {bname:10s}"
        for kb in range(K_B):
            r = argmax_band[kb]["right"]
            fr = np.mean([x == bi for x in r]) if r else float("nan")
            row += f"{fr:8.3f}"
        P(row)

    P("")
    P("-" * 78)
    P("TRAJECTORY ACROSS BREATHS (mass-weighted band share, WRONG slots)")
    P("-" * 78)
    P("  band        " + "".join(f"b{kb}".rjust(8) for kb in range(K_B)))
    for bi, bname in enumerate(BANDS):
        row = f"  {bname:10s}"
        for kb in range(K_B):
            w = mass_band[kb]["wrong"]
            mw = np.mean([v[bi] for v in w]) if w else float("nan")
            row += f"{mw:8.3f}"
        P(row)

    P("")
    P("-" * 78)
    P("TRAJECTORY ACROSS BREATHS (mass-weighted band share, RIGHT slots)")
    P("-" * 78)
    P("  band        " + "".join(f"b{kb}".rjust(8) for kb in range(K_B)))
    for bi, bname in enumerate(BANDS):
        row = f"  {bname:10s}"
        for kb in range(K_B):
            r = mass_band[kb]["right"]
            mr = np.mean([v[bi] for v in r]) if r else float("nan")
            row += f"{mr:8.3f}"
        P(row)

    with open(OUT_TXT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[membrane-scale] wrote {OUT_TXT}")


if __name__ == "__main__":
    if MODE == "collect":
        collect()
    elif MODE == "report":
        report()
    else:
        raise SystemExit(f"unknown MS_MODE={MODE!r}")
