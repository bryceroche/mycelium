"""membrane_rack.py — THE MEMBRANE CENSUS ON THE RACK (2026-10-06, CPU, delegated task Part A).

scripts/membrane_scale.py's collect() runs only the ALT2 two-pass cycle (build_slot_masks +
alt2_fact_buf). It does NOT thread THE THREE CONSULTS (ALG_ALT3) nor THE RACK's ports (rack3/rack5):
on RK_241 (THE RACK, form 2: claim + leaf freeze on the numeral certificate — ledger 2026-10-05
16:13/17:24/18:02) that script would read the body with the rack never firing. This script runs the
SAME full read cycle scripts/loop_val.py runs under ALG_ALT3=1 ALG_CERT=<g> ALG_RACK=1 (grep
`_consult3`, `rack3`, `cert3`, `fact_buf` there) — copied here as a standalone, non-JIT cycle (loop_val's
`_consult3` is a closure local to its `quick_val` function body, not importable without side effects;
this is the "copy the minimal cycle" fallback the task anticipated) — capturing the per-breath head-mean
slots<-tokens attention (out["fat_all"], ALG_MINE_BREATHS=1, exactly membrane_scale's collect() tap) AND
the rack sidecar's per-breath dry/claimed arrays (rack3 after consult 1, rack5 after consult 2, the
union), then computes membrane_scale's band tables on top — importing its band-assignment helpers
(_digit_runs, _decode_tokens, _segment_ids, _band_arrays, BANDS, the clause/mention boundary sets) by
import, never copied, since those are pure module-level functions/constants with no import-time side
effects.

On a body with no ALG_ALT3 (HS_241: ALG_RACK unset too) this reduces to membrane_scale.collect()'s own
ALT2 cycle exactly (same forward() calls, same arguments) — the PARITY GATE this script must pass
before its RK_241 numbers are trusted: run `--tag HS_241 --rows 64` and compare the resulting band
tables against the same tables computed from the ALREADY-BANKED .cache/membrane_raw_wild_HS_241.npz
(sliced to the same 64 rows) with `--parity-check`.

Usage:
  scripts/membrane_rack.py <ckpt> --tag <TAG> [--rows N]
      runs the full cycle (env-gated: ALG_ALT3/ALG_CERT/ALG_RACK/ALG_RACK_TESTS/ALG_RACK_FREEZE/
      ALG_HIER_* etc, all read from the environment the caller sets — the chain-script convention,
      not hardcoded here), writes .cache/membrane_raw_wild_<TAG>.npz (membrane_scale's own schema +
      the rack sidecar arrays) and .cache/membrane_rack_<TAG>.txt (the standard band tables, in
      membrane_scale_HS_241.txt's layout, PLUS the two rack-specific tables when ALG_RACK fired).

  scripts/membrane_rack.py <ckpt> --tag <TAG> --rows 64 --parity-check .cache/membrane_raw_wild_HS_241.npz
      after collecting, also loads the given banked raw npz, slices it to the SAME `--rows`, runs the
      identical band-table computation, and prints a side-by-side diff to 3 decimals (the HS_241 gate).

DEV=CPU is asserted (never takes .cache/gpu.lock; this script never acquires it).
"""
import os
import sys
import argparse
import time
import collections

sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np

# ---------------------------------------------------------------------------
# THE FAMILY ENV (defaults only — os.environ.setdefault; an inherited/caller
# env wins, exactly membrane_scale.py's _set_env() convention). This is
# .cache/rack_chain_RK.sh's $FAM, DEV forced to CPU per this task's brief
# (never PCI+AMD — no GPU, never .cache/gpu.lock). $SURF8 and the arm's own
# X (ALG_HIER_*, ALG_ALT3, ALG_CERT, ALG_RACK*) are NOT defaulted here: the
# caller (the systemd unit's `env ...` invocation) sets them, the same way
# rack_chain_RK.sh sets them for membrane_scale.py's own collect call.
# ---------------------------------------------------------------------------
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
    "ALG_MINE_BREATHS": "1",   # THE TAP: forward() hands back out["fat_all"]
}


def _set_env():
    for k, v in _FAM.items():
        os.environ.setdefault(k, v)
    assert os.environ.get("DEV") == "CPU", f"DEV={os.environ.get('DEV')!r}: this script is CPU-only by brief"


# ===========================================================================
# COLLECT: the full read cycle (ALT2, or ALT3+CERT+RACK when ALG_ALT3=1)
# ===========================================================================

def collect(ckpt, tag, n_rows=None):
    _set_env()
    from tinygrad import Tensor, dtypes
    from tinygrad.nn.state import safe_load
    import phase1_algebra_head as H
    from phase1_algebra_head import (
        build_params, forward, load_alg, build_slot_masks, alt2_fact_buf,
        certifier_bias, rack_pack, K_VARS, L_FAC, L_TOT, T_ALG,
        ALG_CERT, ALG_CERT_IMPLIED, ALG_RACK, ALG_RACK_TESTS)

    ALT3 = int(os.environ.get("ALG_ALT3", "0")) != 0
    print(f"[membrane-rack] collect tag={tag} ALT3={ALT3} CERT={ALG_CERT}/{ALG_CERT_IMPLIED} "
          f"RACK={ALG_RACK} RACK_TESTS={ALG_RACK_TESTS} ckpt={ckpt}", flush=True)
    if ALG_RACK:
        assert ALT3, "ALG_RACK needs ALG_ALT3 (asserted by phase1_algebra_head itself too)"

    vs, vst, vtk, vg, vse = load_alg("test")
    n = len(vs) if not n_rows else min(len(vs), n_rows)
    K_B = int(os.environ.get("ALG_BREATH", "1"))
    print(f"[membrane-rack] fixture=wild n={n}/{len(vs)} K_B={K_B}", flush=True)

    p = build_params(0)
    sd = safe_load(ckpt)
    assert set(sd.keys()) == set(p.keys()), (sorted(set(sd) - set(p))[:4], sorted(set(p) - set(sd))[:4])
    for k in p:
        p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()

    argmax_breath = np.full((n, K_B, L_FAC), -1, np.int32)
    fat_mass = np.zeros((n, K_B, L_FAC, T_ALG), np.float16)
    rack_dry3 = np.zeros((n, L_FAC), np.float32); rack_claim3 = np.zeros((n, T_ALG), np.float32)
    rack_dry5 = np.zeros((n, L_FAC), np.float32); rack_claim5 = np.zeros((n, T_ALG), np.float32)

    t_start = time.time()
    for s0 in range(0, n, 8):
        sl = np.arange(s0, min(s0 + 8, n))
        pad = 8 - len(sl)
        sl_p = np.concatenate([sl, sl[:1].repeat(pad)]) if pad else sl
        ts = Tensor(np.ascontiguousarray(vst[sl_p]), dtype=dtypes.half)
        tk = Tensor(vtk[sl_p].astype(np.float32), dtype=dtypes.float)
        se = Tensor(vse[sl_p].astype(np.int32), dtype=dtypes.int)

        # pass 1 (unmasked parse) -> the evidence-sharing slot mask, exactly membrane_scale/loop_val
        o0 = forward(p, ts, tk, se)
        onp0 = {k: o0[k].realize().numpy() for k in ("fat", "args", "res")}
        mk = build_slot_masks(onp0, vse[sl_p].astype(np.int32))
        mk_t = Tensor(mk, dtype=dtypes.float)
        _nv = np.array([vs[int(i)].get("n_vars", K_VARS) for i in sl_p])
        _ma = np.array([vs[int(i)].get("m", 0) for i in sl_p])

        f3_t = f5_t = c3_t = c5_t = r3_t = r5_t = r3_np = r5_np = None
        if ALT3:
            ck3 = ("pres", "ftype", "op", "dig", "args", "res") + (("dup",) if "dup" in o0 else ())

            def consult3(oo, rack_prev=None, rack_detail=None):
                onp3 = {k: oo[k].realize().numpy() for k in ck3}
                fb_ = alt2_fact_buf(onp3, vse[sl_p].astype(np.int32), _nv, _ma)
                cb_t = rk_t = rk = None
                if ALG_CERT or ALG_CERT_IMPLIED or ALG_RACK:
                    texts3 = [vs[int(i)]["text"] for i in sl_p]
                    rows3 = [{k: onp3[k][bi] for k in onp3} for bi in range(len(sl_p))]
                if ALG_CERT or ALG_CERT_IMPLIED:
                    cb_t = Tensor(certifier_bias(rows3, fb_, texts3, T_ALG, ALG_CERT, ALG_CERT_IMPLIED), dtype=dtypes.float)
                if ALG_RACK:
                    rk = rack_pack(rows3, fb_, texts3, T_ALG, ALG_RACK_TESTS, prev=rack_prev, detail=rack_detail)
                    rk_t = Tensor(rk, dtype=dtypes.float)
                return Tensor(fb_, dtype=dtypes.float), cb_t, rk_t, rk

            oa3 = forward(p, ts, tk, se, slot_mask=mk_t, stop_after=2)
            f3_t, c3_t, r3_t, r3_np = consult3(oa3)
            ob3 = forward(p, ts, tk, se, slot_mask=mk_t, stop_after=4, facts3=f3_t, cert3=c3_t, rack3=r3_t)
            f5_t, c5_t, r5_t, r5_np = consult3(ob3, rack_prev=r3_np)

            o = forward(p, ts, tk, se, slot_mask=mk_t,
                        facts3=f3_t, facts5=f5_t, cert3=c3_t, cert5=c5_t, rack3=r3_t, rack5=r5_t)
        else:
            _ka = ("pres", "ftype", "op", "dig") + (("dup",) if "dup" in o0 else ())
            _oa = {**onp0, **{k: o0[k].realize().numpy() for k in _ka}}
            fb = alt2_fact_buf(_oa, vse[sl_p].astype(np.int32), _nv, _ma)
            fact_t = Tensor(fb, dtype=dtypes.float)
            o = forward(p, ts, tk, se, slot_mask=mk_t, fact_buf=fact_t)

        fat_all = [t.realize().numpy() for t in o["fat_all"]]
        assert len(fat_all) == K_B, (len(fat_all), K_B)
        tkm = vtk[sl_p].astype(np.float32) > 0.5
        neg = np.where(tkm, 0.0, -1e9).astype(np.float32)
        for kb in range(K_B):
            fa = fat_all[kb] + neg[:, None, :]
            argmax_breath[sl, kb, :] = fa[:len(sl)].argmax(-1)
            fat_mass[sl, kb, :, :] = fat_all[kb][:len(sl)].astype(np.float16)

        if ALG_RACK:
            rack_dry3[sl] = r3_np[:len(sl), :L_FAC]; rack_claim3[sl] = r3_np[:len(sl), L_TOT:]
            rack_dry5[sl] = r5_np[:len(sl), :L_FAC]; rack_claim5[sl] = r5_np[:len(sl), L_TOT:]

        if (s0 // 8) % 5 == 0:
            el = time.time() - t_start
            print(f"[membrane-rack] wild {s0}/{n} ({el:.0f}s, {el / max(s0 + 8, 1):.2f}s/row)", flush=True)

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(H.TOKENIZER_JSON)
    ids_all = np.zeros((n, T_ALG), np.int32)
    for i in range(n):
        enc = tok.encode(vs[int(i)]["text"])
        L = min(len(enc.ids), T_ALG)
        ids_all[i, :L] = enc.ids[:L]

    raw_path = f".cache/membrane_raw_wild_{tag}.npz"
    np.savez(raw_path, argmax_breath=argmax_breath, fat_mass=fat_mass,
             ids=ids_all, tokmask=vtk[:n].astype(np.uint8), sent=vse[:n].astype(np.int8),
             g_presence=vg["presence"][:n], g_ftype=vg["ftype"][:n], g_digits=vg["digits"][:n],
             rack_dry3=rack_dry3, rack_claim3=rack_claim3, rack_dry5=rack_dry5, rack_claim5=rack_claim5,
             has_rack=np.array([bool(ALG_RACK)]))
    print(f"[membrane-rack] wrote {raw_path} ({time.time() - t_start:.0f}s total)", flush=True)
    return raw_path


# ===========================================================================
# BAND TABLES — reuses membrane_scale's band-assignment helpers by import
# ===========================================================================

def compute_bands(raw, ok, row_filter=None):
    """Mirrors membrane_scale.report()'s per-slot loop (GIVEN gold slots only,
    textually matched), returning the same aggregates it builds, optionally
    restricted to a subset of rows (row_filter: a set/array of row indices)."""
    import membrane_scale as MS
    from tokenizers import Tokenizer
    import phase1_algebra_head as H
    tok = Tokenizer.from_file(H.TOKENIZER_JSON)
    T_ALG = H.T_ALG

    ids = raw["ids"]; tokmask = raw["tokmask"]; sent = raw["sent"]
    pres = raw["g_presence"]; ftype = raw["g_ftype"]; gdig = raw["g_digits"]
    argmax_breath = raw["argmax_breath"]; fat_mass = raw["fat_mass"].astype(np.float32)
    n, K_B, L_FAC = argmax_breath.shape

    n_wrong = n_right = n_nomatch = n_notjoined = 0
    argmax_band = {kb: {"wrong": [], "right": []} for kb in range(K_B)}
    mass_band = {kb: {"wrong": [], "right": []} for kb in range(K_B)}
    other_sent_dist = {kb: {"wrong": [], "right": []} for kb in range(K_B)}
    cell_info = []   # (i, j, key) for every scored GIVEN slot — the rack tables key off this

    rows_iter = range(n) if row_filter is None else sorted(set(int(x) for x in row_filter) & set(range(n)))
    for i in rows_iter:
        dec = MS._decode_tokens(tok, ids[i], T_ALG)
        clause_id = MS._segment_ids(dec, tokmask[i], sent[i], MS.CONJ_CLAUSE)
        mention_id = MS._segment_ids(dec, tokmask[i], sent[i], MS.CONJ_MENTION_EXTRA | MS.PUNCT_MENTION_EXTRA | MS.VERBISH)
        runs = MS._digit_runs(tok, ids[i], T_ALG)
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
            band = MS._band_arrays(None, occ_tok, mention_id, occ_mention, clause_id, occ_clause, sent[i], occ_sent, tokmask[i])
            real = tokmask[i] > 0
            cell_info.append((i, j, key))
            for kb in range(K_B):
                am = int(argmax_breath[i, kb, j])
                b_am = int(band[am])
                argmax_band[kb][key].append(b_am)
                if b_am == 4:
                    d = min(abs(int(sent[i, am]) - s_) for s_ in occ_sent)
                    other_sent_dist[kb][key].append(d)
                m = fat_mass[i, kb, j]
                mreal = m[real]; tot = float(mreal.sum())
                if tot <= 0:
                    continue
                mvec = np.zeros(5, dtype=np.float64)
                br = band[real]
                for bcode in range(5):
                    sel = br == bcode
                    if sel.any():
                        mvec[bcode] = float(m[real][sel].sum()) / tot
                mass_band[kb][key].append(mvec)
    return dict(n_right=n_right, n_wrong=n_wrong, n_nomatch=n_nomatch, n_notjoined=n_notjoined,
                argmax_band=argmax_band, mass_band=mass_band, other_sent_dist=other_sent_dist,
                K_B=K_B, cell_info=cell_info)


def render_bands(agg, title, lines):
    K_B = agg["K_B"]; argmax_band = agg["argmax_band"]; mass_band = agg["mass_band"]; other_sent_dist = agg["other_sent_dist"]
    BANDS = __import__("membrane_scale").BANDS
    P = lines.append
    P(f"n GIVEN slots: right={agg['n_right']}  wrong={agg['n_wrong']}  no-textual-match={agg['n_nomatch']}  not-in-correctness-bank={agg['n_notjoined']}")
    KF = K_B - 1
    P(""); P("-" * 78); P(f"HEADLINE (final breath = breath {KF}): share of slots by pyramid scale"); P("-" * 78)
    P("  ARGMAX (finest band containing the attended token):")
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
    P(""); P("  MASS-WEIGHTED (share of attention mass, not just the argmax cell):")
    P("    band        " + "wrong".rjust(9) + "right".rjust(9))
    for bi, bname in enumerate(BANDS):
        w = mass_band[KF]["wrong"]; r = mass_band[KF]["right"]
        mw = np.mean([v[bi] for v in w]) if w else float("nan")
        mr = np.mean([v[bi] for v in r]) if r else float("nan")
        P(f"    {bname:10s}  {mw:8.3f}  {mr:8.3f}")
    P(f"    (n wrong={len(mass_band[KF]['wrong'])}, n right={len(mass_band[KF]['right'])})")
    for trj_name, src in (("argmax band share, WRONG slots", (argmax_band, "wrong")),
                          ("argmax band share, RIGHT slots", (argmax_band, "right")),
                          ("mass-weighted band share, WRONG slots", (mass_band, "wrong")),
                          ("mass-weighted band share, RIGHT slots", (mass_band, "right"))):
        store, key = src
        P(""); P("-" * 78); P(f"TRAJECTORY ACROSS BREATHS ({trj_name})"); P("-" * 78)
        P("  band        " + "".join(f"b{kb}".rjust(8) for kb in range(K_B)))
        for bi, bname in enumerate(BANDS):
            row = f"  {bname:10s}"
            for kb in range(K_B):
                vals = store[kb][key]
                if store is argmax_band:
                    fv = np.mean([x == bi for x in vals]) if vals else float("nan")
                else:
                    fv = np.mean([v[bi] for v in vals]) if vals else float("nan")
                row += f"{fv:8.3f}"
            P(row)
    return lines


# ===========================================================================
# THE TWO RACK-SPECIFIC TABLES
# ===========================================================================

def rack_tables(raw, ok, lines):
    import membrane_scale as MS
    from tokenizers import Tokenizer
    import phase1_algebra_head as H
    tok = Tokenizer.from_file(H.TOKENIZER_JSON)
    T_ALG = H.T_ALG
    ids = raw["ids"]; tokmask = raw["tokmask"]; sent = raw["sent"]
    pres = raw["g_presence"]; ftype = raw["g_ftype"]; gdig = raw["g_digits"]
    argmax_breath = raw["argmax_breath"]; fat_mass = raw["fat_mass"].astype(np.float32)
    rack_dry3 = raw["rack_dry3"]; rack_claim3 = raw["rack_claim3"]
    rack_dry5 = raw["rack_dry5"]; rack_claim5 = raw["rack_claim5"]
    n, K_B, L_FAC = argmax_breath.shape

    P = lines.append
    P(""); P("=" * 78); P("RACK TABLE 1 — DRY vs WET slots, breaths after the first consult (b3..b6)"); P("=" * 78)
    P("  population: GIVEN gold slots with a textual match (same census as the band tables above).")
    P("  'claimed mass' = the renormalized attention mass this slot places on tokens ANY dry slot has")
    P("  claimed (rack3's claim mask for b3/b4; rack5's union for b5/b6) — the claim should drive this")
    P("  near 0 for WET slots (their own numeral's tokens are not generally in the claim set) if the")
    P("  freeze/claim road is actually steering attention away from settled dishes.")
    P("  band        " + "  n".rjust(6) + "argmax-token".rjust(14) + "claimed-mass".rjust(14))
    for kb in (3, 4, 5, 6):
        if kb >= K_B:
            continue
        dry_arr, claim_arr = (rack_dry3, rack_claim3) if kb in (3, 4) else (rack_dry5, rack_claim5)
        for state in ("dry", "wet"):
            tok_hits = []; claimed_fracs = []
            for i in range(n):
                dec = MS._decode_tokens(tok, ids[i], T_ALG)
                runs = MS._digit_runs(tok, ids[i], T_ALG)
                for j in range(L_FAC):
                    if pres[i, j] < 0.5 or int(ftype[i, j]) != 1:
                        continue
                    is_dry = dry_arr[i, j] > 0.5
                    if (state == "dry") != bool(is_dry):
                        continue
                    v = int("".join(str(int(x)) for x in gdig[i, j]))
                    matches = [(a, b, val) for a, b, val in runs if val == v]
                    if not matches:
                        continue
                    occ_tok = {t for a, b, val in matches for t in range(a, b)}
                    am = int(argmax_breath[i, kb, j])
                    tok_hits.append(am in occ_tok)
                    real = tokmask[i] > 0
                    m = fat_mass[i, kb, j]; mreal = m[real]; tot = float(mreal.sum())
                    if tot <= 0:
                        continue
                    claim_real = claim_arr[i][real] > 0.5
                    claimed_fracs.append(float(mreal[claim_real].sum()) / tot if claim_real.any() else 0.0)
            th = np.mean(tok_hits) if tok_hits else float("nan")
            cf = np.mean(claimed_fracs) if claimed_fracs else float("nan")
            P(f"  b{kb} {state:3s}     {len(tok_hits):6d}  {th:12.3f}  {cf:12.3f}")

    P(""); P("=" * 78); P("RACK TABLE 2 — THE WALL (b6, wrong argmax other_sent) / THE HIT (b2..b6, right argmax token)"); P("=" * 78)
    P("                  restricted to rows with >= 1 dry slot (ever) vs rows with none")
    has_dry_row = (rack_dry5.max(axis=1) > 0.5) | (rack_dry3.max(axis=1) > 0.5)
    for label, mask in (("rows with >=1 dry slot", has_dry_row), ("rows with 0 dry slots", ~has_dry_row)):
        rows_sel = np.where(mask)[0]
        agg = compute_bands(raw, ok, row_filter=rows_sel)
        KF = agg["K_B"] - 1
        w = agg["argmax_band"][KF]["wrong"]
        wall = np.mean([x == 4 for x in w]) if w else float("nan")
        hits = []
        for kb in range(2, agg["K_B"]):
            r = agg["argmax_band"][kb]["right"]
            hits.append(np.mean([x == 0 for x in r]) if r else float("nan"))
        P(f"  {label:24s} n_rows={len(rows_sel):4d}  WALL(b6, wrong other_sent)={wall:.3f}  "
          f"HIT(b2..b6, right token)=" + " ".join(f"{h:.3f}" for h in hits))
    return lines


# ===========================================================================
# main
# ===========================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--rows", type=int, default=None)
    ap.add_argument("--parity-check", default=None, help="a banked raw npz (e.g. membrane_raw_wild_HS_241.npz) to diff against, sliced to the same --rows")
    ap.add_argument("--skip-collect", action="store_true", help="the raw npz for --tag already exists; report only")
    a = ap.parse_args()

    raw_path = f".cache/membrane_raw_wild_{a.tag}.npz"
    if not a.skip_collect:
        raw_path = collect(a.ckpt, a.tag, n_rows=a.rows)
    raw = np.load(raw_path)

    ps_path = f".cache/ps_legal_wild_{a.tag}.npz"
    ps = np.load(ps_path)
    ok = {(int(r), int(c)): bool(o) for r, c, o in zip(ps["rows"], ps["slots"], ps["ok"])}

    lines = []
    lines.append("=" * 78)
    lines.append(f"THE MEMBRANE CENSUS ON THE RACK — {a.tag}, wild only (2026-10-06)")
    lines.append("=" * 78)
    lines.append("")
    lines.append("Same definitions as membrane_scale.py (gold location = digit-token runs; CLAUSE/MENTION")
    lines.append("segmentation; correctness = the banked MASKED wild read, ps_legal_wild_<tag>.npz); the")
    lines.append("full ALT3+CERT+RACK read cycle runs here (membrane_scale.py's own collect() does not).")

    agg = compute_bands(raw, ok)
    render_bands(agg, a.tag, lines)

    has_rack = bool(raw["has_rack"][0]) if "has_rack" in raw.files else False
    if has_rack:
        rack_tables(raw, ok, lines)
    else:
        lines.append(""); lines.append("(ALG_RACK was not set for this collect — no rack-specific tables; this is the HS_241-parity / no-rack body)")

    out_path = f".cache/membrane_rack_{a.tag}.txt"
    txt = "\n".join(lines) + "\n"
    open(out_path, "w").write(txt)
    print(txt)
    print(f"[membrane-rack] -> {out_path}")

    if a.parity_check:
        bank = np.load(a.parity_check)
        n = raw["argmax_breath"].shape[0]
        sliced = {k: (bank[k][:n] if bank[k].ndim and bank[k].shape[0] == bank["argmax_breath"].shape[0] else bank[k]) for k in bank.files}
        agg_bank = compute_bands(sliced, ok)
        print("\n[parity] fresh-collect vs banked raw, both sliced to", n, "rows:")
        KF = agg["K_B"] - 1
        for key in ("wrong", "right"):
            for bi, bname in enumerate(__import__("membrane_scale").BANDS):
                a1 = np.mean([x == bi for x in agg["argmax_band"][KF][key]]) if agg["argmax_band"][KF][key] else float("nan")
                b1 = np.mean([x == bi for x in agg_bank["argmax_band"][KF][key]]) if agg_bank["argmax_band"][KF][key] else float("nan")
                flag = "OK" if (np.isnan(a1) and np.isnan(b1)) or abs(a1 - b1) < 5e-4 else "MISMATCH"
                print(f"  argmax {key:5s} {bname:10s} fresh={a1:.3f} banked={b1:.3f} {flag}")


if __name__ == "__main__":
    main()
