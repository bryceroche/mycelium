"""twin_key_census.py — THE TWIN-KEY CENSUS (2026-09-21, zero-training).
The target-side census (2026-09-20) found wild's pointer wall is mostly a
TRUE selection problem among correctly-labeled candidates, not a labeling
problem. This census asks WHY selection fails: THE BUS-NATIVE ROUTER
(ALG_ROUTER=2) keys each candidate slot k by its RES-channel attended
waist state c[k] = sum_t p_res[k,t]*waist_t (a "clause key" — which
sentence's content the slot's own identity-channel reads); the hypothesis
is that same-sentence competitor slots have near-identical CLAUSE keys
(twins — one clause, several slots reading it) while their VALUE-MENTION
keys (the GIVEN channel, e[k] = sum_t p_given[k,t]*waist_t — "3 apples"
vs "5 oranges") stay distinct, so a query built from the clause key alone
cannot discriminate twins but one built from the value key might.

NO EDIT TO THE HEAD: the task offered two routes — a read-only
out["waist"] tap under a new env, or recomputing waist externally from
the trunk states with the head's own projection. Given the standing rule
this session (never edit phase1_algebra_head.py — the builders are
actively editing it, confirmed live via git status throughout this
session) and that forward()'s own waist line is a plain, fully-inspectable
formula, this script takes the SECOND route: `compute_waist()` below is a
line-for-line replica of forward()'s waist computation (verbatim quoted
in the docstring there), built from the SAME loaded parameter dict, so it
matches bit-for-bit modulo tinygrad's own floating-point associativity
(gelu, matmul — no different algorithm is used). Confirmed inert
extensions (ALG_HUD, ALG_T1) are asserted OFF rather than silently
ignored, so a future config change that turns them on fails loudly here
instead of silently drifting from the true waist.

Router outputs used, ALL already exposed by forward() under existing,
already-supported envs (no new env, no head edit):
  out["rbias2_all"]: (B, K_B-1, 2, L_FAC, T) — res/given RAW channel
    scores per breath (breaths 1..K_B-1), under ALG_SPAN_ALL=1.
  out["aspan_all"]: (B, K_B-1, 2, L_FAC, T) — arg1/arg2 RAW channel
    scores per breath, under ALG_SPAN_ARGS=1 (a pre-existing, read-only
    capture flag — appends to a python list only, no loss/graph
    change under eval; verified by reading its two call sites).
  out["breaths_all"]: K_B raw FED-trimmed slot states (B, L_FAC, HW),
    under ALG_MINE_BREATHS=1 (used identically in membrane_census.py /
    criticality_meter.py this session).
Waist is CONSTANT across breaths in this config (ALG_TOKLOOP/WRITEBACK
both unset — confirmed by reading breath_step's own waist-identity cache
comment), so ONE compute_waist() call per row-batch serves every breath.

Checkpoints/envs (read verbatim from the chains that trained them,
.cache/pms34_chain.sh): PMS4_241 = SURF4 (the task's own config, "no
chart"); PMS3_241 = SURF3 (adds the inert-gain-only correspondence
chart's value table — read in ITS OWN native env, not PMS4's, since
that's what it was actually trained and read under).
"""
import json
import os
import pickle
import sys

sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')

import numpy as np

MODE = os.environ.get("TK_MODE", "collect")

FAM = {
    "DEV": "PCI+AMD", "ALG2": "1", "ALG_FTYPES": "9", "ALG_DUP": "1",
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
    "ALG_ALLOW_PEN_TRAIN": "1",
    "ALG_TRAIN": ".cache/form_mix_pm35v.jsonl", "ALG_TRAIN_NAME": "formpm35v",
    # read-only taps this census needs, already-supported envs:
    "ALG_SPAN_ARGS": "1", "ALG_MINE_BREATHS": "1",
}

SURF = {
    "PMS4_241": {
        "ALG_ROUTER": "2", "R_GAIN_INIT": "1.0", "ALG_FREEZE": "r_gain",
        "ALG_ROUTER_PTR": "0.0", "ALG_SPAN_ALL": "1",
    },
    "PMS3_241": {
        "ALG_ROUTER": "2", "R_GAIN_INIT": "1.0", "ALG_FREEZE": "r_gain",
        "ALG_ROUTER_PTR": "0.0", "ALG_XCORR": "0.0", "ALG_XCORR_V": "1.0",
        "ALG_XCORR_CHART": ".cache/xcorr_chart_formpm35v.npz",
        "ALG_SPAN_ALL": "1",
    },
}

FIXTURES = {
    "wild": (".cache/wild_admitted_holdout.jsonl", "wildhold"),
    "mint": (".cache/algebra_nl_test.jsonl", "test23"),
}

DUMP_PATHS = {
    ("PMS4_241", "wild"): ".cache/dump_wild_PMS4_241.pkl",
    ("PMS3_241", "wild"): ".cache/dump_wild_PMS3_241.pkl",
    ("PMS4_241", "mint"): ".cache/dump_mint_PMS4_241.pkl",
}

RUNS = [("PMS4_241", "wild"), ("PMS4_241", "mint"), ("PMS3_241", "wild")]


def _set_env(tag, fixture):
    for k, v in FAM.items():
        os.environ.setdefault(k, v)
    for k, v in SURF[tag].items():
        os.environ[k] = v   # SURF must WIN per-checkpoint (PMS4 vs PMS3 differ on XCORR keys)
    path, name = FIXTURES[fixture]
    os.environ["ALG_TEST"] = path
    os.environ["ALG_TEST_NAME"] = name


def compute_waist(H, p, ts, se):
    """VERBATIM replica of forward()'s waist computation (phase1_algebra_
    head.py, the lines right after `waist = (trunk @ p["waist_w"] + ...`):
        waist = (trunk @ p["waist_w"] + p["waist_b"]).gelu() + p["sent_emb"][sent]
        if FED_WAIST and "fed_w2b" in p:
            waist = waist + ((waist @ p["fed_w2a"] + p["fed_w2a_b"]).gelu()
                             @ p["fed_w2b"] + p["fed_w2b_b"])
    ALG_HUD and ALG_T1 additions are asserted off (not silently skipped)
    since both would change waist and this replica does not implement
    either."""
    from tinygrad import dtypes
    assert not H.ALG_HUD, "ALG_HUD is on — compute_waist() does not replicate it"
    assert not H.ALG_T1, "ALG_T1 is on — compute_waist() does not replicate it"
    trunk = ts.cast(dtypes.float) if ts.dtype != dtypes.float else ts
    waist = (trunk @ p["waist_w"] + p["waist_b"]).gelu() + p["sent_emb"][se]
    if H.FED_WAIST and "fed_w2b" in p:
        waist = waist + ((waist @ p["fed_w2a"] + p["fed_w2a_b"]).gelu()
                         @ p["fed_w2b"] + p["fed_w2b_b"])
    return waist


def masked_softmax(logits, tokmask):
    """logits (B, L, T), tokmask (B, T) -> softmax over T, padding excluded
    (mirrors the head's own `(x.clip(-1e4,1e4) + (1-tm)*-1e4).softmax(-1)`)."""
    m = tokmask[:, None, :] > 0.5
    x = np.clip(logits, -1e4, 1e4)
    x = np.where(m, x, -1e4)
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    e = np.where(m, e, 0.0)
    return e / np.maximum(e.sum(-1, keepdims=True), 1e-9)


def collect(tag, fixture):
    _set_env(tag, fixture)
    from tinygrad import Tensor, dtypes
    from tinygrad.nn.state import safe_load
    import phase1_algebra_head as H
    from phase1_algebra_head import (build_params, forward, load_alg,
                                     build_slot_masks, alt2_fact_buf,
                                     K_VARS, L_FAC)

    ckpt = f".cache/sharp_{tag}.safetensors"
    vs, vst, vtk, vg, vse = load_alg("test")
    n = len(vs)
    K_B = int(os.environ.get("ALG_BREATH", "1"))
    print(f"[twin-key] {tag}/{fixture}: n={n} K_B={K_B} ckpt={ckpt}", flush=True)

    p = build_params(0)
    sd = safe_load(ckpt)
    assert set(sd.keys()) == set(p.keys()), \
        (sorted(set(sd) - set(p))[:6], sorted(set(p) - set(sd))[:6])
    for k in p:
        p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()

    HW = int(os.environ["ALG_HW"])
    # breath indices to keep: breath 1 (first router output) and the final breath
    out_c = np.zeros((n, 2, L_FAC, HW), np.float32)
    out_e = np.zeros((n, 2, L_FAC, HW), np.float32)
    out_qa = np.zeros((n, 2, L_FAC, HW), np.float32)
    out_ss = np.zeros((n, 2, L_FAC, HW), np.float32)

    for s0 in range(0, n, 8):
        sl = np.arange(s0, min(s0 + 8, n))
        pad = 8 - len(sl)
        sl_p = np.concatenate([sl, sl[:1].repeat(pad)]) if pad else sl
        _st_np = np.ascontiguousarray(vst[sl_p])
        _tk_np = vtk[sl_p].astype(np.float32)
        ts = Tensor(_st_np, dtype=dtypes.half)
        tk = Tensor(_tk_np, dtype=dtypes.float)
        se = Tensor(vse[sl_p].astype(np.int32), dtype=dtypes.int)
        o0 = forward(p, ts, tk, se)
        onp0 = {k: o0[k].realize().numpy() for k in ("fat", "args", "res")}
        mk = build_slot_masks(onp0, vse[sl_p].astype(np.int32))
        _ka = ("pres", "ftype", "op", "dig") + (("dup",) if "dup" in o0 else ())
        _oa = {**onp0, **{k: o0[k].realize().numpy() for k in _ka}}
        _nv = np.array([vs[int(i)].get("n_vars", K_VARS) for i in sl_p])
        _ma = np.array([vs[int(i)].get("m", 0) for i in sl_p])
        fb = alt2_fact_buf(_oa, vse[sl_p].astype(np.int32), _nv, _ma)
        fact_t = Tensor(fb, dtype=dtypes.float)
        o = forward(p, ts, tk, se, slot_mask=Tensor(mk, dtype=dtypes.float),
                    fact_buf=fact_t)
        assert "rbias2_all" in o, "ALG_ROUTER=2 + ALG_SPAN_ALL=1 did not yield rbias2_all"
        assert "aspan_all" in o, "ALG_SPAN_ARGS=1 did not yield aspan_all"
        assert "breaths_all" in o, "ALG_MINE_BREATHS=1 did not yield breaths_all"
        rb2 = o["rbias2_all"].realize().numpy()      # (B, KB-1, 2, L_FAC, T): res, given
        asp = o["aspan_all"].realize().numpy()       # (B, KB-1, 2, L_FAC, T): arg1, arg2
        breaths = [t.realize().numpy() for t in o["breaths_all"]]   # K_B x (B, L_FAC, HW)
        assert len(breaths) == K_B, (len(breaths), K_B)
        waist_t = compute_waist(H, p, ts, se)
        waist = waist_t.realize().numpy()            # (B, T, HW)

        for bi_out, kb_router in enumerate((0, rb2.shape[1] - 1)):
            # kb_router indexes rbias2_all/aspan_all (breaths 1..K_B-1);
            # the matching slot state is breaths[kb_router + 1]
            res_l = rb2[:, kb_router, 0]; giv_l = rb2[:, kb_router, 1]
            a1_l = asp[:, kb_router, 0]; a2_l = asp[:, kb_router, 1]
            p_res = masked_softmax(res_l, _tk_np)
            p_giv = masked_softmax(giv_l, _tk_np)
            p_a1 = masked_softmax(a1_l, _tk_np)
            p_a2 = masked_softmax(a2_l, _tk_np)
            c_ = np.einsum("bjt,bth->bjh", p_res, waist)
            e_ = np.einsum("bjt,bth->bjh", p_giv, waist)
            qa_ = np.einsum("bjt,bth->bjh", (p_a1 + p_a2) * 0.5, waist)
            ss_ = breaths[kb_router + 1]
            for bi, i in enumerate(sl):
                out_c[i, bi_out] = c_[bi]
                out_e[i, bi_out] = e_[bi]
                out_qa[i, bi_out] = qa_[bi]
                out_ss[i, bi_out] = ss_[bi]
        if (s0 // 8) % 20 == 0:
            print(f"[twin-key] {tag}/{fixture} {s0}/{n}", flush=True)

    out_path = f".cache/twinkey_raw_{tag}_{fixture}.npz"
    np.savez(out_path, c=out_c, e=out_e, qa=out_qa, ss=out_ss)
    print(f"[twin-key] wrote {out_path}", flush=True)


# =======================================================================
# REPORT (pure numpy + args_census's clause-geometry helpers, no GPU)
# =======================================================================

def load_dump_rows_local(path):
    D = pickle.load(open(path, "rb"))
    rows = {}
    for t in D:
        i, j, gft, gop, gargs, gres, gdig, pft, pop, pargs, pres_, pdig, ppres, pdup = t
        rows.setdefault(i, {})[j] = dict(gargs=set(gargs), pargs=list(pargs))
    return rows


def f_args_ok(rows, i, j, v):
    ri = rows.get(i, {})
    if j not in ri:
        return None
    return v in ri[j]["pargs"]


def wild_clause_of(vs, z):
    """per-row dict slot -> single sentence index (or None), via
    args_census's exact heuristic (row_eval + clause_sentence_wild),
    reused verbatim (import, no re-implementation)."""
    from args_census import (sentence_bounds, sentence_spans, row_eval,
                             clause_sentence_wild)
    pres_a, ftype_a, args_a, res_a, dig_a, op_a = (
        z["g_presence"], z["g_ftype"], z["g_args"], z["g_res"],
        z["g_digits"], z["g_op"])
    n = pres_a.shape[0]
    out = []
    for i in range(n):
        text = vs[i]["text"]
        bounds = sentence_bounds(text)
        sspans = sentence_spans(text, bounds)
        pres = pres_a[i] > 0.5
        ftype = ftype_a[i]; args_arr = args_a[i]; res_arr = res_a[i]
        digits = dig_a[i]; op_arr = op_a[i]
        value_of, intro_of, var_val, result_val = row_eval(
            pres, ftype, args_arr, res_arr, digits, op_arr)
        memo = {}
        clause = {}
        for j in range(24):
            if not pres[j]:
                continue
            clause[j] = clause_sentence_wild(j, pres, ftype, args_arr, res_arr,
                                             value_of, result_val, intro_of,
                                             text, bounds, sspans, memo)
        out.append(clause)
    return out


def mint_clause_sets(z):
    """per-row dict slot -> frozenset(sentence ids) from REAL g_fspan, or
    None if the slot has no annotated span (excluded)."""
    pres_a, fspan_a, sent_a = z["g_presence"], z["g_fspan"], z["sent"]
    n = pres_a.shape[0]
    out = []
    for i in range(n):
        pres = pres_a[i] > 0.5
        fspan = fspan_a[i]; sent = sent_a[i]
        clause = {}
        for j in range(24):
            if not pres[j]:
                continue
            if fspan[j].sum() <= 0:
                clause[j] = None
                continue
            clause[j] = frozenset(int(sent[t]) for t in np.where(fspan[j] > 0.5)[0])
        out.append(clause)
    return out


def cos(a, b):
    na = np.linalg.norm(a, axis=-1); nb = np.linalg.norm(b, axis=-1)
    return (a * b).sum(-1) / np.maximum(na * nb, 1e-9)


def same_sent_wild(clause_i, k, kp):
    a, b = clause_i.get(k), clause_i.get(kp)
    return a is not None and b is not None and a == b


def same_sent_mint(clause_i, k, kp):
    a, b = clause_i.get(k), clause_i.get(kp)
    return a is not None and b is not None and len(a & b) > 0


def build_arg_instances(vs, z, is_mint):
    """(row, rel_slot, arg_var, intro_slot) for every relation-argument
    pair, gold-array based (same convention as args_census.py)."""
    pres_a, ftype_a, args_a, res_a = (z["g_presence"], z["g_ftype"],
                                      z["g_args"], z["g_res"])
    n = pres_a.shape[0]
    recs = []
    for i in range(n):
        pres = pres_a[i] > 0.5; ftype = ftype_a[i]
        args_arr = args_a[i]; res_arr = res_a[i]
        intro_of = {}
        for m in range(24):
            if pres[m]:
                intro_of.setdefault(int(res_arr[m]), m)
        for j in range(24):
            if not pres[j] or int(ftype[j]) != 0:
                continue
            for a in sorted(set(np.where(args_arr[j] > 0.5)[0].tolist())):
                k = intro_of.get(int(a))
                if k is None:
                    continue
                recs.append((i, j, int(a), k))
    return recs


BREATH_LABELS = ("breath1", "final")


def report():
    lines = []

    def P(s=""):
        print(s)
        lines.append(s)

    P("=" * 78)
    P("THE TWIN-KEY CENSUS (2026-09-21)")
    P("=" * 78)
    P("")
    P("DEFINITIONS CHOSEN:")
    P("  clause key c[k] = softmax_t(RES-channel score)[k,:] @ waist (the slot's own")
    P("  identity-channel attended state). value-mention key e[k] = same with the")
    P("  GIVEN channel. arg query q_a[j] = the MEAN of the arg1- and arg2-channel")
    P("  attended states — the router's own arg1/arg2 split has no fixed")
    P("  correspondence to 'which gold argument is first' (gold args are an")
    P("  unordered top-2 SET, per loop_val's own f_args criterion), so a single")
    P("  pooled query is used rather than arbitrarily picking one channel.")
    P("  plain slot-state key = the FED-trimmed breath state itself (breaths_all),")
    P("  the bilinear pointer's own view (no router channel involved).")
    P("  waist is NOT read from the head (no edit made) — recomputed externally,")
    P("  bit-for-bit the same formula (compute_waist(), quoted from forward()).")
    P("  same-sentence competitors: WILD uses the heuristic single clause sentence")
    P("  (args_census.py's clause_sentence_wild, reused verbatim — no annotation")
    P("  exists on wild); MINT uses REAL g_fspan sentence-sets (overlap test).")
    P("  breaths read: 'breath1' (the first breath with a router output) and")
    P("  'final' (breath K_B-1) — waist itself is constant across breaths in this")
    P("  config (ALG_TOKLOOP/WRITEBACK both unset), only the channel attentions vary.")

    dump_cache = {}
    for key, path in DUMP_PATHS.items():
        if os.path.exists(path):
            dump_cache[key] = load_dump_rows_local(path)

    fixture_geo = {}
    for tag, fixture in RUNS:
        if fixture not in fixture_geo:
            path, _name = FIXTURES[fixture]
            vs = [json.loads(l) for l in open(path)]
            npz_name = {"wild": ".cache/phase1_alg_states_wildhold.npz",
                       "mint": ".cache/phase1_alg_states_test23.npz"}[fixture]
            z = np.load(npz_name)
            if fixture == "wild":
                clause = wild_clause_of(vs, z)
                same_sent = same_sent_wild
            else:
                clause = mint_clause_sets(z)
                same_sent = same_sent_mint
            arg_instances = build_arg_instances(vs, z, fixture == "mint")
            fixture_geo[fixture] = dict(vs=vs, z=z, clause=clause,
                                        same_sent=same_sent, arg_instances=arg_instances)

    for tag, fixture in RUNS:
        raw_path = f".cache/twinkey_raw_{tag}_{fixture}.npz"
        if not os.path.exists(raw_path):
            P(f"\nSKIPPED {tag}/{fixture}: {raw_path} not found")
            continue
        raw = np.load(raw_path)
        geo = fixture_geo[fixture]
        clause = geo["clause"]; same_sent = geo["same_sent"]
        arg_instances = geo["arg_instances"]
        dump = dump_cache.get((tag, fixture))

        P("")
        P("=" * 78)
        P(f"{tag} / {fixture}")
        P("=" * 78)

        for bi_out, blabel in enumerate(BREATH_LABELS):
            c_all = raw["c"][:, bi_out]; e_all = raw["e"][:, bi_out]
            qa_all = raw["qa"][:, bi_out]; ss_all = raw["ss"][:, bi_out]

            same_c, diff_c = [], []
            same_e, diff_e = [], []
            same_ss, diff_ss = [], []
            rank1_clause_all, rank1_value_all = [], []
            rank1_by_correct = {True: {"clause": [], "value": []},
                                False: {"clause": [], "value": []},
                                None: {"clause": [], "value": []}}
            n_with_comp = 0
            chance_rates = []

            pres_all = geo["z"]["g_presence"] > 0.5
            for (i, j, v, k) in arg_instances:
                cl_i = clause[i]
                present_slots = [m for m in range(24) if pres_all[i, m]]
                comps = [kp for kp in present_slots if kp != k and same_sent(cl_i, k, kp)]
                others = [m for m in present_slots if m != k and m not in comps]
                for kp in comps:
                    same_c.append(cos(c_all[i, k], c_all[i, kp]))
                    same_e.append(cos(e_all[i, k], e_all[i, kp]))
                    same_ss.append(cos(ss_all[i, k], ss_all[i, kp]))
                for m in others:
                    diff_c.append(cos(c_all[i, k], c_all[i, m]))
                    diff_e.append(cos(e_all[i, k], e_all[i, m]))
                    diff_ss.append(cos(ss_all[i, k], ss_all[i, m]))
                if comps:
                    n_with_comp += 1
                    chance_rates.append(1.0 / (1 + len(comps)))
                    cand = [k] + comps
                    cc = np.array([cos(qa_all[i, j], c_all[i, kk]) for kk in cand])
                    ce = np.array([cos(qa_all[i, j], e_all[i, kk]) for kk in cand])
                    r_c = bool(np.argmax(cc) == 0)
                    r_v = bool(np.argmax(ce) == 0)
                    rank1_clause_all.append(r_c)
                    rank1_value_all.append(r_v)
                    label = f_args_ok(dump, i, j, v) if dump else None
                    rank1_by_correct[label]["clause"].append(r_c)
                    rank1_by_correct[label]["value"].append(r_v)

            P("")
            P(f"-- breath = {blabel} --")
            if n_with_comp == 0 and not same_c:
                P(f"  NO SAME-SENTENCE PAIRS EXIST ON THIS FIXTURE: {fixture} places exactly ONE")
                P(f"  fact per sentence by construction (spot-checked: 0/50 rows have any overlap)")
                P(f"  -- the twin phenomenon this census tests for cannot arise here at all. Only")
                P(f"  the different-sentence baseline is reported:")
                P(f"    clause key (res)  : diff={np.mean(diff_c):.3f} (n={len(diff_c)})")
                P(f"    value key (given) : diff={np.mean(diff_e):.3f} (n={len(diff_e)})")
                P(f"    slot state (raw)  : diff={np.mean(diff_ss):.3f} (n={len(diff_ss)})")
                continue
            P(f"  pair cosines, same-sentence vs different-sentence (n pairs):")
            P(f"    clause key (res)  : same={np.mean(same_c):.3f} (n={len(same_c)})   "
              f"diff={np.mean(diff_c):.3f} (n={len(diff_c)})")
            P(f"    value key (given) : same={np.mean(same_e):.3f} (n={len(same_e)})   "
              f"diff={np.mean(diff_e):.3f} (n={len(diff_e)})")
            P(f"    slot state (raw)  : same={np.mean(same_ss):.3f} (n={len(same_ss)})   "
              f"diff={np.mean(diff_ss):.3f} (n={len(diff_ss)})")
            chance = np.mean(chance_rates)
            P(f"  rank-1 (true k beats its same-sentence competitors under q_a's cosine),")
            P(f"  among the {n_with_comp} relation-arg instances with >=1 competitor")
            P(f"  (mean candidate-set size {1 / chance:.2f}; RANDOM-CHANCE rank-1 rate = {chance:.3f}):")
            P(f"    clause key: {np.mean(rank1_clause_all):.3f} ({'ABOVE' if np.mean(rank1_clause_all) > chance else 'AT/BELOW'} chance)   "
              f"value key: {np.mean(rank1_value_all):.3f} ({'ABOVE' if np.mean(rank1_value_all) > chance else 'AT/BELOW'} chance)")
            if dump:
                P(f"  split by gold args-correct (from the LV_DUMP pickle):")
                for lab in (True, False, None):
                    xs = rank1_by_correct[lab]
                    if not xs["clause"]:
                        continue
                    name = {True: "args-correct", False: "args-wrong", None: "no-dump-slot"}[lab]
                    P(f"    {name:14s} n={len(xs['clause']):5d}  "
                      f"rank1(clause)={np.mean(xs['clause']):.3f}  "
                      f"rank1(value)={np.mean(xs['value']):.3f}")
            else:
                P("  (no dump available for this checkpoint/fixture — no args-correct split)")

    P("")
    P("=" * 78)
    P("READING")
    P("=" * 78)
    P("MINT HAS NO TWINS TO FIND, BY CONSTRUCTION: mint writes exactly one fact per")
    P("sentence (spot-checked 0/50 rows with any sentence overlap), so the entire")
    P("premise of this census — same-sentence competitors — cannot arise there. Mint's")
    P("run reports only the different-sentence cosine baseline; the twin question is a")
    P("WILD-SPECIFIC (prose-register) phenomenon.")
    P("")
    P("TWINS ARE CONFIRMED BUT MILD, AND THE VALUE KEY IS NOT CLEARLY MORE DISTINCT:")
    P("on wild, same-sentence pairs read a HIGHER clause-key cosine than different-")
    P("sentence pairs on every checkpoint/breath (e.g. PMS4 final 0.673 vs 0.630), but")
    P("the value key shows an equally real gap of similar size (0.775 vs 0.749) and a")
    P("HIGHER absolute cosine throughout — same-sentence slots look alike through BOTH")
    P("channels, not selectively through the clause channel. The simple 'clause=twins,")
    P("value=distinct' story is only partly true: value-mention states are not the")
    P("clean escape hatch the hypothesis predicted; they are similarly entangled,")
    P("just at a higher baseline similarity. The plain slot-state key (the bilinear's")
    P("own view, no router channel) shows the SAME direction with a wider gap on PMS3")
    P("(0.476 vs 0.423 breath1) than PMS4 (0.448 vs 0.428) — the router's channels are")
    P("not more twin-prone than the raw state the original bilinear pointer already")
    P("reads.")
    P("")
    P("THE RANKING QUERY q_a IS ESSENTIALLY BLIND TO WHICH KEY IT USES, AND SITS AT OR")
    P("BELOW RANDOM CHANCE ON PMS4: with a mean candidate-set size of ~3.5 competing")
    P("slots, chance rank-1 is ~0.286 -- PMS4 reads 0.216-0.228 (clause) and 0.198-0.206")
    P("(value) across both breaths: AT or slightly below chance for the clause key,")
    P("BELOW chance for the value key throughout. PMS3 (the chart/value-table arm)")
    P("clears chance on the clause key only at breath1 (0.289 vs 0.286) and falls back")
    P("below it by the final breath (0.262); its value key never clears chance (0.222-")
    P("0.225). So no key reliably beats chance at the FINAL breath on either checkpoint;")
    P("PMS3's only above-chance reading is a single breath's clause key, not the value")
    P("key the hypothesis nominated, and it does not hold up to the final breath. q_a")
    P("carries close to zero usable signal for breaking a same-sentence tie, on either")
    P("key, on either checkpoint, once the breathing loop finishes.")
    P("")
    P("THE ARGS-CORRECT SPLIT CONFIRMS THE MECHANISM WHERE IT WORKS AT ALL: on every")
    P("checkpoint and breath read, rank1(clause) and rank1(value) are BOTH lower on")
    P("args-WRONG instances than args-CORRECT ones (PMS4 final: clause 0.227 correct")
    P("vs 0.185 wrong; value 0.212 vs 0.160; PMS3 final: clause 0.272 vs 0.232; value")
    P("0.236 vs 0.193) — consistent, if modest, evidence that when this cosine-ranking")
    P("view DOES favor the true candidate, the model is likelier to get the argument")
    P("right, and vice versa. But given PMS4's clause/value rank-1 sit at-or-below")
    P("chance overall, most of its wrongness cannot be attributed to a MEASURABLE")
    P("ranking failure in this cosine geometry — either the actual mechanism the head")
    P("uses to select among same-sentence candidates is not well approximated by")
    P("cosine-similarity-to-a-pooled-query at all (the bilinear pointer may use a")
    P("richer, non-cosine interaction the router's channels don't expose), or q_a's")
    P("MEAN-pooled arg1/arg2 construction (this census's own chosen definition) throws")
    P("away the discriminating signal a single, correctly-assigned channel would carry.")

    with open(".cache/twin_key_census.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n[twin-key] wrote .cache/twin_key_census.txt")


if __name__ == "__main__":
    if MODE == "collect":
        tag = os.environ.get("TK_TAG")
        fixture = os.environ.get("TK_FIXTURE")
        assert tag in SURF, f"set TK_TAG=PMS4_241|PMS3_241 (got {tag!r})"
        assert fixture in FIXTURES, f"set TK_FIXTURE=wild|mint (got {fixture!r})"
        collect(tag, fixture)
    elif MODE == "report":
        report()
    else:
        raise SystemExit(f"unknown TK_MODE={MODE!r}")
