"""phase1_algebra_head.py — the ALGEBRA delta head: the §11 layout, built.

THE OBJECT: parse algebra-in-words into arith3 graphs — the first head whose output
feeds a domain where the symbolic jaw has GENUINE deciding to do (per-sample band
labels 0-3 decisions). Layout per the registered §11 frame:

  TWO SLOT BANKS over the shared waist (frozen Llama L0-L3 trunk, T=256):
    VARIABLE slots (24): identity = positional (slot i <-> letter i); text anchoring
      supervised by the generator's MENTION spans (name->slot binding AS STRUCTURE,
      the §6 law applied prospectively — the text-NACK lesson).
    FACTOR slots (24): span-supervised like the KenKen head.
  POINTERS are BILINEAR (factor-state x variable-state): args = 2-hot BCE over var
  slots; result = CE over var slots. Directly supervised from gold — the attention-
  bootstrap law's sanctioned escape, third deployment.
  RESULT = UNION TYPE: is-literal mode bit; rel factors -> result POINTER; given
  factors -> var pointer + value DIGITS (3 x 10-way, MSD-first, transplanted).
  QUERY POINTER: one global head over variable slots (solving is not answering).

EVAL IS PER-BAND from run one (registered): factor-exact / graph-solve / ANSWER
rate logged per decisions-band — the parse-vs-solve factorization question answers
itself. ANSWER rate = the honest end metric: decode -> problem_from_algebra ->
solve_symbolic -> solution[predicted query] == gold answer.

USAGE:
  Selftest:    .venv/bin/python3 scripts/phase1_algebra_head.py --selftest
  Precompute:  DEV=AMD .venv/bin/python3 scripts/phase1_algebra_head.py --precompute
  Train:       DEV=AMD STEPS=8000 .venv/bin/python3 scripts/phase1_algebra_head.py --train
  Eval:        DEV=AMD .venv/bin/python3 scripts/phase1_algebra_head.py --eval
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

import numpy as np

T_ALG = 256
ALG_REF = int(os.environ.get("ALG_REF", "0"))   # E-FLOOR referent supervision
ALG_DIAL = int(os.environ.get("ALG_DIAL", "0"))  # door #45: the dialect reader
ALG_VALATT = int(os.environ.get("ALG_VALATT", "0"))  # door #61: given-binding aid
ALG_SIXWAVE = int(os.environ.get("ALG_SIXWAVE", "0"))  # door #62: six-wave slot phasing
ALG_LSENT = int(os.environ.get("ALG_LSENT", "0"))    # V2: letter-keyed partition input
ALG_SYNC = int(os.environ.get("ALG_SYNC", "0"))      # sync-complete: one clock, both sides, ticking
ALG_CONSUME = int(os.environ.get("ALG_CONSUME", "0"))  # consume-once credit (any breath, once)
ALG_NOTEBOOK = int(os.environ.get("ALG_NOTEBOOK", "0"))  # the cathedral notebook
ALG_CIRCLE = int(os.environ.get("ALG_CIRCLE", "0"))      # the traffic circle
ALG_STELLAR = int(os.environ.get("ALG_STELLAR", "0"))    # cell-3b: helical handoff
NB_PERSLOT = int(os.environ.get("NB_PERSLOT", "0"))      # per-slot lanes: sharp ink
ALG_SEPHASE_Q = int(os.environ.get("ALG_SEPHASE_Q", "0"))   # identity channel
SEPHASE_Q_SCRAMBLE = int(os.environ.get("SEPHASE_Q_SCRAMBLE", "0"))
NB_FOCAL = float(os.environ.get("NB_FOCAL", "0"))        # magnifying glass: read sharpening
SEPHASE_B_BAND = int(os.environ.get("SEPHASE_B_BAND", "0"))  # banded-overlap stamps
ALG_SEPHASE_W = int(os.environ.get("ALG_SEPHASE_W", "0"))   # waist channel
ALG_SEPHASE_PAIR = int(os.environ.get("ALG_SEPHASE_PAIR", "0"))  # transceiver
ALG_SEPHASE_SETTLE = int(os.environ.get("ALG_SEPHASE_SETTLE", "0"))  # settle transceiver


def phase_alphabet(n, dims, scale, rng, noise=0.2):
    """THE SHARED SIX-PHASE ALPHABET (2026-08-19): one coordinate code for
    every seeded channel — row k rides phase (k mod 6)*pi/3 at frequency
    family k+1, native scale, brownian seasoning (the receipt's recipe)."""
    _k = np.arange(n)
    _d = np.arange(dims)
    _ph = (_k % 6) * (np.pi / 3.0)
    _pat = scale * np.sqrt(2.0) * np.cos(
        _ph[:, None] + _d[None, :] * (2 * np.pi / dims) * (_k[:, None] + 1))
    return (_pat + rng.randn(n, dims) * scale * noise).astype(np.float32)
NB_H = int(os.environ.get("NB_H", "4"))                  # calibrated horizon
H_TRUNK = 2048
H_W = int(os.environ.get("ALG_HW", "512"))  # capacity-probe dial (2026-07-11)
K_VARS = 24
L_FAC = 24
# ANSWER_SPACE_SPEC E1+E3 (2026-08-03): ALG_WIDE=1 widens the digit bank
# to 7 (values to 1e6, MSD-first as always) and adds the sign head.
ALG_WIDE = int(os.environ.get("ALG_WIDE", "0"))
N_DIG = 7 if ALG_WIDE else 3
N_HEADS = 8
SENT_MAX = 32


# ===========================================================================
# THE TERMINAL REGISTRY — the buffer-spec door, v1 (2026-08-04; earned by
# five dens in one day). ONE authority describing every trainable terminal:
# its params, its emission key, its gold keys, and the env gate that brings
# it into existence. v1 is the ASSERT arm of derive-or-assert (#128): the
# wiring below stays hand-written (the deployed gate's code is not
# refactored the week it promoted); every trainer ASSERTS its buffer set
# and every build can assert its emission set against THIS table. The
# DERIVE arm (generating forward/loss/buffers from the table) waits for a
# proper refactor window. A sixth terminal added without updating this
# table fails LOUDLY at build, not at the optimizer.
# ===========================================================================
def _ft(): return int(os.environ.get("ALG_FTYPES", "4"))
TERMINALS = {
    "pres":   {"params": ["h_pres", "h_pres_b"],   "emit": "pres",  "gold": ["presence"], "when": lambda: True},
    "ftype":  {"params": ["h_ftype", "h_ftype_b"], "emit": "ftype", "gold": ["ftype"],    "when": lambda: True},
    "op":     {"params": ["h_op", "h_op_b"],       "emit": "op",    "gold": ["op"],       "when": lambda: True},
    "islit":  {"params": ["h_islit", "h_islit_b"], "emit": "islit", "gold": ["is_lit_f"], "when": lambda: True},
    "dig":    {"params": ["h_dig", "h_dig_b"],     "emit": "dig",   "gold": ["digits"],   "when": lambda: True},
    "args":   {"params": ["W_args"],               "emit": "args",  "gold": ["args"],     "when": lambda: True},
    "dargs":  {"params": ["W_dargs"],              "emit": "dargs", "gold": ["args", "arg_dup"],
               "when": lambda: int(os.environ.get("ALG_DUPPTR", "0")) > 0},
    "res":    {"params": ["W_res"],                "emit": "res",   "gold": ["res"],      "when": lambda: True},
    "query":  {"params": ["W_query"],              "emit": "query", "gold": ["query"],    "when": lambda: True},
    "sel":    {"params": ["h_sel", "h_sel_b"],     "emit": "sel",   "gold": ["sel"],      "when": lambda: _ft() >= 5},
    "dup":    {"params": ["h_dup", "h_dup_b"],     "emit": "dup",   "gold": ["arg_dup"],  "when": lambda: int(os.environ.get("ALG_DUP", "0")) > 0},
    "dig2":   {"params": ["h_dig2", "h_dig2_b"],   "emit": "dig2",  "gold": ["digits2", "is_macro"], "when": lambda: int(os.environ.get("ALG2", "0")) and _ft() >= 7},
    "y":      {"params": ["W_y"],                  "emit": "y",     "gold": ["y"],        "when": lambda: int(os.environ.get("ALG2", "0")) and _ft() >= 7},
    "sgn":    {"params": ["h_sgn", "h_sgn_b"],     "emit": "sgn",   "gold": ["sign"],     "when": lambda: ALG_WIDE},
    "depth":  {"params": ["h_depth", "h_depth_b"], "emit": "depth", "gold": ["depth"],
               "when": lambda: int(os.environ.get("ALG_POSCH", "0")) > 0},
    "term":   {"params": ["h_term", "h_term_b"],   "emit": "term",  "gold": ["term"],
               "when": lambda: int(os.environ.get("ALG_POSCH", "0")) > 0},
    "opc":    {"params": ["W_opc1", "W_opc1_b", "W_opc2", "W_opc2_b"],
               "emit": "opc", "gold": ["opc"],
               "when": lambda: int(os.environ.get("ALG_OPCOUNT", "0")) > 0},
    "bindbus": {"params": (["W_bind1", "W_bind1_b"] + [f"W6_{r}" for r in ("a1", "a2", "rs", "op")]
                           if int(os.environ.get("ALG_BINDBUS", "0")) >= 6
                           else [f"W5_{r}_{s}" for r in ("a1", "a2", "rs", "op")
                                 for s in ("1", "b", "2")]
                           if int(os.environ.get("ALG_BINDBUS", "0")) >= 5
                           else ["W_bind1", "W_bind1_b", "W_bind2"]
                           if int(os.environ.get("ALG_BINDBUS", "0")) >= 2
                           else ["W_bind"]),
                "emit": "bind", "gold": (["bind_ids"] if int(os.environ.get("ALG_BINDBUS", "0")) >= 5
                                         else ["bindvec", "bind_ids"] if int(os.environ.get("ALG_BINDBUS", "0")) >= 3 else ["bindvec"]),
                "when": lambda: int(os.environ.get("ALG_BINDBUS", "0")) > 0},
    "cmt":    {"params": ["W_cmt", "W_cmt_b"],     "emit": "cmt",   "gold": ["ftype", "res"],
               "when": lambda: int(os.environ.get("ALG_RINGS", "0")) and int(os.environ.get("ALG_BREATH", "1")) > 1},
    "cmtreg": {"params": ["w_cmt_reg"],            "emit": "cmt",   "gold": ["ftype", "res"],
               "when": lambda: int(os.environ.get("ALG_RINGS", "0")) and int(os.environ.get("ALG_CMT_REG", "0"))
               and int(os.environ.get("ALG_BREATH", "1")) > 1},
}


def assert_terminals(p=None, emitted=None, gold_keys=None, site="build"):
    """THE DOOR'S ASSERT: every active terminal has (a) its params built,
    (b) its emission present (when `emitted` given), (c) its gold buffers
    present (when `gold_keys` given). Raises with the two-terminal law's
    own words; a missing entry fails at BUILD, never at the optimizer."""
    for name, t in TERMINALS.items():
        if not t["when"]():
            continue
        if p is not None:
            missing = [k for k in t["params"] if k not in p]
            assert not missing, (
                f"TERMINAL '{name}' params missing at {site}: {missing} "
                f"(the two-terminal law — the registry says this terminal "
                f"exists under the current env)")
        if emitted is not None:
            assert t["emit"] in emitted, (
                f"TERMINAL '{name}' NOT EMITTED at {site} — a forward "
                f"variant left it out (the fifth-den shape); grad will be "
                f"None at the optimizer unless every emission ships")
        if gold_keys is not None:
            missing = [g for g in t["gold"] if g not in gold_keys]
            assert not missing, (
                f"TERMINAL '{name}' gold buffers missing at {site}: "
                f"{missing} (the fourth-den shape — without these the "
                f"terminal leaves the graph: None grads)")


ALG_TRAIN = os.environ.get("ALG_TRAIN", ".cache/algebra_nl_train.jsonl")
ALG_TEST = os.environ.get("ALG_TEST", ".cache/algebra_nl_test.jsonl")
TEST_NAME = os.environ.get("ALG_TEST_NAME", "test")   # states-file key for the test slice
TRAIN_NAME = os.environ.get("ALG_TRAIN_NAME", "train")  # states-file key for the train slice
STATES_NPZ = ".cache/phase1_alg_states_{split}.npz"
STATES_NPY = ".cache/phase1_alg_states_{split}_states.npy"  # memmap sibling (gen-7+)
ALG_CKPT = os.environ.get("ALG_CKPT", ".cache/phase1_algebra_head.safetensors")
TOKENIZER_JSON = ".cache/llama-3.2-1b-weights/tokenizer.json"


# ===========================================================================
# GOLD (CPU): jsonl + offsets -> slot tensors per the §11 layout
# ===========================================================================

def _spans_to_tokmask(spans, offs, out):
    for (cs, ce) in spans:
        for ti, (ts, te) in enumerate(offs):
            if ti >= T_ALG:
                break
            if ts < ce and te > cs:
                out[ti] = 1.0


def _pos_meta(r):
    """per-factor (graph depth 0-5, terminality) — the position channel's
    gold (2026-08-24; labels derived from the factor list, key-lawful)."""
    facs = r["factors"]; q = r.get("query_var", r.get("query", 0))
    def var_of(f): return f.get("result", f.get("var", 0))
    def dep(fi, seen):
        f = facs[fi]
        if f["ftype"] == "given": return 0
        srcs = f.get("args", []) if "args" in f else [f.get("var", 0)]
        ds = []
        for v in srcs:
            for fj, gg in enumerate(facs):
                if fj != fi and var_of(gg) == v and fj not in seen:
                    ds.append(dep(fj, seen | {fj}))
        return 1 + max(ds, default=0)
    return [(min(dep(fi, {fi}), 5), 1.0 if var_of(f) == q else 0.0)
            for fi, f in enumerate(facs)]


OPC_CLASSES = ["add", "sub", "mul", "div", "sq", "opa", "fr",
               "given", "mod", "sel", "pct", "fdiv"]
OPC_CAP = 7


def _opc_meta(r):
    """per-row op-class counts (lever 3, 2026-08-25) — the SAME op-grain
    mapping as chain_decode's grader (rel->op / dup-mul->sq / macro->opa|fr
    / frac->fr / else ftype); gold at the ROW grain, where exactness lives."""
    from collections import Counter
    cnt = Counter()
    for f in r["factors"]:
        if f["ftype"] == "rel":
            if f.get("op") == "mul" and len(set(f.get("args", []))) == 1:
                cnt["sq"] += 1
            else:
                cnt[f.get("op", "add")] += 1
        elif f["ftype"] == "macro":
            cnt["opa" if f.get("name") == "OP_APPLY" else "fr"] += 1
        elif f["ftype"] == "frac":
            cnt["fr"] += 1
        else:
            cnt[f["ftype"]] += 1
    return [min(cnt.get(c, 0), OPC_CAP) for c in OPC_CLASSES]


def build_gold(samples, offsets):
    n = len(samples)
    g = {
        "presence": np.zeros((n, L_FAC), np.float32),
        "ftype": np.zeros((n, L_FAC), np.int32),         # 0=rel 1=given 2=mod 3=sel
        "op": np.zeros((n, L_FAC), np.int32),            # 0=add 1=mul
        "args": np.zeros((n, L_FAC, K_VARS), np.float32),
        "res": np.zeros((n, L_FAC), np.int32),
        "is_lit": np.zeros((n, L_FAC), np.float32),
        "digits": np.zeros((n, L_FAC, N_DIG), np.int32),
        "sign": np.zeros((n, L_FAC), np.float32),        # E1: 1.0 = negative literal
        "fspan": np.zeros((n, L_FAC, T_ALG), np.float32),
        "vspan": np.zeros((n, K_VARS, T_ALG), np.float32),
        **({"refvar": np.full((n, T_ALG), -1, np.int8)} if ALG_REF else {}),
        **({"is_ind": np.zeros((n, L_FAC), np.float32)} if ALG_DIAL else {}),
        "query": np.zeros((n,), np.int32),
        "band": np.zeros((n,), np.int32),
        # the tranche (2026-07-09): explicit per-kind masks (rel mask was
        # (1-is_lit) — wrong once mod/sel exist) + selector-type gold
        "sel": np.zeros((n, L_FAC), np.int32),           # SEL_TO_ID
        "is_rel": np.zeros((n, L_FAC), np.float32),
        "is_mod": np.zeros((n, L_FAC), np.float32),
        "is_sel": np.zeros((n, L_FAC), np.float32),
        # tranche 2 (2026-07-10): pct=4, fdiv=5 — the mod head-shape
        "is_pct": np.zeros((n, L_FAC), np.float32),
        "is_fdiv": np.zeros((n, L_FAC), np.float32),
        # gen-9 (2026-07-12): arg multiplicity — args=[a,a] was
        # UNREPRESENTABLE (multi-hot gold + top-2-distinct decode); the
        # [85] fix. 1.0 on rel factors whose two args are the same var.
        "arg_dup": np.zeros((n, L_FAC), np.float32),
        # gen-15 (2026-07-20): OP_APPLY macro floor (ALG_FTYPES=7) — second
        # digit bank (k2) + ordered second operand pointer (y). Structural
        # entry per the pointer law; gold-fed from birth (two-terminal law).
        "digits2": np.zeros((n, L_FAC, N_DIG), np.int32),
        "is_macro": np.zeros((n, L_FAC), np.float32),
        "is_frac": np.zeros((n, L_FAC), np.float32),   # mg2: FRAC_OF (ftype 7)
        "is_chain": np.zeros((n, L_FAC), np.float32),  # mg3: CHAIN_MUL (ftype 8)
        **({"valspan": np.zeros((n, L_FAC, T_ALG), np.float32)} if ALG_VALATT else {}),
        **({"lsent": np.zeros((n, K_VARS, T_ALG), np.float32)} if ALG_LSENT else {}),
        "y": np.zeros((n, L_FAC), np.int32),
    }
    for i, (smp, offs) in enumerate(zip(samples, offsets)):
        # gen-10 prose rows: factors may carry NO spans (raw prose has no
        # letter anchors) — sort them last; span losses auto-mask on zeros
        facs = sorted(smp["factors"],
                      key=lambda f: (min(s for s, _ in f["spans"])
                                     if f.get("spans") else 10 ** 9))
        assert len(facs) <= L_FAC and smp["n_vars"] <= K_VARS
        g["query"][i] = smp["query_var"]
        g["band"][i] = smp["decisions"]
        _indvars = set()
        for v_str, spans in smp["mentions"].items():
            _spans_to_tokmask(spans, offs, g["vspan"][i, int(v_str)])
            if any(sp[1] - sp[0] > 2 for sp in spans):
                _indvars.add(int(v_str))
            if ALG_REF:   # E-FLOOR (door #43): indirect sites only (char>2)
                _ind = [sp for sp in spans if sp[1] - sp[0] > 2]
                if _ind:
                    _m = np.zeros(T_ALG, np.float32)
                    _spans_to_tokmask(_ind, offs, _m)
                    g["refvar"][i][_m > 0] = int(v_str)
        if ALG_LSENT:
            import re as _re2
            _parts=[(m.start(),m.end()) for m in _re2.finditer(r"[^.]+\.", smp["text"])]
            for _v in range(min(smp.get("n_vars",K_VARS),K_VARS)):
                _nm=chr(ord("a")+_v)
                for _a,_b in _parts:
                    if _re2.search(r"\b"+_nm+r"\b", smp["text"][_a:_b]):
                        _spans_to_tokmask([(_a,_b)], offs, g["lsent"][i, _v])
        for j, f in enumerate(facs):
            g["presence"][i, j] = 1.0
            if ALG_DIAL and f["ftype"] == "rel" and any(a in _indvars for a in f.get("args", [])):
                g["is_ind"][i, j] = 1.0
            _spans_to_tokmask(f.get("spans") or [], offs, g["fspan"][i, j])
            if f["ftype"] == "rel":
                g["ftype"][i, j] = 0
                g["is_rel"][i, j] = 1.0
                g["op"][i, j] = 0 if f["op"] == "add" else 1
                for a in f["args"]:
                    g["args"][i, j, a] = 1.0
                if f["args"][0] == f["args"][1]:
                    g["arg_dup"][i, j] = 1.0
                g["res"][i, j] = f["result"]
            elif f["ftype"] == "given":
                g["ftype"][i, j] = 1
                g["is_lit"][i, j] = 1.0
                if ALG_VALATT and f.get("spans"):
                    _va, _vb = f["spans"][0]
                    import re as _re
                    for _vmm in _re.finditer(r"\d+", smp["text"][_va:_vb]):
                        if int(_vmm.group()) == int(f["value"]):
                            _spans_to_tokmask([(_va + _vmm.start(),
                                                _va + _vmm.end())],
                                              offs, g["valspan"][i, j])
                            break
                g["res"][i, j] = f["var"]
                t = int(f["value"])
                if t < 0:                    # E1: sign gold; digits carry |t|
                    assert ALG_WIDE, "negative literal outside ALG_WIDE"
                    g["sign"][i, j] = 1.0
                    t = -t
                assert t < 10 ** N_DIG
                for d in range(N_DIG):
                    g["digits"][i, j, d] = (t // 10 ** (N_DIG - 1 - d)) % 10
            elif f["ftype"] == "mod":
                g["ftype"][i, j] = 2
                g["is_mod"][i, j] = 1.0
                g["args"][i, j, f["var"]] = 1.0
                g["res"][i, j] = f["result"]
                t = int(f["k"])                # modulus via the digit head
                assert t < 10 ** N_DIG
                for d in range(N_DIG):
                    g["digits"][i, j, d] = (t // 10 ** (N_DIG - 1 - d)) % 10
            elif f["ftype"] == "sel":
                from mycelium.csp_domains import SEL_TO_ID
                g["ftype"][i, j] = 3
                g["is_sel"][i, j] = 1.0
                g["sel"][i, j] = SEL_TO_ID[f["sel"]]
                for a in f["args"]:
                    g["args"][i, j, a] = 1.0
                g["res"][i, j] = f["result"]
            elif f["ftype"] == "pct":
                g["ftype"][i, j] = 4
                g["is_pct"][i, j] = 1.0
                g["args"][i, j, f["args"][0]] = 1.0
                g["res"][i, j] = f["args"][1]
                t = int(f["p"])
                assert t < 10 ** N_DIG
                for d in range(N_DIG):
                    g["digits"][i, j, d] = (t // 10 ** (N_DIG - 1 - d)) % 10
            elif f["ftype"] == "fdiv":
                g["ftype"][i, j] = 5
                g["is_fdiv"][i, j] = 1.0
                g["args"][i, j, f["var"]] = 1.0
                g["res"][i, j] = f["result"]
                t = int(f["k"])
                for d in range(N_DIG):
                    g["digits"][i, j, d] = (t // 10 ** (N_DIG - 1 - d)) % 10
            elif f["ftype"] == "macro" and f.get("name") == "FRAC_OF":
                g["ftype"][i, j] = 7
                g["is_frac"][i, j] = 1.0
                g["args"][i, j, f["x"]] = 1.0
                g["res"][i, j] = f["result"]
                for t_, arr in ((int(f["a"]), "digits"), (int(f["k"]), "digits2")):
                    assert t_ < 10 ** N_DIG
                    for d in range(N_DIG):
                        g[arr][i, j, d] = (t_ // 10 ** (N_DIG - 1 - d)) % 10
            elif f["ftype"] == "macro" and f.get("name") == "CHAIN_MUL":
                g["ftype"][i, j] = 8                     # mg3: the tower pilot
                g["is_chain"][i, j] = 1.0
                for v in f["xs"]:
                    g["args"][i, j, v] = 1.0             # multi-hot xs
                g["res"][i, j] = f["result"]
            elif f["ftype"] == "macro":
                assert f.get("name") == "OP_APPLY"
                g["ftype"][i, j] = 6
                g["is_macro"][i, j] = 1.0
                g["op"][i, j] = 0 if f["op"] == "add" else 1   # add/sub on macro slots
                g["args"][i, j, f["x"]] = 1.0                  # x via args argmax
                g["y"][i, j] = f["y"]                          # y via the fresh pointer
                g["res"][i, j] = f["result"]
                for t_, key, arr in ((int(f["k1"]), "k1", "digits"),
                                     (int(f["k2"]), "k2", "digits2")):
                    assert t_ < 10 ** N_DIG
                    for d in range(N_DIG):
                        g[arr][i, j, d] = (t_ // 10 ** (N_DIG - 1 - d)) % 10
            else:
                raise ValueError(f"unknown gold ftype {f['ftype']!r}")
    if int(os.environ.get("ALG_POSCH", "0")):
        g["depth"] = np.zeros((n, L_FAC), np.int32)
        g["term"] = np.zeros((n, L_FAC), np.float32)
        for ri, smp in enumerate(samples):
            try:
                for j, (d, t) in enumerate(_pos_meta(smp)[:L_FAC]):
                    g["depth"][ri, j] = d; g["term"][ri, j] = t
            except Exception:
                pass
    if int(os.environ.get("ALG_BINDBUS", "0")):
        # THE ROTATIONAL BINDING BUS (2026-08-28, word given): per-slot gold
        # bound vectors — role-phase rotations of var/op codes (VSA in the
        # Fourier domain; codes deterministic from .cache/bindbus_codes.npz)
        _bz = np.load(os.environ.get("BIND_CODES", ".cache/bindbus_codes.npz"))
        _CB = _bz["CB"]; _P = _CB.shape[1] // 2
        def _rotv(v, th):
            v2 = v.reshape(_P, 2)
            c_, s_ = np.cos(th), np.sin(th)
            return np.stack([c_ * v2[:, 0] - s_ * v2[:, 1],
                             s_ * v2[:, 0] + c_ * v2[:, 1]], -1).reshape(-1)
        _TH = {r: _bz[f"theta_{r}"] for r in ("arg1", "arg2", "res", "op")}
        g["bindvec"] = np.zeros((n, L_FAC, _CB.shape[1]), np.float32)
        for i2 in range(n):
            for j2 in range(L_FAC):
                if g["presence"][i2, j2] <= 0: continue
                aidx = np.where(g["args"][i2, j2] > 0)[0]
                if len(aidx) == 0: a1 = a2 = int(g["res"][i2, j2])
                elif len(aidx) == 1: a1 = a2 = int(aidx[0])
                else: a1, a2 = int(aidx[0]), int(aidx[1])
                r2 = int(g["res"][i2, j2]); opv = 24 + min(int(g["ftype"][i2, j2]), 7)
                z = (_rotv(_CB[a1], _TH["arg1"]) + _rotv(_CB[a2], _TH["arg2"])
                     + _rotv(_CB[r2], _TH["res"]) + _rotv(_CB[opv], _TH["op"]))
                g["bindvec"][i2, j2] = z
    if int(os.environ.get("ALG_OPCOUNT", "0")):
        # feed-door completion (2026-08-26): gold is built here, but CACHED
        # npzs from before the surgery lack g_opc — load_alg's consumer
        # must hard-error rather than zero-train (the first opc fire's
        # lesson, now fenced at the missing-gold case too)
        g["opc"] = np.zeros((n, len(OPC_CLASSES)), np.int32)
        for ri, smp in enumerate(samples):
            try:
                g["opc"][ri] = _opc_meta(smp)
            except Exception:
                pass
    return g


def tokenize(path):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(TOKENIZER_JSON)
    samples = [json.loads(l) for l in open(path)]
    ids = np.zeros((len(samples), T_ALG), np.int32)
    mask = np.zeros((len(samples), T_ALG), np.float32)
    offsets = []
    for i, s in enumerate(samples):
        e = tok.encode(s["text"])
        if len(e.ids) > T_ALG:
            raise RuntimeError(f"TRUNCATION {path}:{i} — {len(e.ids)} > {T_ALG}")
        ids[i, :len(e.ids)] = e.ids
        mask[i, :len(e.ids)] = 1.0
        offsets.append(list(e.offsets))
    return samples, ids, mask, offsets


def sent_indices(text, offs, mask_row):
    bounds = []
    i = text.find(". ")
    while i != -1:
        bounds.append(i + 1)
        i = text.find(". ", i + 1)
    out = np.zeros((T_ALG,), np.int32)
    ntk = int(mask_row.sum())
    arr = np.asarray(offs[:min(ntk, T_ALG)], dtype=np.int64)
    if len(arr):
        idx = np.searchsorted(np.asarray(bounds, dtype=np.int64), arr[:, 0], "right")
        out[:len(arr)] = np.minimum(idx, SENT_MAX - 1)
    return out


def tails_of(sent_rows):
    out = np.zeros_like(sent_rows, dtype=np.float32)
    for ri in range(sent_rows.shape[0]):
        s = sent_rows[ri]; i = 0
        while i < len(s):
            j = i
            while j + 1 < len(s) and s[j + 1] == s[i]: j += 1
            out[ri, i + (j - i + 1) // 2:j + 1] = 1.0
            i = j + 1
    return out


def build_slot_masks(o_np, sent_rows):
    """Evidence-sharing slot mask (B, L, L) from the model's OWN breath-0
    outputs (deployable): same-sentence (attention-argmax sentence) OR
    shared-variable (top-2 args + res argmax overlap) OR self."""
    B = o_np["fat"].shape[0]
    masks = np.zeros((B, L_FAC, L_FAC), np.float32)
    for bi in range(B):
        tok_star = o_np["fat"][bi].argmax(-1)              # (L,)
        s_id = sent_rows[bi][np.minimum(tok_star, T_ALG - 1)]
        same = s_id[:, None] == s_id[None, :]
        M = np.zeros((L_FAC, K_VARS), bool)
        for j in range(L_FAC):
            for a in np.argsort(-o_np["args"][bi, j])[:2]:
                M[j, a] = True
            M[j, int(o_np["res"][bi, j].argmax())] = True
        shared = (M.astype(np.int32) @ M.astype(np.int32).T) > 0
        masks[bi] = (same | shared | np.eye(L_FAC, dtype=bool)).astype(np.float32)
    return masks


# ===========================================================================
# PRECOMPUTE (GPU once) — small corpus, plain npz
# ===========================================================================

def do_precompute():
    from tinygrad import Tensor, dtypes
    from mycelium.llama_loader import (
        attach_llama_layers, load_llama_weights, LLAMA_3_2_1B_CFG, _rms_norm)

    class _H:
        pass
    host = _H()
    sd = load_llama_weights(os.path.join(_ROOT, ".cache/llama-3.2-1b-weights/model.safetensors"))
    attach_llama_layers(host, n_layers=4, sd=sd, cfg=LLAMA_3_2_1B_CFG)
    del sd
    jobs = [(TRAIN_NAME, ALG_TRAIN), (TEST_NAME, ALG_TEST)]
    if os.environ.get("PRECOMPUTE_ONLY"):
        jobs = [(n_, p_) for n_, p_ in jobs if n_ == os.environ["PRECOMPUTE_ONLY"]]
    for split, path in jobs:
        samples, ids, mask, offsets = tokenize(path)
        n = len(samples)
        # states stream straight to a disk-backed memmap — holding the full
        # (n, T, 2048) fp16 array in RAM beside the AM driver's pinned pages
        # OOMs at gen-7 scale (three kills, all during the giant-array write)
        states = np.lib.format.open_memmap(
            STATES_NPY.format(split=split), mode="w+", dtype=np.float16,
            shape=(n, T_ALG, H_TRUNK))
        for s0 in range(0, n, 8):
            sl = slice(s0, min(s0 + 8, n))
            x = host.llama_embed[Tensor(ids[sl], dtype=dtypes.int)]
            for layer in host.llama_layers:
                x = layer(x, host.llama_rope_cos, host.llama_rope_sin)
            x = _rms_norm(x, host.llama_layers[-1].ffn_norm, host.llama_cfg.rms_norm_eps)
            c = x.cast(dtypes.float).realize().numpy()
            assert np.isfinite(c).all()
            states[sl] = c.astype(np.float16)
            if (s0 // 8) % 200 == 0:
                print(f"[precompute] {split}: batch {s0//8}/{(n+7)//8}", flush=True)
        states.flush()
        shape = states.shape
        del states
        gold = build_gold(samples, offsets)
        sent = np.stack([sent_indices(s["text"], o, mask[i])
                         for i, (s, o) in enumerate(zip(samples, offsets))])
        np.savez(STATES_NPZ.format(split=split),
                 tokmask=mask.astype(np.uint8), sent=sent.astype(np.int8),
                 **{f"g_{k}": v for k, v in gold.items()})
        print(f"[precompute] {split}: {shape} (states -> memmap npy)", flush=True)


def load_alg(split):
    is_train = split == "train"
    split = TRAIN_NAME if is_train else (TEST_NAME if split == "test" else split)
    z = np.load(STATES_NPZ.format(split=split))
    _mix_path = ALG_TRAIN if is_train else ALG_TEST
    # THE SHA-FENCE (2026-08-10): staged arrays are a JOIN against the mix
    # file, keyed by TRAIN_NAME — a name, not content. Three fossils rode
    # that gap (the ALG_TRAIN_NAME replay trained on another mix's arrays
    # to the digit). Stamped npz -> hard-assert; unstamped -> say so loudly.
    if "mix_sha" in z.files:
        from mycelium.era import mix_sha16
        _want = str(z["mix_sha"])
        _got = mix_sha16(_mix_path)
        assert _got == _want, (
            f"SHA-FENCE: staged arrays were built from a different mix "
            f"(npz stamp {_want} != {_mix_path} sha {_got}) — the "
            f"name-key points at the wrong content; re-assemble.")
    elif is_train:
        print(f"[sha-fence] UNSTAMPED train arrays for '{split}' — "
              f"fence blind here until re-assembled", flush=True)
    samples = [json.loads(l) for l in open(_mix_path)]
    # CUSTODY-GOLD GUARD (deep clean 2026-07-28): load_alg is the single door
    # every battery consumer's gold flows through, and ALG_TRAIN/ALG_TEST env
    # is the only redirection mechanism. Pen rows (delegated book annotations,
    # gen.src_idx present) carry solution vectors that are pen-side scratch —
    # NEVER custody-side gold (543/550 banked book rows hold all-zero vectors
    # that disagree with harvest gold). Downstream code reads
    # sample["solution"][query_var] as gold in ~60 places; rather than patch
    # sixty readers, refuse the poison at the door.
    # Train side: pen rows are LAWFUL diet members (prose dose; supervision
    # comes from the states-file gold arrays, not the solution field) — warn
    # loudly so the count is visible. Test side: solution-as-gold judges
    # verdicts downstream, so pen rows REFUSE at the door.
    n_pen = sum(1 for s in samples if "src_idx" in s.get("gen", {}))
    if n_pen and is_train:
        # WARN-SIDE FENCE (bench 2026-07-28): an ambient warning becomes the
        # new silent hazard in six weeks. Pen rows on the train side are an
        # ERROR unless the launch explicitly acknowledges the diet via
        # ALG_ALLOW_PEN_TRAIN=1 (set by diet-aware trainer launches only).
        if os.environ.get("ALG_ALLOW_PEN_TRAIN") == "1":
            print(f"[load_alg] diet acknowledged: {n_pen} pen rows in "
                  f"ALG_TRAIN (solution fields are NOT gold — custody law)")
        else:
            raise RuntimeError(
                f"load_alg: {n_pen} PEN rows in ALG_TRAIN without "
                f"ALG_ALLOW_PEN_TRAIN=1 — if this is a deliberate diet mix, "
                f"set the env in the launch script; otherwise this path is "
                f"battery-adjacent and pen rows are an error (custody-gold "
                f"law, warn-side fence)")
    elif n_pen:
        raise RuntimeError(
            f"load_alg: {n_pen} PEN rows (gen.src_idx present) in ALG_TEST — "
            f"pen solution fields are not gold; route through "
            f"mycelium.custody_gold.row_gold or use a mint fixture "
            f"(docs/ARITH3_DIALECT.md, custody-gold law)")
    gold = {k[2:]: z[k] for k in z.files if k.startswith("g_")}
    if os.path.exists(STATES_NPY.format(split=split)):
        states = np.load(STATES_NPY.format(split=split), mmap_mode="r")
        # SAMPLES-STATES DESYNC GUARD (deep clean 2026-07-30): samples come
        # from ALG_TRAIN/ALG_TEST while states come from ALG_TRAIN_NAME's
        # files — two env vars with no cross-check. A forgotten
        # ALG_TRAIN_NAME would silently pair states[i] with samples[i] from
        # two unrelated corpora. Fail loudly instead.
        if len(states) != len(samples):
            raise RuntimeError(
                f"load_alg({split}): states file has {len(states)} rows but "
                f"the samples jsonl has {len(samples)} — ALG_TRAIN_NAME and "
                f"ALG_TRAIN point at different corpora (desync guard)")
    else:
        states = z["states"]   # legacy artifacts (gen<=6): states inside npz
    return samples, states, z["tokmask"], gold, z["sent"]


# ===========================================================================
# MODEL — two slot banks + bilinear pointers (all supervised)
# ===========================================================================

def build_params(seed=0):
    from tinygrad import Tensor, dtypes
    rng = np.random.RandomState(seed)

    def t(a):
        x = Tensor(a.astype(np.float32), dtype=dtypes.float,
                   requires_grad=True).contiguous().realize()
        x.requires_grad = True
        return x

    def lin(i, o):
        return t(rng.randn(i, o) / math.sqrt(i)), t(np.zeros((o,)))

    p = {}
    p["waist_w"], p["waist_b"] = lin(H_TRUNK, H_W)
    if ALG_SEPHASE_W:
        _ww = p["waist_w"].numpy()
        _ww += phase_alphabet(H_TRUNK, H_W, 1.0 / math.sqrt(H_TRUNK), rng) * 0.5
        p["waist_w"] = t(_ww)
    if int(os.environ.get("ALG_SEPHASE", "0")):
        # the properly-wired receiver (2026-08-17): six-phase structure seeded
        # INTO the learned sync channel (init, not bias — the model may keep
        # or reshape it; amplitude matches the native init scale)
        if int(os.environ.get("SEPHASE_SCRAMBLE", "0")):
            _s6 = np.random.RandomState(9).uniform(0, 2 * np.pi, SENT_MAX)
        else:
            _s6 = np.arange(SENT_MAX) % 6 * (np.pi / 3.0)
        _d6 = np.arange(H_W) * (2 * np.pi / H_W)
        _se = 0.1 * np.sqrt(2.0) * np.cos(_s6[:, None] + _d6[None, :] * 6)
        p["sent_emb"] = t(_se + rng.randn(SENT_MAX, H_W) * 0.02)
    else:
        p["sent_emb"] = t(rng.randn(SENT_MAX, H_W) * 0.1)
    if ALG_SEPHASE_Q:
        if SEPHASE_Q_SCRAMBLE:
            _rs = np.random.RandomState(227)
            def _scram(n, dims, scale):
                _ph = _rs.uniform(0, 2 * np.pi, n)
                _d = np.arange(dims)
                _pt = scale * np.sqrt(2.0) * np.cos(_ph[:, None] + _d[None, :] * (2 * np.pi / dims) * (np.arange(n)[:, None] + 1))
                return (_pt + rng.randn(n, dims) * scale * 0.2).astype(np.float32)
            p["vq"] = t(_scram(K_VARS, H_W, 0.02))
            p["fq"] = t(_scram(L_FAC, H_W, 0.02))
        else:
            p["vq"] = t(phase_alphabet(K_VARS, H_W, 0.02, rng))
            p["fq"] = t(phase_alphabet(L_FAC, H_W, 0.02, rng))
    else:
        p["vq"] = t(rng.randn(K_VARS, H_W) * 0.02)
        p["fq"] = t(rng.randn(L_FAC, H_W) * 0.02)
    p["qq"] = t(rng.randn(1, H_W) * 0.02)
    for nm in ("wq", "wk", "wv", "wo"):
        p[f"attn_{nm}"], p[f"attn_{nm}_b"] = lin(H_W, H_W)
    p["ffn_w1"], p["ffn_b1"] = lin(H_W, 2 * H_W)
    p["ffn_w2"], p["ffn_b2"] = lin(2 * H_W, H_W)
    p["h_pres"], p["h_pres_b"] = lin(H_W, 1)
    # ALG2=1 -> tranche geometry (4-way ftype + selector head). Default keeps
    # the legacy 2-way build BYTE-COMPATIBLE with deployed checkpoints — every
    # lattice script loads the old ckpt through here.
    if int(os.environ.get("ALG2", "0")):
        nft = int(os.environ.get("ALG_FTYPES", "4"))  # 6 = +pct/fdiv (T2)
        p["h_ftype"], p["h_ftype_b"] = lin(H_W, nft)
        p["h_sel"], p["h_sel_b"] = lin(H_W, 4)       # larger/smaller/even/odd
    else:
        p["h_ftype"], p["h_ftype_b"] = lin(H_W, 2)
    p["h_op"], p["h_op_b"] = lin(H_W, 2)
    if int(os.environ.get("ALG_DUP", "0")):   # gen-9: arg-multiplicity bit
        p["h_dup"], p["h_dup_b"] = lin(H_W, 1)
    p["h_islit"], p["h_islit_b"] = lin(H_W, 1)
    p["h_dig"], p["h_dig_b"] = lin(H_W, N_DIG * 10)
    if ALG_WIDE:                              # E1: the sign terminal
        p["h_sgn"], p["h_sgn_b"] = lin(H_W, 1)
    if ALG_REF:                               # E-FLOOR: token-grain referent
        p["h_ref"], p["h_ref_b"] = lin(H_W, K_VARS)
    if int(os.environ.get("ALG2", "0")) and \
            int(os.environ.get("ALG_FTYPES", "4")) >= 7:   # gen-15: OP_APPLY
        p["h_dig2"], p["h_dig2_b"] = lin(H_W, N_DIG * 10)
        p["W_y"] = t(rng.randn(H_W, H_W) / math.sqrt(H_W))
    K_B = int(os.environ.get("ALG_BREATH", "1"))
    if K_B > 1:   # BRICK-P: masked slot-to-slot breathing (2026-07-09)
        if ALG_SEPHASE_SETTLE:   # the settle transceiver: slot-to-slot
            _stp = phase_alphabet(H_W, H_W, 1.0 / math.sqrt(H_W), rng)
            p["W_bq"] = t(_stp + rng.randn(H_W, H_W).astype(np.float32) / math.sqrt(H_W) * 0.2)
            p["W_bq_b"] = t(np.zeros((H_W,)))
            p["W_bk"] = t(_stp + rng.randn(H_W, H_W).astype(np.float32) / math.sqrt(H_W) * 0.2)
            p["W_bk_b"] = t(np.zeros((H_W,)))
        else:
            p["W_bq"], p["W_bq_b"] = lin(H_W, H_W)
        if not ALG_SEPHASE_SETTLE:
            p["W_bk"], p["W_bk_b"] = lin(H_W, H_W)
        p["W_bv"], p["W_bv_b"] = lin(H_W, H_W)
        p["W_bo"] = t(np.zeros((H_W, H_W)))          # zero-init: breath deltas
        p["W_bo_b"] = t(np.zeros(H_W))               # start silent
        if int(os.environ.get("ALG_SEPHASE_B", "0")):
            # sephase-B (2026-08-18): the temporal sibling — breath_emb born
            # with orthogonal phase stamps at native amplitude (init, the
            # winning idiom; SEPHASE_B_SCRAMBLE randomizes phases, placebo)
            _bk = np.arange(K_B)
            if int(os.environ.get("SEPHASE_B_SCRAMBLE", "0")):
                _bph = np.random.RandomState(227).uniform(0, 2 * np.pi, K_B)
            else:
                _bph = _bk * np.pi / max(K_B, 1)
            _bd = np.arange(H_W)
            if SEPHASE_B_BAND == 2:  # the 1/3 dose: stride-2 bands
                _be = 0.02 * np.sqrt(2.0 / 3.0) * sum(   # {2k,2k+1,2k+2} —
                    np.cos(_bph[:, None] + _bd[None, :] * (2 * np.pi / H_W)
                           * (2 * _bk[:, None] + 1 + _f)) for _f in range(3))
            elif SEPHASE_B_BAND:  # the orientation form: bands {k,k+1,k+2} —
                _be = 0.02 * np.sqrt(2.0 / 3.0) * sum(   # adjacent corr ~2/3
                    np.cos(_bph[:, None] + _bd[None, :] * (2 * np.pi / H_W)
                           * (_bk[:, None] + 1 + _f)) for _f in range(3))
            else:
                _be = 0.02 * np.sqrt(2.0) * np.cos(
                    _bph[:, None] + _bd[None, :] * (2 * np.pi / H_W) * (_bk[:, None] + 1))
            p["breath_emb"] = t(_be + rng.randn(K_B, H_W) * 0.004)
        else:
            p["breath_emb"] = t(rng.randn(K_B, H_W) * 0.02)
        p["breath_gate"] = t(np.full(K_B, float(os.environ.get(
            "BREATH_GATE_INIT", "-2.0"))))   # init-closed convex blend;
            # door #44-RESCUE: the organ does not self-open (gates at init
            # after 4k) — BREATH_GATE_INIT=0.0 opens the door at birth
        if int(os.environ.get("ALG_RINGS", "0")):    # RUNG-3 v1: the soft pawl
            p["W_cmt"] = t(np.zeros((H_W, 1)))       # zero-init commit head
            p["W_cmt_b"] = t(np.full(1, float(os.environ.get(
                "CMT_B_INIT", "-4.0"))))             # init: commit ~nothing;
                                                     # door #54-R: bias-open
        pass
    if int(os.environ.get("ALG_BINDBUS", "0")):
        _bd = int(os.environ.get("ALG_BIND_D", "128"))
        if int(os.environ["ALG_BINDBUS"]) >= 6:
            # v6 the middle door: SHARED 512-gelu trunk (load-bearing, the
            # v5 kill's finding) + per-role OUTPUT projections only
            p["W_bind1"] = t(rng.randn(H_W, 512) / math.sqrt(H_W))
            p["W_bind1_b"] = t(np.zeros(512))
            for _r6 in ("a1", "a2", "rs", "op"):
                p[f"W6_{_r6}"] = t(rng.randn(512, 32) / math.sqrt(512))
        elif int(os.environ["ALG_BINDBUS"]) >= 5:
            # v5: four role heads (docs/rotational_bus.md) — heads point,
            # frozen phasors bind, the wire sums (single-wire fence)
            for _r5 in ("a1", "a2", "rs", "op"):
                p[f"W5_{_r5}_1"] = t(rng.randn(H_W, 256) / math.sqrt(H_W))
                p[f"W5_{_r5}_b"] = t(np.zeros(256))
                p[f"W5_{_r5}_2"] = t(rng.randn(256, 32) / math.sqrt(256))
        elif int(os.environ["ALG_BINDBUS"]) >= 2:
            p["W_bind1"] = t(rng.randn(H_W, 512) / math.sqrt(H_W))
            p["W_bind1_b"] = t(np.zeros(512))
            p["W_bind2"] = t(rng.randn(512, _bd) / math.sqrt(512))
        else:
            p["W_bind"] = t(rng.randn(H_W, _bd) / math.sqrt(H_W))
    if int(os.environ.get("ALG_RINGS", "0")):
        if True:
            if int(os.environ.get("ALG_CMT_REG", "0")):
                p["w_cmt_reg"] = t(np.zeros(1))      # zero-init: register
                                                     # enters at no-effect
    p["W_args"] = t(rng.randn(H_W, H_W) / math.sqrt(H_W))
    if int(os.environ.get("ALG_DUPPTR", "0")):
        p["W_dargs"] = t(rng.randn(H_W, H_W) / math.sqrt(H_W))
    if ALG_DIAL:                              # door #45: dialect args pointer
        p["W_iargs"] = t(rng.randn(H_W, H_W) / math.sqrt(H_W))
    p["W_res"] = t(rng.randn(H_W, H_W) / math.sqrt(H_W))
    p["W_query"] = t(rng.randn(H_W, H_W) / math.sqrt(H_W))
    if ALG_SIXWAVE:      # door #62: carrier gate — structure enters at zero
        p["sw_g"] = t(np.zeros((1,)))
    if ALG_NOTEBOOK:     # the cathedral (2026-08-18)
        if ALG_SEPHASE_PAIR:   # the transceiver: ink and query born in the
            _shared = phase_alphabet(H_W, H_W, 1.0 / math.sqrt(H_W), rng)
            p["W_sil"] = t(_shared + rng.randn(H_W, H_W).astype(np.float32) / math.sqrt(H_W) * 0.2)
            p["W_nq"] = t(_shared + rng.randn(H_W, H_W).astype(np.float32) / math.sqrt(H_W) * 0.2)
        else:                   # same coordinate code
            p["W_sil"] = t(rng.randn(H_W, H_W) / math.sqrt(H_W))
            p["W_nq"] = t(rng.randn(H_W, H_W) / math.sqrt(H_W))
    if int(os.environ.get("ALG_POSCH", "0")):
        # THE POSITION CHANNEL (2026-08-24, word given; deficit twice-banked:
        # ladder depth-at-chance + A0 balanced .222): supervised depth/term
        # terminals — position TAUGHT into the states (gold outcomes, the
        # lawful side of the Goodhart boundary)
        p["h_depth"] = t(rng.randn(H_W, 6) / math.sqrt(H_W))
        p["h_depth_b"] = t(np.zeros(6))
        p["h_term"] = t(rng.randn(H_W, 1) / math.sqrt(H_W))
        p["h_term_b"] = t(np.zeros(1))
    if int(os.environ.get("ALG_OPCOUNT", "0")):
        # LEVER 3 (2026-08-25, word given): the op-multiset COUNT head —
        # pooled waist -> per-op-class count logits, supervised at the ROW
        # grain from mint-source gold (whole-row exactness is where
        # enumeration lives; per-slot decode compounds, a count read doesn't)
        p["W_opc1"] = t(rng.randn(H_W, 256) / math.sqrt(H_W))
        p["W_opc1_b"] = t(np.zeros(256))
        p["W_opc2"] = t(rng.randn(256, len(OPC_CLASSES) * (OPC_CAP + 1))
                        / math.sqrt(256))
        p["W_opc2_b"] = t(np.zeros(len(OPC_CLASSES) * (OPC_CAP + 1)))
    if int(os.environ.get("ALG_TRUNK_LORA", "0")):
        # THE TRUNK-COPY LoRA (2026-08-22, constitutional word): rank-r
        # adapters on a RUNTIME copy of L0-L3 (wq/wo/wdown — the hook
        # LlamaLayer already carries). Base trunk safetensors NEVER touched;
        # research lineage only; zero-init B = zero delta at step 0.
        _lr_r = int(os.environ.get("ALG_LORA_R", "16"))
        _span = os.environ.get("ALG_LORA_SPAN", "0123")   # pool axis: layers
        _proj = os.environ.get("ALG_LORA_PROJ", "all")    # pool axis: all|wq|wod
        _pset = {"all": ("wq", "wo", "wdown"), "wq": ("wq",),
                 "wod": ("wo", "wdown")}[_proj]
        _lrng = np.random.RandomState(seed + 7000)
        for _li in range(4):
            if str(_li) not in _span: continue
            for _nm, _din in (("wq", 2048), ("wo", 2048), ("wdown", 8192)):
                if _nm not in _pset: continue
                p[f"lora{_li}_{_nm}_A"] = t(_lrng.randn(_din, _lr_r) * 0.01)
                p[f"lora{_li}_{_nm}_B"] = t(np.zeros((_lr_r, 2048)))
    return p


NB_STAMPS = None
if ALG_NOTEBOOK:
    assert int(os.environ.get("ALG_BREATH", "1")) <= 8, "NB_STAMPS holds 8 rows (audit #11)"
    _ks = np.arange(8)
    _ds = np.arange(512)
    NB_STAMPS = np.cos(_ks[:, None] * np.pi / 8.0 * 7
                       + _ds[None, :] * (2 * np.pi / 512)
                       * (_ks[:, None] + 1)).astype(np.float32)
    NB_STAMPS /= np.linalg.norm(NB_STAMPS, axis=1, keepdims=True)
    _cc = np.abs(NB_STAMPS @ NB_STAMPS.T - np.eye(8)).max()
    assert _cc < 0.35, f"sharpness assert FAILED: stamp cos {_cc:.3f}"


def forward(p, trunk, tokmask, sent, slot_mask=None, revoke=None, tail=None, drop=None, anchor=None, amask=None, gmod=None, pmask=None, lsent=None, reg=None):
    B = trunk.shape[0]
    waist = (trunk @ p["waist_w"] + p["waist_b"]).gelu() + p["sent_emb"][sent]

    def bank(queries, nq, extra=None, pbias=None):
        q_in = queries.unsqueeze(0) + (extra if extra is not None else 0)
        q = q_in @ p["attn_wq"] + p["attn_wq_b"]
        k = waist @ p["attn_wk"] + p["attn_wk_b"]
        v = waist @ p["attn_wv"] + p["attn_wv_b"]
        hd = H_W // N_HEADS
        qh = q.reshape(B if extra is not None else 1, nq, N_HEADS, hd).permute(0, 2, 1, 3)
        kh = k.reshape(B, -1, N_HEADS, hd).permute(0, 2, 1, 3)
        vh = v.reshape(B, -1, N_HEADS, hd).permute(0, 2, 1, 3)
        sc = (qh @ kh.transpose(-2, -1)) / math.sqrt(hd)
        if pbias is not None:   # door #62: six-wave phase-resonance bias
            sc = sc + pbias
        sc = sc.clip(-1e4, 1e4) + (1.0 - tokmask.reshape(B, 1, 1, -1)) * -1e4
        at = sc.softmax(-1)
        st = (at @ vh).permute(0, 2, 1, 3).reshape(B, nq, H_W)
        st = st @ p["attn_wo"] + p["attn_wo_b"] + q_in.reshape(-1, nq, H_W)
        st = st + ((st @ p["ffn_w1"] + p["ffn_b1"]).gelu() @ p["ffn_w2"] + p["ffn_b2"])
        return st, at.mean(1)

    _lb = None
    if lsent is not None:               # V2: letter-keyed partition — imposed
        _lb = lsent.reshape(B, 1, K_VARS, -1) * float(os.environ.get("LS_A", "1.0"))
    vst, vat = bank(p["vq"], K_VARS, pbias=_lb)
    _pb = None
    _pb_prior = None
    _sync = None
    _swturn = None
    if ALG_SYNC:
        from tinygrad import Tensor, dtypes
        _A = float(os.environ.get("SYNC_A", "1.0"))
        _phi = np.pi / 3.0 * (np.arange(L_FAC) % 6)
        _cph = Tensor(np.cos(_phi).astype(np.float32), dtype=dtypes.float)
        _sph = Tensor(np.sin(_phi).astype(np.float32), dtype=dtypes.float)
        _th0 = (sent - (sent // 6) * 6).float() * (math.pi / 3.0)
        _cth, _sth = _th0.cos(), _th0.sin()
        _scr = None
        if int(os.environ.get("SYNC_SCRAMBLE", "0")):
            _scr = np.random.RandomState(227).uniform(0, 2 * np.pi, 8)
        def _mk_pb(kb):
            _d = float(_scr[kb]) if _scr is not None else kb * (math.pi / 3.0)
            ck, sk = math.cos(_d), math.sin(_d)
            cthk = _cth * ck - _sth * sk
            sthk = _sth * ck + _cth * sk
            return (_cph.reshape(1, 1, L_FAC, 1) * cthk.reshape(B, 1, 1, -1)
                    + _sph.reshape(1, 1, L_FAC, 1)
                    * sthk.reshape(B, 1, 1, -1)) * _A
        _php = np.zeros((L_FAC, H_W), np.float32)
        def _mk_osc(kb):   # the receiver's local oscillator — same clock
            _d = float(_scr[kb]) if _scr is not None else kb * (math.pi / 3.0)
            _o = _php.copy()
            _o[:, 0] = np.cos(_phi + _d) * _A
            _o[:, 1] = np.sin(_phi + _d) * _A
            return Tensor(_o, dtype=dtypes.float).reshape(1, L_FAC, H_W)
        _sync = (_mk_pb, _mk_osc)
        _pb = _mk_pb(0)
    if ALG_SIXWAVE:
        from tinygrad import Tensor, dtypes
        # six helical carriers: token phase from sentence index (mod 6, 60
        # deg apart, antiphase pairs); slot phase from slot index mod 6.
        # Resonance bias cos(phi_slot - theta_tok) enters the factor bank's
        # scores through the zero-init gate — structure, never supervision.
        if int(os.environ.get("SW_SCRAMBLE", "0")):
            _tab = Tensor(np.random.RandomState(227).randint(0, 6, 16)
                          .astype(np.float32) * (math.pi / 3.0), dtype=dtypes.float)
            _th = _tab[sent - (sent // 16) * 16]
        else:
            _th = (sent - (sent // 6) * 6).float() * (math.pi / 3.0)
        _phi = np.pi / 3.0 * (np.arange(L_FAC) % 6)
        _cph = Tensor(np.cos(_phi).astype(np.float32), dtype=dtypes.float)
        _sph = Tensor(np.sin(_phi).astype(np.float32), dtype=dtypes.float)
        _sw_term = (_cph.reshape(1, 1, L_FAC, 1) * _th.cos().reshape(B, 1, 1, -1)
               + _sph.reshape(1, 1, L_FAC, 1)
               * _th.sin().reshape(B, 1, 1, -1)) * p["sw_g"].reshape(1, 1, 1, 1)
        _pb = _sw_term if _pb is None else _pb + _sw_term  # audit #6: adds
        if int(os.environ.get("ALG_SWTURN", "0")):
            # R4-alpha (2026-08-29, word given): THE HELIX TURNS — sentence
            # phase advances one 60deg bucket per breath cycle inside the
            # breath loop (time-keyed recirculation). Zero new params;
            # sw_g shared; ALG_SWTURN=0 is byte-identical to static.
            def _mk_turn(_kb):
                _s6 = sent + (_kb + 1)
                _tht = (_s6 - (_s6 // 6) * 6).float() * (math.pi / 3.0)
                return (_cph.reshape(1, 1, L_FAC, 1)
                        * _tht.cos().reshape(B, 1, 1, -1)
                        + _sph.reshape(1, 1, L_FAC, 1)
                        * _tht.sin().reshape(B, 1, 1, -1)) \
                    * p["sw_g"].reshape(1, 1, 1, 1)
            _swturn = _mk_turn
    if pmask is not None:                 # A0: imposed route-mask (wiring,
        _pb = pmask if _pb is None else _pb + pmask   # not knobs — no grad)
    fst, fat = bank(p["fq"], L_FAC, pbias=_pb)
    qst, _qa = bank(p["qq"], 1)

    # BRICK-P breathing (2026-07-09): K-1 refinement passes. Each breath
    # re-reads the text conditioned on current beliefs (bank-with-extra) +
    # MASKED slot-to-slot settling — evidence-sharing topology AS STRUCTURE
    # (the v98 escape; free-form slot attention is the perceiver trap and is
    # not built). Deltas enter via zero-init W_bo + init-closed gates:
    # at init the K-breath output is byte-identical to the incumbent.
    K_B = int(os.environ.get("ALG_BREATH", "1"))
    breaths = [fst]
    RINGS = int(os.environ.get("ALG_RINGS", "0")) and "W_cmt" in p
    # ORGAN-2: the reverse gear (spec §2). Release DYNAMICS only — the
    # revoke signal is an INPUT PORT: transport arrives solver-side, from
    # outside the neural partition (#152's leading candidate). Rates are
    # PINNED (2-3 breath release scale; #136), never tuned by feel. No
    # new params: no new terminal; the register clause holds (no settle,
    # no entropy — revoke is a named contradiction or nothing).
    XOUT = RINGS and int(os.environ.get("ALG_XOUT", "0"))
    XARM = os.environ.get("ALG_XARM", "dump")        # dump|graded|elastic
    XR_GRADED, XR_ELASTIC = 0.5, 0.15                # pinned, breath-scale
    if RINGS:
        m_c = (fst * 0.0).sum(-1, keepdim=True)      # (B,L_FAC,1) zeros
        anchor = fst
        cmt_logits = []
        x_rel = m_c                                   # released-mass ledger
    if K_B > 1 and slot_mask is not None and "W_bo" in p:
        cur = fst
        for kb in range(1, K_B):
            if ALG_NOTEBOOK and kb == 1:
                from tinygrad import Tensor as _T2, dtypes as _dt2
                _nb_st = _T2(NB_STAMPS, dtype=_dt2.float)
                _nb = [(cur @ p["W_sil"]) if NB_PERSLOT
                       else (cur.mean(1) @ p["W_sil"])]   # sharp vs blurred ink
            q_extra = cur + p["breath_emb"][kb].reshape(1, 1, -1)
            if ALG_NOTEBOOK:
                if NB_PERSLOT:      # per-slot lanes: each slot queries the
                    _q = cur @ p["W_nq"]              # shelf and reads ITS OWN
                    _sc = (_q @ _nb_st[:len(_nb)].transpose(1, 0)) / math.sqrt(H_W)
                    if NB_FOCAL > 0:
                        _sc = _sc * NB_FOCAL          # the magnifying glass
                    _at = _sc.softmax(-1)             # (B, L, k)
                    _rd = sum(_at[:, :, j:j + 1] * _nb[j] for j in range(len(_nb)))
                    q_extra = q_extra + _rd           # (B, L, H) — no blur
                else:
                    _q = cur.mean(1) @ p["W_nq"]
                    _sc = (_q @ _nb_st[:len(_nb)].transpose(1, 0)) / math.sqrt(H_W)
                    if NB_FOCAL > 0:
                        _sc = _sc * NB_FOCAL          # the magnifying glass
                    _at = _sc.softmax(-1)
                    _rd = sum(_at[:, j:j + 1] * _nb[j] for j in range(len(_nb)))
                    q_extra = q_extra + _rd.reshape(B, 1, -1)
                if ALG_STELLAR:                        # cell-3b: the twist in
                    _w = math.cos(kb * math.pi / (2 * K_B)) ** 2   # geometry —
                    cur = _w * cur + (1 - _w) * _rd.reshape(B, 1, -1)
                    q_extra = cur + p["breath_emb"][kb].reshape(1, 1, -1) + _rd.reshape(B, 1, -1)
                                                          # no cliff, no gate
                if ALG_CIRCLE and kb == NB_H + 1:     # the traffic circle:
                    cur = cur * 0.0 + _rd.reshape(B, 1, -1)   # residual severed
                    q_extra = (cur + p["breath_emb"][kb].reshape(1, 1, -1)
                               + _rd.reshape(B, 1, -1))       # memory the road
            if _sync is not None:   # sync-complete: transmitter ON during
                q_extra = q_extra + _sync[1](kb)     # settle; receiver locked
            h_tok, fat_cur = bank(p["fq"], L_FAC, extra=q_extra,
                                  pbias=(_sync[0](kb) if _sync is not None
                                         else _swturn(kb)
                                         if _swturn is not None else None))
            bq = cur @ p["W_bq"] + p["W_bq_b"]
            bk = cur @ p["W_bk"] + p["W_bk_b"]
            bv = cur @ p["W_bv"] + p["W_bv_b"]
            sc2 = (bq @ bk.transpose(-2, -1)) / math.sqrt(H_W)
            sc2 = sc2.clip(-1e4, 1e4) + (1.0 - slot_mask) * -1e4
            if RINGS and int(os.environ.get("ALG_BEXIT", "0")):
                # BEAM EXIT (door #8): committed slots leave the mixer as
                # keys, proportional to mass — soft, init-closed (m starts 0)
                sc2 = sc2 + m_c.reshape(B, 1, L_FAC) * -8.0
            h_slot = (sc2.softmax(-1) @ bv) @ p["W_bo"] + p["W_bo_b"]
            # ABLATION arms (2026-07-10): zero-mult keeps every param in the
            # graph (defined zero grads — the None-grad lesson, applied)
            arm = os.environ.get("ALG_BREATH_ARM", "both")
            if arm == "tok":
                h_slot = h_slot * 0.0
            elif arm == "slot":
                h_tok = h_tok * 0.0
            elif arm == "depth":
                # the decider control: plain per-slot MLP second pass — same
                # params repurposed, NO attention, no mask, no re-read
                h_tok = h_tok * 0.0
                h_slot = h_slot * 0.0 + ((cur @ p["W_bq"] + p["W_bq_b"])
                                         .gelu() @ p["W_bv"] + p["W_bv_b"]) \
                    @ p["W_bo"] + p["W_bo_b"]
            g = p["breath_gate"][kb].sigmoid()
            if drop is not None:            # door #52: BREATH DROPOUT —
                g = g * drop                # per-STEP coin; drop=0 makes the
                                            # breath an exact identity (silent)
            if gmod is not None:            # NAZARE (B)-site smoke: per-slot
                g = g * gmod                # authority INSIDE the loop
            cur_new = cur + g * (h_tok + h_slot - cur)
            if RINGS:  # the soft pawl: monotone commitment mass + anchor
                cl = (cur_new @ p["W_cmt"] + p["W_cmt_b"])     # (B,L_FAC,1)
                if reg is not None and "w_cmt_reg" in p:
                    # REGISTER-AWARE COMMITMENT (2026-08-26, word given):
                    # the mouth's length-corrected read as the pawl's INPUT
                    # — commit boldly in-register, reluctantly on the
                    # frontier. The mouth stays out of every loss (Goodhart
                    # fence); the pawl is handed the map, not the meter.
                    cl = cl + reg.reshape(B, 1, 1) * p["w_cmt_reg"]
                cmt_logits.append(cl.squeeze(-1))
                if XOUT:  # release BEFORE the pawl: the same-breath commit
                    # pressure is the graded arm's RESISTING term (#150)
                    rel = m_c * 0.0
                    if XARM == "elastic":            # standing leak toward
                        rel = rel + XR_ELASTIC * m_c  # rest; self-resetting
                    if revoke is not None:
                        rv = revoke.reshape(B, L_FAC, 1)
                        if XARM == "dump":
                            rel = rel + m_c * rv      # instant
                        else:                         # graded|elastic
                            rel = rel + XR_GRADED * m_c * rv
                    rel = rel.minimum(m_c)            # never below zero mass
                    m_c = m_c - rel
                    x_rel = x_rel + rel
                dm = (1.0 - m_c) * cl.sigmoid()
                if int(os.environ.get("ALG_CLOCK", "0")) and tail is not None:
                    # CLOCK v1 (door #10): commit gated on OWN-SENTENCE
                    # COMPLETION — attention mass on sentence-tail tokens
                    c_j = (fat_cur * tail.reshape(B, 1, -1)).sum(-1, keepdim=True)                         / (fat_cur.sum(-1, keepdim=True) + 1e-6)
                    _fl = float(os.environ.get("ALG_CLOCK_FLOOR", "0"))
                    dm = dm * (_fl + (1.0 - _fl) * c_j)
                anchor = (m_c * anchor + dm * cur_new) / (m_c + dm + 1e-6)
                m_c = m_c + dm
                cur = m_c * anchor + (1.0 - m_c) * cur_new
            else:
                cur = cur_new
            if ALG_NOTEBOOK:
                _nb.append((cur @ p["W_sil"]) if NB_PERSLOT
                           else (cur.mean(1) @ p["W_sil"]))
            breaths.append(cur)

    def heads_of(s):
        return {
            "pres": (s @ p["h_pres"] + p["h_pres_b"]).squeeze(-1),
            "ftype": s @ p["h_ftype"] + p["h_ftype_b"],
            "op": s @ p["h_op"] + p["h_op_b"],
            **({"sel": s @ p["h_sel"] + p["h_sel_b"]} if "h_sel" in p else {}),
            **({"dup": (s @ p["h_dup"] + p["h_dup_b"]).squeeze(-1)}
               if "h_dup" in p else {}),
            "islit": (s @ p["h_islit"] + p["h_islit_b"]).squeeze(-1),
            "dig": (s @ p["h_dig"] + p["h_dig_b"]).reshape(B, L_FAC, N_DIG, 10),
            **({"sgn": (s @ p["h_sgn"] + p["h_sgn_b"]).squeeze(-1)}
               if "h_sgn" in p else {}),
            "args": (s @ p["W_args"]) @ vst.transpose(-2, -1),
            **({"dargs": (s @ p["W_dargs"]) @ vst.transpose(-2, -1)}
               if "W_dargs" in p else {}),
            **({"iargs": (s @ p["W_iargs"]) @ vst.transpose(-2, -1)}
               if "W_iargs" in p else {}),
            "res": (s @ p["W_res"]) @ vst.transpose(-2, -1),
            **({"dig2": (s @ p["h_dig2"] + p["h_dig2_b"])
                .reshape(B, L_FAC, N_DIG, 10),
                "y": (s @ p["W_y"]) @ vst.transpose(-2, -1)}
               if "h_dig2" in p else {}),
        }
    if int(os.environ.get("ALG_MINE_BREATHS", "0")):
        out_breaths = breaths          # v3: the dialect ladder's raw states
    _s_final = breaths[-1]
    if anchor is not None and amask is not None:     # FORM (A): state-side
        _s_final = amask * anchor + (1.0 - amask) * _s_final   # anchors —
                                                     # structural re-entry the
                                                     # forward cannot ignore
    out = heads_of(_s_final)
    if int(os.environ.get("ALG_MINE_BREATHS", "0")) and K_B > 1 and slot_mask is not None:
        out["breaths_all"] = out_breaths
    if (int(os.environ.get("ALG_DEEPSUP", "0")) or ALG_CONSUME) and K_B > 1 and len(breaths) > 1:
        out["_early"] = [heads_of(b) for b in breaths[:-1]]   # V2: the whole
                                                  # supply chain gets gradient
    if int(os.environ.get("ALG_INV", "0")):
        out["fst_s"] = breaths[-1]        # ORGAN: invariance fire — the pair
                                          # agreement term reads the waist here
    if RINGS and K_B > 1 and slot_mask is not None and "W_bo" in p:
        out["cmt"] = cmt_logits[0].stack(*cmt_logits[1:], dim=1) if len(cmt_logits) > 1 \
            else cmt_logits[0].unsqueeze(1)               # (B, K_B-1, L_FAC)
        out["cmt_m"] = m_c.squeeze(-1)                  # final mass (B, L_FAC)
        if XOUT:
            out["xrel"] = x_rel.squeeze(-1)             # released-mass ledger
                                                        # (revisability meter's
                                                        # raw feed; audit-side)
    out["query"] = ((qst @ p["W_query"]) @ vst.transpose(-2, -1)).reshape(B, K_VARS)
    if "h_depth" in p:
        out["depth"] = fst @ p["h_depth"] + p["h_depth_b"]
        out["term"] = (fst @ p["h_term"] + p["h_term_b"]).squeeze(-1)
    if "W5_a1_1" in p or "W6_a1" in p:
        # v5/v6 binding by construction: heads answer WHICH (logits over the
        # codebook, softmax -> code mixture); frozen phasors answer WHERE
        # (+theta rotation); the wire answers TRANSMIT (sum of four).
        # v6: logits from a SHARED gelu trunk + per-role projections.
        global _BINDF
        try: _BINDF
        except NameError: _BINDF = None
        if _BINDF is None:
            import numpy as _np5
            _bz5 = _np5.load(os.environ.get("BIND_CODES", ".cache/bindbus_codes.npz"))
            _CB5 = Tensor(_bz5["CB"].astype(_np5.float32))
            _rot5 = {}
            for _ri5, _rn5 in enumerate(("arg1", "arg2", "res", "op")):
                _th5 = _bz5[f"theta_{_rn5}"]
                _rot5[_ri5] = (Tensor(_np5.cos(_th5).astype(_np5.float32)),
                               Tensor(_np5.sin(_th5).astype(_np5.float32)))
            _BINDF = (_CB5, _rot5)
        _CB5, _rot5 = _BINDF
        _P5 = _CB5.shape[1] // 2                # P_planes (D_real // 2)
        _wire = None; _lgs = []
        _h6 = ((fst @ p["W_bind1"] + p["W_bind1_b"]).gelu()
               if "W6_a1" in p else None)
        for _ri5, _r5 in enumerate(("a1", "a2", "rs", "op")):
            _lg5 = (_h6 @ p[f"W6_{_r5}"] if _h6 is not None else
                    (fst @ p[f"W5_{_r5}_1"] + p[f"W5_{_r5}_b"]).gelu() @ p[f"W5_{_r5}_2"])
            _lgs.append(_lg5)
            _v5 = _lg5.softmax(-1) @ _CB5       # soft pointer: codebook hull
            _vr5 = _v5.reshape(*_v5.shape[:-1], _P5, 2)
            _c5, _s5 = _rot5[_ri5]
            _bx = _vr5[..., 0] * _c5 - _vr5[..., 1] * _s5
            _by = _vr5[..., 0] * _s5 + _vr5[..., 1] * _c5
            _b5 = Tensor.stack(_bx, _by, dim=-1).reshape(*_v5.shape)
            _wire = _b5 if _wire is None else _wire + _b5
        out["bind"] = _wire
        out["bind_lg"] = Tensor.stack(*_lgs, dim=-2)   # (B, L_FAC, 4, 32)
    elif "W_bind2" in p:
        out["bind"] = (fst @ p["W_bind1"] + p["W_bind1_b"]).gelu() @ p["W_bind2"]
    elif "W_bind" in p:
        out["bind"] = fst @ p["W_bind"]
    if "W_opc1" in p:
        if int(os.environ.get("ALG_OPCOUNT", "0")) == 2:
            # rescue variant (registered pre-fire; mechanism-cleared): pool
            # over the FIXED 24 slots — constant denominator keeps the
            # readout EXTENSIVE (counts survive; token-mean normalized
            # them away — the intensive-readout exclusion)
            _pool = fst.mean(1)
        else:
            _pool = ((waist * tokmask.unsqueeze(-1)).sum(1)
                     / (tokmask.sum(1, keepdim=True) + 1e-6))
        out["opc"] = ((_pool @ p["W_opc1"] + p["W_opc1_b"]).gelu()
                      @ p["W_opc2"] + p["W_opc2_b"]).reshape(
                          B, len(OPC_CLASSES), OPC_CAP + 1)
    out["fat"], out["vat"] = fat, vat
    if "h_ref" in p:
        out["ref"] = waist @ p["h_ref"] + p["h_ref_b"]   # (B, T_ALG, K_VARS)
    if len(breaths) > 1:
        out["breaths"] = [heads_of(s) for s in breaths]
    return out


def loss_fn(o, g):
    """Ladder wrapper: with breaths, per-breath weighted CE (1 + k/(K-1)) —
    the v98 ladder; the shared query/span terms ride the final breath only."""
    if "breaths" in o:
        K_B = len(o["breaths"])
        tot = None
        for kb, ob in enumerate(o["breaths"]):
            full = dict(o, **ob)
            w = 1.0 + kb / max(K_B - 1, 1)
            term = _loss_single(full, g) * w
            tot = term if tot is None else tot + term
        if int(os.environ.get("BREATH_NORM", "0")):
            _wsum = sum(1.0 + kb / max(K_B - 1, 1) for kb in range(K_B))
            return tot / _wsum          # door #50: TRUE-scale ladder — the
                                        # 1.5x heat confound removed at birth
        return tot / K_B
    return _loss_single(o, g)


def _loss_single(o, g):
    from tinygrad import Tensor
    pres = g["presence"]
    n_p = pres.sum() + 1e-6
    # explicit per-kind masks (tranche); legacy callers without them fall back
    # to the old two-kind arithmetic (byte-identical behavior)
    is_rel = g["is_rel"] if "is_rel" in g else (1.0 - g["is_lit_f"])
    is_mod = g["is_mod"] if "is_mod" in g else g["is_lit_f"] * 0.0
    is_sel = g["is_sel"] if "is_sel" in g else g["is_lit_f"] * 0.0
    is_pct = g["is_pct"] if "is_pct" in g else g["is_lit_f"] * 0.0
    is_fdiv = g["is_fdiv"] if "is_fdiv" in g else g["is_lit_f"] * 0.0
    rel = pres * is_rel
    n_rel = rel.sum() + 1e-6

    def bce(lg, tg):
        return lg.maximum(0) - lg * tg + (1 + (-lg.abs()).exp()).log()

    def ce(lg, tg):
        return (lg.log_softmax(-1) * -1).gather(-1, tg.unsqueeze(-1)).squeeze(-1)

    l = bce(o["pres"], pres).mean()
    l = l + (ce(o["ftype"], g["ftype"]) * pres).sum() / n_p
    l = l + (ce(o["op"], g["op"]) * rel).sum() / n_rel
    l = l + bce(o["islit"], g["is_lit_f"]).mean()
    if "depth" in o and "depth" in g:      # the position channel (gold-fed)
        l = l + (ce(o["depth"], g["depth"]) * pres).sum() / n_p
        l = l + (bce(o["term"], g["term"]) * pres).sum() / n_p
    if "opc" in o and "opc" in g:          # lever 3: row-grain count CE
        l = l + ce(o["opc"], g["opc"]).mean()
    if "bind" in o and "bindvec" in g and int(os.environ.get("ALG_BINDBUS", "0")) < 5:
        # the rotational binding bus (v5 excluded: pure CE, no wire loss)
        _e = o["bind"]; _t = g["bindvec"]
        _cos = (_e * _t).sum(-1) / ((_e.pow(2).sum(-1).sqrt() + 1e-6)
                                    * (_t.pow(2).sum(-1).sqrt() + 1e-6))
        _w = 1.0 if int(os.environ.get("ALG_BINDBUS", "0")) < 3 else 0.2
        l = l + _w * ((1.0 - _cos) * pres).sum() / n_p
    if "bind_lg" in o and "bind_ids" in g:
        # v5 native loss: pure per-role classification CE — zero cross-role
        # gradient (no wire-level term; parameters disjoint by construction)
        l = l + 0.5 * (ce(o["bind_lg"], g["bind_ids"])
                       * pres.unsqueeze(-1)).sum() / n_p
    elif "bind" in o and "bind_ids" in g and int(os.environ.get("ALG_BINDBUS", "0")) >= 3:
        # v3 THE ROLE-FACTORED LOSS: supervise each role's unbound cleanup
        # directly — conjugate-rotate the emission, CE against the codebook
        global _BINDC
        try: _BINDC
        except NameError: _BINDC = None
        if _BINDC is None:
            import numpy as _np
            from tinygrad import Tensor as _T3
            _bz = _np.load(os.environ.get("BIND_CODES", ".cache/bindbus_codes.npz"))
            _CBt = _T3(_bz["CB"].astype(_np.float32))
            _rc = {}
            for _ri, _r in enumerate(("arg1", "arg2", "res", "op")):
                _th = _bz[f"theta_{_r}"]
                _rc[_ri] = (_T3(_np.cos(-_th).astype(_np.float32)),
                            _T3(_np.sin(-_th).astype(_np.float32)))
            _BINDC = (_CBt, _rc)
        _CBt, _rc = _BINDC
        _P2 = _e.shape[-1] // 2
        _er = _e.reshape(*_e.shape[:-1], _P2, 2)
        _ex, _ey = _er[..., 0], _er[..., 1]
        for _ri in range(4):
            _c3, _s3 = _rc[_ri]
            _ux = _ex * _c3 - _ey * _s3
            _uy = _ex * _s3 + _ey * _c3
            _u = Tensor.stack(_ux, _uy, dim=-1).reshape(*_e.shape)
            _lg = _u @ _CBt.T                       # (B, L_FAC, 32)
            _y = g["bind_ids"][..., _ri]
            l = l + 0.5 * (ce(_lg, _y) * pres).sum() / n_p
    is_macro = g["is_macro"] if "is_macro" in g else is_mod * 0.0
    is_frac = g["is_frac"] if "is_frac" in g else is_mod * 0.0
    dm = g["is_lit_f"] + is_mod + is_pct + is_fdiv + is_macro + is_frac
    l = l + (ce(o["dig"], g["digits"]).mean(-1) * dm).sum() / (dm.sum() + 1e-6) \
        * float(os.environ.get("OBJW_DIG", "1.0"))       # pool axis OBJW
    if "sgn" in o and "sign" in g:              # E1: sign BCE on value slots
        l = l + (bce(o["sgn"], g["sign"]) * dm).sum() / (dm.sum() + 1e-6)
    if "cmt" in o:      # RUNG-3: commit-when-correct (self-labeled, DETACHED
                        # targets from gold-match — settle appears nowhere)
        ok_ft = (o["ftype"].argmax(-1) == g["ftype"]).float()
        ok_res = (o["res"].argmax(-1) == g["res"]).float()
        correct = (ok_ft * ok_res * pres).detach()      # (B, L_FAC)
        ck = o["cmt"]                                    # (B, KB-1, L_FAC)
        l = l + (bce(ck, correct.unsqueeze(1).expand(*ck.shape)) *
                 pres.unsqueeze(1)).sum() / (pres.sum() * ck.shape[1] + 1e-6)
    if "sel" in o and "sel" in g:               # selector-type CE (closed vocab)
        sm = pres * is_sel
        l = l + (ce(o["sel"], g["sel"]) * sm).sum() / (sm.sum() + 1e-6)
    if "dup" in o and "arg_dup" in g:           # gen-9: arg-multiplicity BCE
        l = l + (bce(o["dup"], g["arg_dup"]) * rel).sum() / n_rel
    if "dargs" in o and "arg_dup" in g:         # door #12: the dedicated dup
        dm2 = rel * g["arg_dup"]                # pointer — single-target gold
        l = l + (ce(o["dargs"], g["args"].argmax(-1)) * dm2).sum() / (dm2.sum() + 1e-6)
    if "valspan" in g:                          # door #61: given-binding aid —
        _vs = g["valspan"]                       # the pointer law's structural
        _vm = (_vs.sum(-1) > 0).float()          # entry at the value grain
        _vsn = _vs / (_vs.sum(-1, keepdim=True) + 1e-6)
        l = l + float(os.environ.get("VALATT_W", "1.0")) * (
            (-(o["fat"] + 1e-9).log() * _vsn).sum(-1) * _vm).sum() / (_vm.sum() + 1e-6)
    if "iargs" in o and "is_ind" in g:          # door #45: dialect reader —
        im = is_rel * g["is_ind"] * pres        # indirect-population gold only
        l = l + float(os.environ.get("DIAL_W", "1.0")) * (
            (bce(o["iargs"], g["args"]).mean(-1) * im).sum() / (im.sum() + 1e-6))
    if "dig2" in o and "is_macro" in g:        # gen-15: OP_APPLY terms
        if "is_frac" in g:                     # mg2: frac k rides the dig2 CE
            mac2 = pres * (g["is_macro"] + g["is_frac"])
            l = l + (ce(o["dig2"], g["digits2"]) .mean(-1) * mac2).sum() / (mac2.sum() + 1e-6)
        mac = pres * g["is_macro"]
        n_mac = mac.sum() + 1e-6
        l = l + (ce(o["op"], g["op"]) * mac).sum() / n_mac
        l = l + (ce(o["dig2"], g["digits2"]).mean(-1) * mac).sum() / n_mac
        l = l + (ce(o["y"], g["y"]) * mac).sum() / n_mac * 2.0
    args_w = 1.0 + 4.0 * g["args"]
    is_chain = g["is_chain"] if "is_chain" in g else is_mod * 0.0
    am = pres * (is_rel + is_sel + is_mod + is_pct + is_fdiv + is_macro + is_frac + is_chain)
    n_am = am.sum() + 1e-6
    _ow_ptr = float(os.environ.get("OBJW_PTR", "1.0"))   # pool axis OBJW
    l = l + ((bce(o["args"], g["args"]) * args_w).mean(-1) * am).sum() / n_am * 2.0 * _ow_ptr
    l = l + (ce(o["res"], g["res"]) * pres).sum() / n_p * 2.0 * _ow_ptr
    l = l + ce(o["query"], g["query"]).mean() * 2.0 * _ow_ptr
    fsn = g["fspan"] / (g["fspan"].sum(-1, keepdim=True) + 1e-6)
    # FAT_W (routing-canvas dose probe, gut #55 amended): the fat-CE canvas
    # has hung here at weight 1 all along; the probe doses it, never adds it.
    fat_w = float(os.environ.get("FAT_W", "1"))
    l = l + fat_w * ((-(o["fat"] + 1e-9).log() * fsn).sum(-1) * pres).sum() / n_p
    vsn = g["vspan"] / (g["vspan"].sum(-1, keepdim=True) + 1e-6)
    vmask = (g["vspan"].sum(-1) > 0).float()
    l = l + ((-(o["vat"] + 1e-9).log() * vsn).sum(-1) * vmask).sum() / (vmask.sum() + 1e-6)
    if "ref" in o and "refoh" in g:
        _rm = g["refoh"].sum(-1)                          # (B, T_ALG) site mask
        _rce = (-(o["ref"].log_softmax(-1)) * g["refoh"]).sum(-1)
        l = l + float(os.environ.get("REF_W", "0.1")) * (_rce * _rm).sum() / (_rm.sum() + 1e-6)
    return l


# ===========================================================================
# DECODE + PER-BAND EVAL (factor-exact / graph-solve / ANSWER, per band)
# ===========================================================================

def decode(o_np):
    from mycelium.csp_domains import ID_TO_SEL
    facs = []
    for j in range(L_FAC):
        if o_np["pres"][j] <= 0:
            continue
        res = int(o_np["res"][j].argmax())
        ft = int(o_np["ftype"][j].argmax()) if o_np["ftype"].shape[-1] >= 4 \
            else (1 if o_np["islit"][j] > 0 else 0)

        def digval():
            digs = o_np["dig"][j].argmax(-1)
            return int(sum(d * 10 ** (N_DIG - 1 - i)
                           for i, d in enumerate(digs)))
        if ft == 1:
            v = digval()
            if "sgn" in o_np and o_np["sgn"][j] > 0:   # E1: negative literal
                v = -v
            facs.append({"ftype": "given", "var": res, "value": v})
        elif ft == 0:
            op = "add" if o_np["op"][j].argmax() == 0 else "mul"
            if "dup" in o_np and o_np["dup"][j] > 0:
                a0 = int(np.argmax(o_np["dargs"][j])) if "dargs" in o_np \
                    else int(np.argmax(o_np["args"][j]))   # door #12 / gen-9
                args = [a0, a0]
            else:
                args = sorted(int(a) for a in
                              np.argsort(-o_np["args"][j])[:2])
            facs.append({"ftype": "rel", "op": op,
                         "args": args, "result": res})
        elif ft == 2:
            var = int(np.argmax(o_np["args"][j]))
            facs.append({"ftype": "mod", "var": var, "k": max(digval(), 2),
                         "result": res})
        elif ft == 4:
            facs.append({"ftype": "pct",
                         "args": [int(np.argmax(o_np["args"][j])), res],
                         "p": max(digval(), 1)})
        elif ft == 5:
            facs.append({"ftype": "fdiv",
                         "var": int(np.argmax(o_np["args"][j])),
                         "k": max(digval(), 2), "result": res})
        elif ft == 8:
            xs = sorted(int(v) for v in np.where(o_np["args"][j] > 0)[0].tolist())
            if len(xs) >= 3:
                facs.append({"ftype": "macro", "name": "CHAIN_MUL",
                             "xs": xs, "result": res})
            continue
        elif ft == 7 and "dig2" in o_np:
            k_ = int(sum(d * 10 ** (N_DIG - 1 - i2) for i2, d in
                         enumerate(o_np["dig2"][j].argmax(-1))))
            facs.append({"ftype": "macro", "name": "FRAC_OF",
                         "a": max(digval(), 1), "k": max(k_, 2),
                         "x": int(np.argmax(o_np["args"][j])), "result": res})
        elif ft == 6 and "dig2" in o_np:
            k2 = int(sum(d * 10 ** (N_DIG - 1 - i2) for i2, d in
                         enumerate(o_np["dig2"][j].argmax(-1))))
            facs.append({"ftype": "macro", "name": "OP_APPLY",
                         "op": "add" if o_np["op"][j].argmax() == 0 else "sub",
                         "k1": max(digval(), 1), "x": int(np.argmax(o_np["args"][j])),
                         "k2": max(k2, 1), "y": int(np.argmax(o_np["y"][j])),
                         "result": res})
        else:
            args = list(np.argsort(-o_np["args"][j])[:2])
            sel = ID_TO_SEL[int(o_np["sel"][j].argmax())]
            facs.append({"ftype": "sel", "sel": sel,
                         "args": sorted(int(a) for a in args), "result": res})
    return facs, int(o_np["query"].argmax())


def do_eval():
    from tinygrad import Tensor, dtypes
    from tinygrad.nn.state import safe_load
    from mycelium.csp_domains import problem_from_algebra
    from mycelium.csp_core import solve_symbolic

    samples, states, tokmask, gold, sent = load_alg("test")
    p = build_params(0)
    sd = safe_load(ALG_CKPT)
    for k in p:
        p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
    n = len(samples)
    if int(os.environ.get("ALG_TRUNK_LORA", "0")):
        # audit 2026-08-22 #3: banked states are PURE-trunk; a LoRA ckpt
        # must be ringed through its adapters or the number is a no-op
        from beacon_closing_arm import _trunk_host
        from mycelium.llama_loader import _rms_norm as _lrms
        _host = _trunk_host()
        _sc = float(os.environ.get("ALG_LORA_SCALE", "8.0"))
        _ro = int(os.environ.get("ALG_ROPE_OFF", "0"))
        _LD = [{f"{_nm}_{_ab}": (p[f"lora{_li}_{_nm}_{_ab}"] * (_sc if _ab == "B" else 1.0))
                for _nm in ("wq", "wo", "wdown") for _ab in ("A", "B")
                if f"lora{_li}_{_nm}_A" in p} or None
               for _li in range(4)]
        _samples2, _ids2, _msk2, _off2 = tokenize(ALG_TEST)
        assert len(_samples2) == n, "eval tokenize/load_alg desync"
        _st2 = np.zeros((n, T_ALG, H_TRUNK), np.float16)
        for _s0 in range(0, n, 8):
            _sl = slice(_s0, min(_s0 + 8, n))
            _x = _host.llama_embed[Tensor(_ids2[_sl], dtype=dtypes.int)]
            for _li, _layer in enumerate(_host.llama_layers):
                _x = _layer(_x, _host.llama_rope_cos[_ro:] if _ro else _host.llama_rope_cos,
                            _host.llama_rope_sin[_ro:] if _ro else _host.llama_rope_sin,
                            lora=_LD[_li])
            _x = _lrms(_x, _host.llama_layers[-1].ffn_norm,
                       _host.llama_cfg.rms_norm_eps)
            _st2[_sl] = _x.cast(dtypes.float).realize().numpy().astype(np.float16)
        states = _st2
        print(f"[eval] TRUNK_LORA: states recomputed through adapters "
              f"(scale {_sc}) — HONEST LoRA-ring", flush=True)
    per_band = {}
    for s0 in range(0, n, 8):
        sl = np.arange(s0, min(s0 + 8, n))
        pad = 8 - len(sl)
        sl_p = np.concatenate([sl, sl[:1].repeat(pad)]) if pad else sl
        t_tr = Tensor(states[sl_p].astype(np.float32), dtype=dtypes.float)
        t_tk = Tensor(tokmask[sl_p].astype(np.float32), dtype=dtypes.float)
        t_se = Tensor(sent[sl_p].astype(np.int32), dtype=dtypes.int)
        _tl = Tensor(tails_of(sent[sl_p]), dtype=dtypes.float) \
            if int(os.environ.get("ALG_CLOCK", "0")) else None
        out = forward(p, t_tr, t_tk, t_se, tail=_tl)
        if int(os.environ.get("ALG_BREATH", "1")) > 1 and "W_bo" in p \
                and not int(os.environ.get("BREATH_SILENT", "0")):
            o0 = {k: out[k].realize().numpy() for k in ("fat", "args", "res")}
            mk = build_slot_masks(o0, sent[sl_p])
            out = forward(p, t_tr, t_tk, t_se, tail=_tl,
                          slot_mask=Tensor(mk, dtype=dtypes.float))
        keys = ("pres", "ftype", "op", "islit", "dig", "args", "res",
                "query") + (("sel",) if "sel" in out else ()) + (("dup",) if "dup" in out else ()) \
            + (("dargs",) if "dargs" in out else ())
        o = {k: out[k].realize().numpy() for k in keys}
        for bi, i in enumerate(sl):
            i = int(i)
            smp = samples[i]
            band = int(gold["band"][i])
            st = per_band.setdefault(band, {"n": 0, "fac_ok": 0, "fac_tot": 0,
                                            "solve": 0, "answer": 0, "query_ok": 0})
            st["n"] += 1
            facs, q_pred = decode({k: o[k][bi] for k in o})

            def fkey(f):
                if f["ftype"] == "rel":
                    return ("rel", f["op"], tuple(sorted(f["args"])),
                            f["result"])
                if f["ftype"] == "given":
                    return ("given", f["var"], f["value"])
                if f["ftype"] == "mod":
                    return ("mod", f["var"], f["k"], f["result"])
                if f["ftype"] == "pct":
                    return ("pct", tuple(f["args"]), f["p"])
                if f["ftype"] == "fdiv":
                    return ("fdiv", f["var"], f["k"], f["result"])
                return ("sel", f["sel"], tuple(sorted(f["args"])), f["result"])
            gset = set(fkey(f) for f in smp["factors"])
            st["fac_ok"] += len(gset & set(fkey(f) for f in facs))
            st["fac_tot"] += len(gset)
            st["query_ok"] += int(q_pred == smp["query_var"])
            gv = {f["var"]: f["value"] for f in facs if f["ftype"] == "given"}

            def fvars(f):
                if f["ftype"] in ("rel", "sel", "pct"):
                    return list(f["args"]) + ([f["result"]]
                                              if "result" in f else [])
                if f["ftype"] in ("mod", "fdiv"):
                    return [f["var"], f["result"]]
                return [f["var"]]
            try:
                from mycelium.csp_domains import problem_from_algebra3
                nv = max([smp["n_vars"]] + [v + 1 for f in facs
                                            for v in fvars(f)])
                res = solve_symbolic(problem_from_algebra3(nv, facs, gv,
                                                           smp["m"]),
                                     budget=200_000, seed=0)
                if res["status"] == "solved":
                    sol = [int(res["assignment"][v]) for v in range(nv)]
                    if sol[:smp["n_vars"]] == smp["solution"]:
                        st["solve"] += 1
                    gold_ans = smp["solution"][smp["query_var"]]
                    if q_pred < len(sol) and sol[q_pred] == gold_ans:
                        st["answer"] += 1
            except Exception:
                pass

    print(f"\n[algebra eval] per-BAND (the registered stratification):")
    print(f"  band |  n | fac-F1ish | query | graph-solve | ANSWER")
    tot = {"n": 0, "solve": 0, "answer": 0}
    for b in sorted(per_band):
        st = per_band[b]
        print(f"    {b:2d} | {st['n']:2d} |   {st['fac_ok']/max(st['fac_tot'],1):.3f}   "
              f"|  {st['query_ok']/max(st['n'],1):.2f} |     {st['solve']:2d}      |   {st['answer']:2d}")
        for k in tot:
            tot[k] += st[k] if k != "n" else st["n"]
    print(f"  TOTAL: {tot['solve']}/{tot['n']} graph-solve, "
          f"{tot['answer']}/{tot['n']} ANSWER")
    print(f"  (factorization read: flat fac-exact across bands = reading and"
          f" reasoning on independent axes)")




# ===========================================================================
# TAXONOMY (--errors): prediction #2-algebra's grader (spec §11)
# ===========================================================================
# Classes: CORRECT | DETECT_malformed | DETECT_unsat | DETECT_multi (ban-and-resolve
# uniqueness probe, gold-free) | SILENT (unique-solves to the WRONG answer). Literal
# errors attributed by the gold given's ROLE: chain_k vs pair_sum/pair_diff — the
# registered ordering: chain-literals >> coupled-literals > structural (~0).

def do_errors():
    from tinygrad import Tensor, dtypes
    from tinygrad.nn.state import safe_load
    from mycelium.csp_domains import problem_from_algebra
    from mycelium.csp_core import solve_symbolic

    samples, states, tokmask, gold, sent = load_alg("test")
    p = build_params(0)
    sd = safe_load(ALG_CKPT)
    for k in p:
        p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
    n = len(samples)
    cats = {"CORRECT": 0, "DETECT_malformed": 0, "DETECT_unsat": 0,
            "DETECT_multi": 0, "SILENT": 0}
    lit_err = {}     # class -> {role: count} for wrong-literal cases
    for s0 in range(0, n, 8):
        sl = np.arange(s0, min(s0 + 8, n))
        pad = 8 - len(sl)
        sl_p = np.concatenate([sl, sl[:1].repeat(pad)]) if pad else sl
        out = forward(p, Tensor(states[sl_p].astype(np.float32), dtype=dtypes.float),
                      Tensor(tokmask[sl_p].astype(np.float32), dtype=dtypes.float),
                      Tensor(sent[sl_p].astype(np.int32), dtype=dtypes.int),
                      lsent=(Tensor(gold["lsent"][sl_p].astype(np.float32),
                                    dtype=dtypes.float) if ALG_LSENT and "lsent" in gold else None))
        keys = ("pres", "ftype", "op", "islit", "dig", "args", "res",
                "query") + (("sel",) if "sel" in out else ()) + (("dup",) if "dup" in out else ()) \
            + (("dargs",) if "dargs" in out else ())
        o = {k: out[k].realize().numpy() for k in keys}
        for bi, i in enumerate(sl):
            i = int(i)
            smp = samples[i]
            facs, q_pred = decode({k: o[k][bi] for k in o})
            # wrong-literal attribution (pred given value != gold given value, by var)
            gold_gv = {f["var"]: (f["value"], f.get("role", "?"))
                       for f in smp["factors"] if f["ftype"] == "given"}
            wrong_lits = [(v, gold_gv[v][1]) for f in facs if f["ftype"] == "given"
                          for v in [f["var"]] if v in gold_gv
                          and f["value"] != gold_gv[v][0]]
            rels = [(f["op"], f["args"][0], f["args"][1], f["result"])
                    for f in facs if f["ftype"] == "rel"]
            gv = {f["var"]: f["value"] for f in facs if f["ftype"] == "given"}
            gold_ans = smp["solution"][smp["query_var"]]
            cat = None
            try:
                nv = max([smp["n_vars"]] + [v + 1 for f in facs for v in
                         ((list(f["args"]) + [f["result"]]) if f["ftype"] == "rel"
                          else [f["var"]])])
                if nv > 26:
                    cat = "DETECT_malformed"
                else:
                    res = solve_symbolic(problem_from_algebra(nv, rels, gv, smp["m"]),
                                         budget=200_000, seed=0)
                    if res["status"] != "solved":
                        cat = "DETECT_unsat"
                    else:
                        sol = [int(res["assignment"][v]) for v in range(nv)]
                        ans_ok = q_pred < len(sol) and sol[q_pred] == gold_ans
                        if ans_ok and not wrong_lits:
                            cat = "CORRECT"
                        else:
                            # uniqueness probe on the PREDICTED graph (gold-free)
                            multi = False
                            for v in range(nv):
                                if v in gv:
                                    continue
                                p2 = problem_from_algebra(nv, rels, gv, smp["m"])
                                p2.domains0[v].discard(sol[v])
                                if p2.domains0[v]:
                                    r2 = solve_symbolic(p2, budget=100_000, seed=0)
                                    if r2["status"] != "unsat":  # deep clean 2026-07-30: budget is not a uniqueness certificate
                                        multi = True
                                        break
                            if ans_ok:
                                cat = "CORRECT"   # right answer, benign literal drift
                            elif multi:
                                cat = "DETECT_multi"
                            else:
                                cat = "SILENT"
            except Exception:
                cat = "DETECT_malformed"
            cats[cat] += 1
            if wrong_lits and cat in ("SILENT", "DETECT_unsat", "DETECT_multi"):
                for _v, role in wrong_lits:
                    lit_err.setdefault(cat, {}).setdefault(role, 0)
                    lit_err[cat][role] += 1

    wrong = n - cats["CORRECT"]
    det = cats["DETECT_malformed"] + cats["DETECT_unsat"] + cats["DETECT_multi"]
    print(f"\n[algebra errors] n={n}")
    for k, v in cats.items():
        print(f"  {k:18s} {v}")
    if wrong:
        print(f"  DETECTABLE fraction: {det}/{wrong} = {det/wrong:.2f} "
              f"(KenKen was 1.00 seven times — the INVERSION is the prediction)")
    print(f"  wrong-literal attribution by role x class: {lit_err}")
    print(f"  (registered ordering: chain_k literals >> pair literals (parity ~half-"
          f"caught) > structural ~0)")


# ===========================================================================
# TRAIN
# ===========================================================================

def do_train(steps, lr, batch, seed):
    from tinygrad import Tensor, dtypes
    from tinygrad.engine.jit import TinyJit
    from tinygrad.nn.optim import AdamW
    from tinygrad.nn.state import safe_save

    samples, states, tokmask, gold, sent = load_alg("train")
    n = states.shape[0]
    p = build_params(seed)
    TRUNK_LORA = int(os.environ.get("ALG_TRUNK_LORA", "0"))
    if TRUNK_LORA:
        from beacon_closing_arm import _trunk_host
        HOST = _trunk_host()
        _npz_p = STATES_NPY.format(split=os.environ.get("ALG_TRAIN_NAME", "train")).replace("_states.npy", ".npz")
        try:
            _z = np.load(_npz_p)
            IDS_ALL = _z["ids"]     # perf audit 2026-08-23 #4: assembly already
            print(f"[lora] ids from sidecar {_npz_p}", flush=True)   # tokenized
        except Exception:
            _, IDS_ALL, _, _ = tokenize(os.environ["ALG_TRAIN"])
        LORA_SCALE = float(os.environ.get("ALG_LORA_SCALE", "8.0"))
        ROPE_OFF = int(os.environ.get("ALG_ROPE_OFF", "0"))
        print(f"[lora] trunk-in-the-loop: r={os.environ.get('ALG_LORA_R','16')} "
              f"scale={LORA_SCALE} rows={len(IDS_ALL)}", flush=True)
    if int(os.environ.get("RESUME", "0")) and not os.path.exists(ALG_CKPT):
        # RESUME GUARD (deep clean 2026-07-30): a missing ckpt under RESUME=1
        # previously fell through to FRESH RANDOM INIT silently — a
        # silently-wrong segment is far costlier than a loud stop.
        raise RuntimeError(f"RESUME=1 but {ALG_CKPT} does not exist — "
                           f"refusing to cold-start silently (resume guard)")
    if int(os.environ.get("RESUME", "0")) and os.path.exists(ALG_CKPT):
        from tinygrad.nn.state import safe_load as _sl
        import shutil
        shutil.copy2(ALG_CKPT, ALG_CKPT + ".pre_resume")   # house rule
        # (2026-08-16): RESUME never clobbers the only copy — the pre-swap
        # checkpoint stays readable after the continuation overwrites in place
        sd0 = _sl(ALG_CKPT)
        assert set(sd0.keys()) == set(p.keys()), "resume key mismatch (hard error)"
        for k in p:
            p[k].assign(sd0[k].to(p[k].device).cast(p[k].dtype)).realize()
        print(f"[train] RESUMED from {ALG_CKPT}", flush=True)
    elif os.environ.get("WARM_FROM"):
        # tranche warm-start: load every shape-matching key from a LEGACY ckpt;
        # skips are EXPLICIT AND PRINTED (warm-start may skip loudly — eval
        # loads must hard-error; this is the training-side allowance)
        from tinygrad.nn.state import safe_load as _sl
        sd0 = _sl(os.environ["WARM_FROM"])
        n_load = 0
        for k in p:
            if k in sd0 and tuple(sd0[k].shape) == tuple(p[k].shape):
                p[k].assign(sd0[k].to(p[k].device).cast(p[k].dtype)).realize()
                n_load += 1
            elif k in sd0 and len(sd0[k].shape) == len(p[k].shape) and all(
                    o <= n_ for o, n_ in zip(sd0[k].shape, p[k].shape)):
                # PAD-WARM (2026-07-10, the ftype-router lesson): old shape is
                # a prefix of new — copy the trained slice, keep fresh init on
                # the new rows. Discarding a trained ROUTER forces relearning
                # inside a converged circuit (the bootstrap-trap family).
                import numpy as _np
                cur = p[k].detach().numpy()
                old = sd0[k].to(p[k].device).cast(p[k].dtype).numpy()
                sl = tuple(slice(0, o) for o in old.shape)
                cur[sl] = old
                p[k].assign(Tensor(cur, dtype=p[k].dtype)).realize()
                n_load += 1
                print(f"[warm] PAD-WARM {k} {tuple(old.shape)} -> "
                      f"{tuple(cur.shape)}", flush=True)
            else:
                print(f"[warm] SKIP {k} (fresh init: "
                      f"{'missing' if k not in sd0 else 'shape'})", flush=True)
        print(f"[train] WARM from {os.environ['WARM_FROM']}: "
              f"{n_load}/{len(p)} keys", flush=True)
        if "W_bo" in p and int(os.environ.get("BREATH_WARM_BO", "0")) and "attn_wo" in sd0:
            p["W_bo"].assign(sd0["attn_wo"].to(p["W_bo"].device)
                             .cast(p["W_bo"].dtype)).realize()   # door #50:
                             # the pipe warm-seeded from the trained router
                             # (never discard a trained router — the W_bo
                             # zero-saddle's named key)
        if "W_iargs" in p and "W_iargs" not in sd0 and "W_args" in sd0:
            p["W_iargs"].assign(sd0["W_args"].to(p["W_iargs"].device)
                                .cast(p["W_iargs"].dtype)).realize()
        if "W_dargs" in p and "W_dargs" not in sd0 and "W_args" in sd0:
            p["W_dargs"].assign(sd0["W_args"].to(p["W_dargs"].device)
                                .cast(p["W_dargs"].dtype)).realize()
            print("[warm] W_dargs seeded from trained W_args (door #12)", flush=True)
    _frz = [k_ for k_ in (("h_dup", "h_dup_b") if int(os.environ.get("ALG_FREEZE_DUP", "0")) else ())]
    if os.environ.get("ALG_TRAIN_ONLY"):
        # the frozen-parse wake (2026-08-25): train ONLY the named params;
        # everything else frozen — drift eliminated by construction, new
        # organs learn around a byte-identical parse (clean attribution)
        _only = set(os.environ["ALG_TRAIN_ONLY"].split(","))
        _missing = _only - set(p.keys())
        assert not _missing, f"ALG_TRAIN_ONLY names absent params: {_missing}"
        _frz = [k_ for k_ in p if k_ not in _only]
    if _frz: print(f"[freeze] excluded from optimizer: {len(_frz)} params (freeze-or-refold law)", flush=True)
    opt = AdamW([v_ for k_, v_ in p.items() if k_ not in _frz], lr=lr, weight_decay=0.01)
    rng = np.random.RandomState(seed)

    K_B = int(os.environ.get("ALG_BREATH", "1"))
    CLOCK = int(os.environ.get("ALG_CLOCK", "0"))
    TAILS = None
    if CLOCK:
        TAILS = np.zeros_like(sent, dtype=np.uint8)
        for _ri in range(sent.shape[0]):
            _s = sent[_ri]; _i = 0
            while _i < len(_s):
                _j = _i
                while _j + 1 < len(_s) and _s[_j + 1] == _s[_i]: _j += 1
                _h = _i + (_j - _i + 1) // 2
                TAILS[_ri, _h:_j + 1] = 1
                _i = _j + 1
        print(f"[clock] tails ready ({TAILS.mean():.2f} of tokens)", flush=True)
    MASKS = None
    if K_B > 1:
        # mask-prep pass: masks from the WARM-STARTED head's own breath-0
        # parses (deployable-from-birth; frozen for training efficiency)
        print("[breath] mask-prep pass ...", flush=True)
        MASKS = np.zeros((n, L_FAC, L_FAC), np.float32)
        for s0 in range(0, n, 8):
            sl = np.arange(s0, min(s0 + 8, n))
            pad = 8 - len(sl)
            sl_p = np.concatenate([sl, sl[:1].repeat(pad)]) if pad else sl
            out0 = forward(p, Tensor(states[sl_p].astype(np.float32), dtype=dtypes.float),
                           Tensor(tokmask[sl_p].astype(np.float32), dtype=dtypes.float),
                           Tensor(sent[sl_p].astype(np.int32), dtype=dtypes.int),
                           lsent=(Tensor(gold["lsent"][sl_p].astype(np.float32),
                                         dtype=dtypes.float)
                                  if ALG_LSENT and "lsent" in gold else None))
            o0 = {k: out0[k].realize().numpy() for k in ("fat", "args", "res")}
            MASKS[sl] = build_slot_masks(o0, sent[sl_p])[:len(sl)]
        print(f"[breath] masks ready (mean degree "
              f"{MASKS.sum(-1).mean():.1f}/{L_FAC})", flush=True)
    MG = None
    if int(os.environ.get("ALG_MASK_GOLD", "0")) and MASKS is not None:
        # M3 (2026-08-26, word given; the nazare constitution): TRAINING
        # masks from GOLD wiring — the key-gated truth, the one fully
        # independent ocean; the head's own decoded wirings NEVER
        # masquerade as mask gold (recirculation's sibling). Mixed with
        # the heuristic mask per row (flat mix — no curriculum) so
        # inference never meets an unseen regime.
        _A = gold["args"]; _R = gold["res"].astype(np.int64)
        _P = gold["presence"]
        _n = len(_P)
        MG = np.zeros((_n, L_FAC, L_FAC), np.uint8)
        _roh = np.zeros((_n, L_FAC, K_VARS), np.float32)
        np.put_along_axis(_roh, _R[:, :, None], 1.0, axis=2)
        _roh *= _P[:, :, None]
        for i in range(_n):
            MG[i] = ((_A[i] @ _roh[i].T) > 0) & (_P[i][:, None] > 0) \
                & (_P[i][None, :] > 0)
            np.fill_diagonal(MG[i], 1)
        print(f"[m3] gold-wiring masks ready (mean degree "
              f"{MG.sum(-1).mean():.1f}/{L_FAC}, mix p="
              f"{os.environ.get('ALG_MASK_GOLD_P', '0.5')})", flush=True)

    def fix(a, dt):
        return Tensor(a, dtype=dt).contiguous().realize()
    b_tr = fix(np.zeros((batch, T_ALG, H_TRUNK), np.float32), dtypes.float)
    b_ids = fix(np.zeros((batch, T_ALG), np.int32), dtypes.int) if TRUNK_LORA else None
    b_tk = fix(np.zeros((batch, T_ALG), np.float32), dtypes.float)
    b_se = fix(np.zeros((batch, T_ALG), np.int32), dtypes.int)
    b_mask = fix(np.zeros((batch, L_FAC, L_FAC), np.float32), dtypes.float) \
        if K_B > 1 else None
    b_tail = fix(np.zeros((batch, T_ALG), np.float32), dtypes.float) if CLOCK else None
    _GOLD_ALIAS = {"is_lit_f": "is_lit", "refoh": "refvar"}
    _GOLD_OPTIONAL = {"opspan", "arg_dup", "sel", "sign", "y", "digits2",
                      "is_macro", "is_frac", "is_chain"}
    for _tn, _t in TERMINALS.items():
        if not _t["when"](): continue
        for _gk in _t["gold"]:
            if _gk in _GOLD_OPTIONAL: continue
            _gk2 = _GOLD_ALIAS.get(_gk, _gk)
            assert _gk2 in gold, \
                (f"terminal {_tn!r} is ACTIVE but gold {_gk2!r} is missing from "
                 f"the loaded states npz — a stale cache would zero-train it "
                 f"silently (the feed-door fence, GENERALIZED after the "
                 f"bindvec starvation). Re-precompute or inject g_{_gk}.")
    b_reg = fix(np.zeros((batch,), np.float32), dtypes.float) \
        if int(os.environ.get("ALG_CMT_REG", "0")) else None
    REG = np.load(os.environ.get("ALG_REG_NPY", ".cache/reg_form8.npy")) \
        if b_reg is not None else None
    if REG is not None:
        assert len(REG) == len(samples), \
            f"reg npy {len(REG)} rows vs {len(samples)} samples (desync)"
    b_drop = fix(np.ones((1,), np.float32), dtypes.float) \
        if os.environ.get("BREATH_DROPOUT") else None   # door #52 coin buffer
    b_ls = fix(np.zeros((batch, K_VARS, T_ALG), np.float32), dtypes.float) \
        if ALG_LSENT else None                          # V2: letter partition
    if ALG_CONSUME:
        DEFINER = np.full((len(samples), K_VARS), -1, np.int32)
        ARGM = gold["args"] > 0.5
        for _i in range(len(samples)):
            for _j in range(L_FAC):
                if gold["presence"][_i, _j] > 0:
                    DEFINER[_i, int(gold["res"][_i, _j])] = _j
        PARENTS = np.zeros((len(samples), L_FAC, L_FAC), np.float32)
        for _i in range(len(samples)):
            for _j in range(L_FAC):
                if gold["presence"][_i, _j] > 0 and gold["is_rel"][_i, _j] > 0:
                    for _v in np.where(ARGM[_i, _j])[0]:
                        _d = DEFINER[_i, _v]
                        if _d >= 0 and _d != _j:
                            PARENTS[_i, _j, _d] = 1.0
        print(f"[consume] parent DAG built (mean parents "
              f"{PARENTS.sum(-1)[gold['presence']>0].mean():.2f})", flush=True)
    bg = {}
    for k, shape, dt in (("presence", (L_FAC,), dtypes.float),
                         ("is_lit_f", (L_FAC,), dtypes.float),
                         ("args", (L_FAC, K_VARS), dtypes.float),
                         ("fspan", (L_FAC, T_ALG), dtypes.float),
                         ("vspan", (K_VARS, T_ALG), dtypes.float),
                         *((("refoh", (T_ALG, K_VARS), dtypes.float),) if ALG_REF else ()),
                         *((("is_ind", (L_FAC,), dtypes.float),) if ALG_DIAL else ()),
                         ("ftype", (L_FAC,), dtypes.int),
                         ("op", (L_FAC,), dtypes.int),
                         ("res", (L_FAC,), dtypes.int),
                         ("digits", (L_FAC, N_DIG), dtypes.int),
                         ("sel", (L_FAC,), dtypes.int),
                         ("is_rel", (L_FAC,), dtypes.float),
                         ("is_mod", (L_FAC,), dtypes.float),
                         ("is_sel", (L_FAC,), dtypes.float),
                         ("is_pct", (L_FAC,), dtypes.float),
                         ("is_fdiv", (L_FAC,), dtypes.float),
                         ("arg_dup", (L_FAC,), dtypes.float),
                         # gen-15: OP_APPLY gold buffers (two-terminal law —
                         # without these, h_dig2/W_y leave the graph: None grads)
                         *((("is_macro", (L_FAC,), dtypes.float),
                            ("digits2", (L_FAC, N_DIG), dtypes.int),
                            ("y", (L_FAC,), dtypes.int))
                           if int(os.environ.get("ALG_FTYPES", "4")) >= 7 else ()),
                         *((("is_frac", (L_FAC,), dtypes.float),)
                           if int(os.environ.get("ALG_FTYPES", "4")) >= 8 else ()),
                         *((("is_chain", (L_FAC,), dtypes.float),)
                           if int(os.environ.get("ALG_FTYPES", "4")) >= 9 else ()),
                         *((("valspan", (L_FAC, T_ALG), dtypes.float),)
                           if ALG_VALATT else ()),
                         # E1 (2026-08-03): sign gold buffer — two-terminal
                         # law, the same lesson as the gen-15 comment above
                         # (without it h_sgn leaves the graph: None grad)
                         *((("sign", (L_FAC,), dtypes.float),)
                           if ALG_WIDE else ()),
                         *((("depth", (L_FAC,), dtypes.int),
                            ("term", (L_FAC,), dtypes.float))
                           if int(os.environ.get("ALG_POSCH", "0")) else ()),
                         *((("opc", (len(OPC_CLASSES),), dtypes.int),)
                           if int(os.environ.get("ALG_OPCOUNT", "0")) else ()),
                         *((("bindvec", (L_FAC, int(os.environ.get("ALG_BIND_D", "128"))), dtypes.float),)
                           if int(os.environ.get("ALG_BINDBUS", "0")) else ()),
                         *((("bind_ids", (L_FAC, 4), dtypes.int),)
                           if int(os.environ.get("ALG_BINDBUS", "0")) >= 3 else ()),
                         ("query", (), dtypes.int)):
        npdt = np.float32 if dt == dtypes.float else np.int32
        bg[k] = fix(np.zeros((batch,) + shape, npdt), dt)
    if ALG_CONSUME:
        bg["parents"] = fix(np.zeros((batch, L_FAC, L_FAC), np.float32), dtypes.float)
        bg["claimed"] = fix(np.zeros((batch, L_FAC), np.float32), dtypes.float)
        CLAIMED = np.zeros((len(samples), L_FAC), np.float32)  # the persistent
        # ledger: once per fact per RUN — the sharpening subsidy excluded
    if int(os.environ.get("ALG_OPATT", "0")):
        bg["opspan"] = fix(np.zeros((batch, L_FAC, T_ALG), np.float32), dtypes.float)
    assert_terminals(p=p, gold_keys=set(bg.keys()), site="do_train buffers")

    XOUT_TR = int(os.environ.get("ALG_XOUT", "0")) and \
        int(os.environ.get("ALG_RINGS", "0"))
    INV_TR = int(os.environ.get("ALG_INV", "0"))
    INV_PAIRS_ARR = np.load(os.environ["INV_PAIRS"]) if INV_TR else None
    OPATT = int(os.environ.get("ALG_OPATT", "0"))
    OPGOLD = np.load(os.environ["OPATT_GOLD"], mmap_mode="r") if OPATT else None

    _NEWCL = [None]
    @TinyJit
    def step():
        Tensor.training = True
        if TRUNK_LORA:
            from mycelium.llama_loader import _rms_norm as _ll_rms
            _x = HOST.llama_embed[b_ids].detach()
            _rc = HOST.llama_rope_cos[ROPE_OFF:] if ROPE_OFF else HOST.llama_rope_cos
            _rs = HOST.llama_rope_sin[ROPE_OFF:] if ROPE_OFF else HOST.llama_rope_sin
            for _li, _layer in enumerate(HOST.llama_layers):
                _ld = {f"{_nm}_{_ab}": (p[f"lora{_li}_{_nm}_{_ab}"] * (LORA_SCALE if _ab == "B" else 1.0))
                       for _nm in ("wq", "wo", "wdown") for _ab in ("A", "B")
                       if f"lora{_li}_{_nm}_A" in p}
                _x = _layer(_x, _rc, _rs,
                            lora=(_ld if _ld else None))
            _x = _ll_rms(_x, HOST.llama_layers[-1].ffn_norm, HOST.llama_cfg.rms_norm_eps)
            s_tr = _x.cast(dtypes.float)
        else:
            s_tr = b_tr
        if XOUT_TR:
            # ORGAN-2 fire (registered 2026-08-05): two-pass — the first
            # read finds wrong bindings (revoke gold = solver-refuted
            # commits, self-labeled from gold like the commit loss,
            # DETACHED); the second trains under live release dynamics.
            o0 = forward(p, s_tr, b_tk, b_se, slot_mask=b_mask, tail=b_tail,
                         reg=b_reg)
            ok = ((o0["ftype"].argmax(-1) == bg["ftype"]).float()
                  * (o0["res"].argmax(-1) == bg["res"]).float())
            rv = (bg["presence"] * (1.0 - ok)).detach()
            o = forward(p, s_tr, b_tk, b_se, slot_mask=b_mask, revoke=rv,
                        tail=b_tail, reg=b_reg)
        elif int(os.environ.get("NAZ_TRAIN", "0")):
            # NAZARÉ TRAINING (door #55): the organ-2 two-forward pattern —
            # pre-pass yields the intra-pass event field IN-GRAPH (detached);
            # the training pass runs FOCUSED. Training-time criterion
            # (declared per the criterion-key clause, distinct from the
            # read-time dup-aware argpair): argmax-change OR dup-flip on
            # rel-typed present slots, breath-0 vs final.
            o0 = forward(p, s_tr, b_tk, b_se, slot_mask=b_mask, tail=b_tail)
            _b0 = o0["breaths"][0]
            _relmask = (o0["pres"].squeeze(-1) > 0).float() \
                * (o0["ftype"].argmax(-1) == 0).float()
            _ev = (((_b0["args"].argmax(-1) != o0["args"].argmax(-1)).float()
                    + ((_b0["dup"].squeeze(-1) > 0) != (o0["dup"].squeeze(-1) > 0)).float())
                   .clip(0, 1) * _relmask).detach()
            _bgauth = float(os.environ.get("NAZ_BG", "0.05"))
            _gm = (_bgauth + (1.0 - _bgauth) * _ev).unsqueeze(-1).detach()
            o = forward(p, s_tr, b_tk, b_se, slot_mask=b_mask, tail=b_tail,
                        gmod=_gm)
        else:
            _bd = os.environ.get("BREATH_DROPOUT")
            o = forward(p, s_tr, b_tk, b_se, slot_mask=b_mask, tail=b_tail,
                        drop=(b_drop if _bd else None), lsent=b_ls, reg=b_reg)
        l = loss_fn(o, bg)
        if ALG_CONSUME and "_early" in o:   # support-gated consume-once:
            # any breath claims, each fact pays once; eligibility = the DAG
            _cw = float(os.environ.get("CO_W", "0.5"))
            def _ce(lg, tg):
                return (lg.log_softmax(-1) * -1).gather(-1, tg.unsqueeze(-1)).squeeze(-1)
            def _bce(lg, tg):
                return lg.maximum(0) - lg * tg + (1 + (-lg.abs()).exp()).log()
            _stages = list(o["_early"]) + [o]
            _prev = bg["claimed"] * 1.0   # the ledger seeds prev: paid facts
                                           # never pay again this run
            _P = bg["parents"]              # (B, L, L) adjacency
            _need = _P.sum(-1)
            for _ok in _stages:
                _corr = ((_ok["ftype"].argmax(-1) == bg["ftype"]).float()
                         * (_ok["res"].argmax(-1) == bg["res"]).float())
                _elig = ((_P * _prev.unsqueeze(1)).sum(-1) >= _need - 1e-3).float()
                _first = _corr * _elig * (1.0 - _prev) * bg["presence"]
                _n1 = _first.sum() + 1e-6
                l = l + _cw * ((_ce(_ok["ftype"], bg["ftype"]) * _first).sum()
                               + (_ce(_ok["res"], bg["res"]) * _first).sum()
                               + (_bce(_ok["args"], bg["args"]).mean(-1) * _first).sum()) / _n1
                _prev = (_prev + _first).clip(0, 1)
            _newcl = (_prev - bg["claimed"]).clip(0, 1).realize()
            _NEWCL[0] = _newcl
        if "_early" in o and not ALG_CONSUME:  # deepsup (convicted; kept gated)
            _dw = float(os.environ.get("DEEPSUP_W", "0.3"))   # every breath
            for _ok in o["_early"]:
                l = l + _dw * loss_fn({**_ok, "fat": o["fat"], "vat": o["vat"],
                                       "query": o["query"]}, bg)
        if INV_TR and "fst_s" in o:
            _d = o["fst_s"][0:4:2] - o["fst_s"][1:4:2]
            _pm = bg["presence"][0:4:2].unsqueeze(-1)
            l = l + 0.1 * (_d * _d * _pm).sum() / (_pm.sum() * o["fst_s"].shape[-1] + 1e-6)
        if OPATT:  # dup_staging_cure arm 1: operand-attention TARGET
            _op = bg["opspan"]
            _dm = (_op.sum(-1) > 0).float()
            _opn = _op / (_op.sum(-1, keepdim=True) + 1e-6)
            l = l + float(os.environ.get("OPATT_W", "1")) * ((-(o["fat"] + 1e-9).log() * _opn).sum(-1) * _dm).sum() / (_dm.sum() + 1e-6)
        opt.zero_grad()
        l.backward()
        opt.step()
        return l.realize()

    # HYGIENE (the stack-at-convergence protocol): cosine LR decay + periodic
    # validation on the SMALL test slice (bigtest stays untouched as measurement
    # set) + PICK-BEST-BY-VAL. The overnight constant-lr spike taught this.
    lr_min = lr / 30.0
    val_every = int(os.environ.get("VAL_EVERY", "4000"))

    def _quick_val():
        vs, vst, vtk, vg, vse = load_split_val
        n_ok = n_tot = 0
        for s0 in range(0, len(vs), 8):
            sl = np.arange(s0, min(s0 + 8, len(vs)))
            pad = 8 - len(sl)
            sl_p = np.concatenate([sl, sl[:1].repeat(pad)]) if pad else sl
            o = forward(p, Tensor(vst[sl_p].astype(np.float32), dtype=dtypes.float),
                        Tensor(vtk[sl_p].astype(np.float32), dtype=dtypes.float),
                        Tensor(vse[sl_p].astype(np.int32), dtype=dtypes.int))
            onp = {k: o[k].realize().numpy() for k in
                   (("pres", "ftype", "op", "islit", "dig", "args", "res") + (("dup",) if "h_dup" in p else ()))}
            for bi, i in enumerate(sl):
                i = int(i)
                for j in range(L_FAC):
                    if vg["presence"][i, j] < 0.5:
                        continue
                    n_tot += 1
                    ok = (onp["pres"][bi, j] > 0)
                    ok &= int(onp["ftype"][bi, j].argmax()) == vg["ftype"][i, j]
                    ok &= int(onp["res"][bi, j].argmax()) == vg["res"][i, j]
                    if vg["ftype"][i, j] == 0:
                        ok &= int(onp["op"][bi, j].argmax()) == vg["op"][i, j]
                        gset = set(np.where(vg["args"][i, j] > .5)[0].tolist())
                        if len(gset) == 1 and "dup" in onp:
                            # gen-9: repeated-arg rel — dup bit + top-1 match
                            ok &= bool(onp["dup"][bi, j] > 0)
                            ok &= int(np.argmax(onp["args"][bi, j])) in gset
                        else:
                            top2 = set(np.argsort(-onp["args"][bi, j])[:2].tolist())
                            ok &= top2 == gset
                    else:
                        ok &= bool((onp["dig"][bi, j].argmax(-1) ==
                                    vg["digits"][i, j]).all())
                    n_ok += ok
        return n_ok / max(n_tot, 1)

    load_split_val = load_alg("test")
    best_val, best_snap = -1.0, None

    # CURRICULUM=1 (2026-07-10 ablation): coarse->fine by SAMPLE ORDERING.
    # Teeth score from sample artifacts (post-hoc, deployable-blind): oblique
    # (mention span >2 chars) + shuffled (letter != LETTERS[v]) + irrelevant.
    # Phase 1/3: score==0 only; 2/3: score<=1; 3/3: full mix.
    pools = None
    if int(os.environ.get("CURRICULUM", "0")):
        from characterize_survivors import sample_teeth
        score = np.array([int(t["oblique"]) + int(t["shuffled"])
                          + int(t["irrelevant"])
                          for t in (sample_teeth(s_) for s_ in samples)])
        pools = [np.where(score == 0)[0], np.where(score <= 1)[0],
                 np.arange(n)]
        print(f"[curriculum] pools: easy={len(pools[0])} "
              f"mid={len(pools[1])} full={n}", flush=True)

    # RATION (gen-18 charter, inputs #4/#5): hot-phase upweighting of a
    # marked row set — the dilution law's antidote carries mass AND
    # placement (the boundary toll charges in the hot phase).
    ration_w = None
    if os.environ.get("RATION_FILE"):
        r_idx = json.load(open(os.environ["RATION_FILE"]))
        ration_w = np.ones(n, np.float64)
        ration_w[np.array(r_idx, int)] = float(os.environ.get("RATION_W", "1.5"))
        print(f"[ration] {len(r_idx)} rows upweighted x{os.environ.get('RATION_W', '1.5')} in hot phase", flush=True)
        if os.environ.get("RATION_FILE2"):
            r2 = json.load(open(os.environ["RATION_FILE2"]))
            ration_w[np.array(r2, int)] = np.maximum(
                ration_w[np.array(r2, int)], float(os.environ.get("RATION_W2", "1.5")))
            print(f"[ration2] {len(r2)} rows upweighted x{os.environ.get('RATION_W2', '1.5')} (max-combine on overlap)", flush=True)

    # THE STRAW (2026-08-22, Bryce's air-mattress gut; env ALG_STRAW=1):
    # when the front of the diet collapses (easy mass deflates), uniform
    # sampling seals off the back — a channel keeps pulling from rows
    # that still hold air. v1: class-prior x visit-decay weights riding
    # the ration_w fitting (dialect trickle 0.15, wild 1.0, each visit
    # deflates its row by 1/sqrt(1+visits)). Curriculum-grave cited at
    # registration; regime-tagged to trunk-LoRA joint fires. v2 (loss-EMA
    # straw) deferred pending per-row loss emission from step().
    STRAW = int(os.environ.get("ALG_STRAW", "0"))
    if STRAW:
        _sw_base = np.ones(n, np.float64)
        _sw_h = float(os.environ.get("STRAW_HUMAN", "3.0"))
        def _tier(smp):
            g = smp.get("gen")
            if not (isinstance(g, str) and g.startswith("b22")): return 0.15
            return _sw_h if g == "b22human" else 1.0   # straw v2: three tiers
        _sw_wild = np.array([_tier(smp) for smp in samples], np.float64)
        _sw_visits = np.zeros(n, np.float64)
        ration_w = _sw_base * _sw_wild
        print(f"[straw] armed: wild rows {(1.0 == _sw_wild).sum()} @1.0, "
              f"base {(0.15 == _sw_wild).sum()} @0.15, visit-decay live",
              flush=True)

    t0 = time.time()
    for s in range(steps):
        cur_lr = lr_min + 0.5 * (lr - lr_min) * (1 + math.cos(math.pi * s / steps))
        opt.lr.assign(Tensor([cur_lr], dtype=dtypes.float)).realize()
        pool = (pools[min(3 * s // steps, 2)] if pools is not None
                else np.arange(n))
        if STRAW:
            ration_w = _sw_wild / np.sqrt(1.0 + _sw_visits)
        if ration_w is not None and (STRAW or cur_lr > 0.5 * (lr + lr_min)):
            pw = ration_w[pool] / ration_w[pool].sum()
            idx = rng.choice(pool, batch, replace=False, p=pw)
        else:
            idx = rng.choice(pool, batch, replace=False)
        if STRAW:
            _sw_visits[idx] += 1.0
        if INV_TR:  # inv-fire v2: 2 pairs (pos 0-3) + carrier rows (pos 4-7);
            # pairs from INV_PAIRS side file — the carrier mix rides (the
            # substrate lesson: never train on the patch alone)
            pk = rng.choice(len(INV_PAIRS_ARR), 2, replace=False)
            idx = np.concatenate([INV_PAIRS_ARR[pk].reshape(-1),
                                  rng.choice(n, batch - 4, replace=False)])
        if not TRUNK_LORA:   # audit #15: b_tr is dead under the in-graph trunk
            b_tr.assign(Tensor(states[idx].astype(np.float32), dtype=dtypes.float).contiguous()).realize()
        _rl = [b_tk.assign(Tensor(tokmask[idx].astype(np.float32), dtype=dtypes.float).contiguous()),
               b_se.assign(Tensor(sent[idx].astype(np.int32), dtype=dtypes.int).contiguous())]
        if TRUNK_LORA:
            _rl.append(b_ids.assign(Tensor(IDS_ALL[idx].astype(np.int32), dtype=dtypes.int).contiguous()))
        Tensor.realize(*_rl)   # perf audit #2: one combined schedule, not N dispatches
        if ALG_CONSUME:
            bg["parents"].assign(Tensor(PARENTS[idx], dtype=dtypes.float).contiguous()).realize()
            bg["claimed"].assign(Tensor(CLAIMED[idx], dtype=dtypes.float).contiguous()).realize()
        if b_ls is not None:
            b_ls.assign(Tensor(gold["lsent"][idx].astype(np.float32), dtype=dtypes.float).contiguous()).realize()
        if b_mask is not None:
            _mfeed = MASKS[idx]
            if MG is not None:
                _coin = np.random.RandomState(1000 + s).random(len(idx)) \
                    < float(os.environ.get("ALG_MASK_GOLD_P", "0.5"))
                _mfeed = _mfeed.copy()
                _mfeed[_coin] = MG[idx][_coin].astype(np.float32)
            b_mask.assign(Tensor(_mfeed, dtype=dtypes.float).contiguous()).realize()
        if b_tail is not None:
            b_tail.assign(Tensor(TAILS[idx].astype(np.float32), dtype=dtypes.float).contiguous()).realize()
        if b_reg is not None:
            b_reg.assign(Tensor(REG[idx].astype(np.float32), dtype=dtypes.float).contiguous()).realize()
        if b_drop is not None:
            b_drop.assign(Tensor(np.array(
                [1.0 if rng.rand() >= float(os.environ["BREATH_DROPOUT"]) else 0.0],
                np.float32), dtype=dtypes.float).contiguous()).realize()
        feed = {"presence": gold["presence"][idx], "is_lit_f": gold["is_lit"][idx],
                **({"opspan": OPGOLD[idx].astype(np.float32)} if OPATT else {}),
                "args": gold["args"][idx], "fspan": gold["fspan"][idx],
                "vspan": gold["vspan"][idx], "ftype": gold["ftype"][idx],
                **({"refoh": (np.eye(K_VARS, dtype=np.float32)[
                        np.maximum(gold["refvar"][idx].astype(np.int32), 0)]
                        * (gold["refvar"][idx] >= 0)[..., None])} if ALG_REF and "refvar" in gold else {}),
                **({"is_ind": gold["is_ind"][idx]} if ALG_DIAL and "is_ind" in gold else {}),
                "op": gold["op"][idx], "res": gold["res"][idx],
                "digits": gold["digits"][idx], "query": gold["query"][idx],
                "sel": gold["sel"][idx], "is_rel": gold["is_rel"][idx],
                "is_mod": gold["is_mod"][idx], "is_sel": gold["is_sel"][idx],
                "is_pct": gold["is_pct"][idx], "is_fdiv": gold["is_fdiv"][idx],
                **({"is_chain": gold["is_chain"][idx]}
                   if "is_chain" in gold
                   and int(os.environ.get("ALG_FTYPES", "4")) >= 9 else {}),
                **({"valspan": gold["valspan"][idx]} if ALG_VALATT and "valspan" in gold else {}),
                "arg_dup": (gold["arg_dup"][idx] if "arg_dup" in gold
                            else np.zeros_like(gold["is_rel"][idx])),
                **({"is_macro": gold["is_macro"][idx],
                    "digits2": gold["digits2"][idx],
                    "y": gold["y"][idx]} if "is_macro" in bg else {}),
                **({"is_frac": gold["is_frac"][idx]} if "is_frac" in bg else {}),
                **({"sign": gold["sign"][idx]} if "sign" in bg else {})}
        # THE FEED DOOR (2026-08-25): the buffer list and the feed dict are
        # TWO doors — a terminal registered in one but not the other trains
        # against silent zeros (the opc/posch specimen: emission + gold
        # buffer both present, grads flowing, gold all-zero). Any bg buffer
        # with a same-named gold array rides automatically; new terminals
        # can never miss the feed again.
        for k in bg:
            if k not in feed and k in gold:
                feed[k] = gold[k][idx]
        for k, v in feed.items():
            npdt = np.float32 if bg[k].dtype == dtypes.float else np.int32
            bg[k].assign(Tensor(v.astype(npdt), dtype=bg[k].dtype).contiguous()).realize()
        lv = step()
        if ALG_CONSUME and _NEWCL[0] is not None:
            CLAIMED[idx] = np.clip(CLAIMED[idx] + _NEWCL[0].numpy(), 0, 1)
        if s % 500 == 0 or s == steps - 1:
            v = float(lv.numpy())
            assert np.isfinite(v)
            print(f"  step {s:5d} loss={v:.4f} lr={cur_lr:.1e} "
                  f"({(time.time()-t0)/(s+1):.2f}s/step)", flush=True)
        if (s + 1) % val_every == 0 or s == steps - 1:
            fv = _quick_val()
            mark = ""
            if fv > best_val:
                best_val = fv
                best_snap = {k: t.detach().numpy().copy() for k, t in p.items()}
                mark = "  <-- BEST"
            print(f"  [val @{s+1}] fac-exact-proxy={fv:.4f}{mark}", flush=True)
        # SNAP_EVERY (gut #57's snapshot rider): periodic trajectory ckpts for
        # the wobble census. Snapshots NEVER read bar fixtures (val picks,
        # bars judge); they exist so checkpoint variance can be measured.
        snap_every = int(os.environ.get("SNAP_EVERY", "0"))
        if snap_every and (s + 1) % snap_every == 0:
            sp = ALG_CKPT.replace(".safetensors", f"_s{s+1}.safetensors")
            safe_save(p, sp)
            print(f"  [snap @{s+1}] -> {sp}", flush=True)
    if best_snap is not None and TRUNK_LORA:
        # audit 2026-08-22 #3: _quick_val reads PURE-trunk states — under
        # LoRA it is a mismatched proxy; save FINAL-step params instead
        # (snapshots carry the trajectory for selection)
        print("[train] TRUNK_LORA: best-by-val restore SKIPPED (proxy is "
              "pure-trunk; final-step params saved)", flush=True)
    elif best_snap is not None:
        for k in p:
            p[k].assign(Tensor(best_snap[k], dtype=p[k].dtype)).realize()
        print(f"[train] restored BEST ckpt (val {best_val:.4f})", flush=True)
    safe_save(p, ALG_CKPT)
    print(f"[train] saved {ALG_CKPT} (best-by-val)", flush=True)


# ===========================================================================
# SELFTEST
# ===========================================================================

def selftest():
    os.environ.setdefault("DEV", "CPU")
    from tinygrad import Tensor, dtypes
    p = build_params(0)
    B = 2
    o = forward(p, Tensor(np.random.RandomState(0).randn(B, T_ALG, H_TRUNK).astype(np.float32) * .1),
                Tensor(np.ones((B, T_ALG), np.float32)),
                Tensor(np.zeros((B, T_ALG), np.int32), dtype=dtypes.int))
    assert o["args"].shape == (B, L_FAC, K_VARS) and o["query"].shape == (B, K_VARS)
    g = {"presence": Tensor(np.ones((B, L_FAC), np.float32)),
         "is_lit_f": Tensor(np.zeros((B, L_FAC), np.float32)),
         "args": Tensor(np.zeros((B, L_FAC, K_VARS), np.float32)),
         "fspan": Tensor(np.ones((B, L_FAC, T_ALG), np.float32)),
         "vspan": Tensor(np.ones((B, K_VARS, T_ALG), np.float32)),
         "ftype": Tensor(np.zeros((B, L_FAC), np.int32), dtype=dtypes.int),
         "op": Tensor(np.zeros((B, L_FAC), np.int32), dtype=dtypes.int),
         "res": Tensor(np.zeros((B, L_FAC), np.int32), dtype=dtypes.int),
         "digits": Tensor(np.zeros((B, L_FAC, N_DIG), np.int32), dtype=dtypes.int),
         "query": Tensor(np.zeros((B,), np.int32), dtype=dtypes.int)}
    l = loss_fn(o, g)
    lv = float(l.numpy())
    assert np.isfinite(lv)
    l.backward()
    assert float(p["W_res"].grad.abs().max().numpy()) > 0
    print(f"  [OK] forward/loss/backward (loss={lv:.3f}); pointer grads flow")
    # decode round trip
    onp = {"pres": np.full((L_FAC,), -9., np.float32),
           "ftype": np.zeros((L_FAC, 2), np.float32),
           "op": np.zeros((L_FAC, 2), np.float32),
           "islit": np.full((L_FAC,), -9., np.float32),
           "dig": np.zeros((L_FAC, N_DIG, 10), np.float32),
           "args": np.full((L_FAC, K_VARS), -9., np.float32),
           "res": np.zeros((L_FAC, K_VARS), np.float32),
           "query": np.zeros((K_VARS,), np.float32)}
    onp["pres"][0] = 9.
    onp["args"][0, [2, 5]] = 9.
    onp["res"][0, 7] = 9.
    onp["query"][5] = 9.
    facs, q = decode(onp)
    assert facs == [{"ftype": "rel", "op": "add", "args": [2, 5], "result": 7}]
    assert q == 5
    print("  [OK] decode round trip (rel factor + query pointer)")
    # gold builder on a real corpus line
    samples, ids, mask, offsets = tokenize(ALG_TEST)
    g2 = build_gold(samples[:2], offsets[:2])
    assert g2["vspan"][0].sum() > 0 and g2["fspan"][0].sum() > 0
    print("  [OK] gold builder on real corpus (mentions + factor spans)")
    print("[selftest] PASS")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--precompute", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--errors", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        selftest()
    elif args.precompute:
        do_precompute()
    elif args.train:
        do_train(int(os.environ.get("STEPS", "8000")),
                 float(os.environ.get("LR", "3e-4")),
                 int(os.environ.get("BATCH", "8")),
                 int(os.environ.get("SEED", "0")))
    elif args.eval:
        do_eval()
    elif args.errors:
        do_errors()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
