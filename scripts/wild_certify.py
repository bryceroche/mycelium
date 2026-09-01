"""wild_certify.py — the round-7 + GSM8K wild-lane certification pass
(2026-09-01). Adaptation of book8_certify.py to the WILD register: rows
are draft graphs over ORIGINAL problem text (no dialect), from two lanes:

  (A) .cache/book14_r7_drafts_{a,b,c}.json — {"drafts": [...]} rows with
      src_idx (harvest idx), original, answer, n_vars, m, query_var, factors.
  (B) .cache/gsm8k_wild_drafts.jsonl — rows with src="gsm8k_train",
      src_idx, original, answer, n_vars, m, query_var, factors.

STANDARD chain kept from book8_certify: manifest-driven gate checkpoint
(B8_CKPT override honored), 5 permuted views (tta_views.permuted_view),
parse each view, vote >=3, solve via tta_alg2_dials.solve2, grade against
an INDEPENDENT key.

THE KEY RULE (custody): never the row's own answer field. The key routes
through mycelium.custody_gold (which carries both the MATH harvest table
and the GSM8K #### key table), looked up by original.strip() — text
identity, the one cross-book key. A text missing from both tables is
REFUSED loudly (counted, never crashed, never fallen back).

TWO-BIT VERDICT: every keyed row is also graded MECHANICALLY on its own
draft graph (propagate givens through add/mul factors, inverse moves
included) -> (graph_true, vote_pass). Four buckets:
  certified             — vote>=3 on the key AND solved==key AND graph_true
  graph_true_vote_fail  — the register wall: true rows the parser can't
                          yet see (diet-admission candidates)
  vote_pass_graph_false — alarm bucket, expected ~0 (graph mechanically
                          DERIVES a value contradicting the key)
  refused_other         — no custody key / non-integer key / both bits
                          off / graph INDETERMINATE (simultaneous-equation
                          graphs forward propagation cannot decide — those
                          never certify and never raise the alarm)

ENV NOTE (deviation from book8_certify, deliberate): the manifest's env
dict is applied via setdefault BEFORE the head import — book8's four
setdefaults omit ALG_WIDE=1, which the deployed g41 gate REQUIRES (sign
arm + 7-digit banks change the param set; the ckpt-key assert would fail
without it). book8's parse_batch key list also drops sgn/dig2/y, which
decode consumes under the deployed envs; the list here is the decode-
complete superset, availability-gated.

Attestation v3 and the basin tripwire are NOT carried: attestation.py's
own jurisdiction note bars the literal fence on wild registers until the
number-word extractor upgrade, and trace_trips is calibrated on the
dialect register (division prose in wild GSM8K text would misfire it).

Knobs: WC_INPUTS (comma list of input files, default all four),
WC_LIMIT (int cap per input file, for smoke). GPU work (trunk + head)
happens only under __main__.

USAGE: DEV=AMD .venv/bin/python3 scripts/wild_certify.py
"""
import sys, os, json
sys.path.insert(0, "."); sys.path.insert(0, "scripts")

# --- env: manifest first (the authority — deployed envs ARE the manifest's),
# book8's setdefaults as backstop. Must precede any phase1_algebra_head
# import: ALG_WIDE/ALG_HW are read at module level and set param SHAPES.
MANIFEST = ".cache/GENERATION.json"
if os.path.exists(MANIFEST):
    for _k, _v in json.load(open(MANIFEST)).get("env", {}).items():
        os.environ.setdefault(_k, str(_v))
os.environ.setdefault("ALG2", "1"); os.environ.setdefault("ALG_FTYPES", "8")
os.environ.setdefault("ALG_HW", "512"); os.environ.setdefault("ALG_DUP", "1")

from collections import Counter
from math import isqrt

from mycelium.custody_gold import row_gold   # pure file reads; no GPU

DEFAULT_INPUTS = [
    ".cache/book14_r7_drafts_a.json",
    ".cache/book14_r7_drafts_b.json",
    ".cache/book14_r7_drafts_c.json",
    ".cache/gsm8k_wild_drafts.jsonl",
]
OUT_RESULTS = ".cache/wild_cert_results.json"
OUT_R7 = ".cache/book14_certified.jsonl"
OUT_GSM = ".cache/gsm8k_certified.jsonl"


def load_inputs():
    """[(source_tag, file, row)] — 'gsm8k' if the row says so, else 'r7'.
    WC_LIMIT caps rows PER FILE (smoke)."""
    paths = [p for p in os.environ.get(
        "WC_INPUTS", ",".join(DEFAULT_INPUTS)).split(",") if p]
    limit = int(os.environ["WC_LIMIT"]) if os.environ.get("WC_LIMIT") else None
    out = []
    for path in paths:
        if path.endswith(".jsonl"):
            rows = [json.loads(l) for l in open(path)]
        else:
            rows = json.load(open(path))["drafts"]
        if limit is not None:
            rows = rows[:limit]
        for r in rows:
            tag = "gsm8k" if r.get("src") == "gsm8k_train" else "r7"
            out.append((tag, path, r))
    return out


def custody_key(text, src_idx=None):
    """Independent gold via the custody table, keyed by text identity.
    Returns (int_key, None) or (None, refusal_reason). Never the row's
    own answer; never a silent fallback."""
    try:
        return row_gold({"text": text, "gen": {"src_idx": src_idx}}), None
    except KeyError:
        return None, "no_custody_key"
    except ValueError as e:
        return None, f"non_integer_key: {e}"


class _Conflict(Exception):
    pass


def propagate_graph(factors):
    """Mechanical fixpoint propagation of the row's OWN draft graph:
    givens seed values; add/mul rels fire forward and by inverse moves
    (r-a for add, exact division for mul; duplicate-arg inverses r/2 and
    integer sqrt). Returns (vals_dict, None) or (None, reason)."""
    vals = {}

    def put(v, x):
        if v in vals:
            if vals[v] != x:
                raise _Conflict
            return False
        vals[v] = x
        return True

    try:
        changed = True
        while changed:
            changed = False
            for f in factors:
                ft = f["ftype"]
                if ft == "given":
                    changed |= put(f["var"], int(f["value"]))
                    continue
                if ft != "rel":
                    return None, f"ungradeable_ftype:{ft}"
                args, r, op = list(f["args"]), f["result"], f["op"]
                known = [vals[a] for a in args if a in vals]
                unk = [a for a in args if a not in vals]
                if not unk:
                    if op == "add":
                        changed |= put(r, sum(known))
                    else:
                        pk = 1
                        for k in known:
                            pk *= k
                        changed |= put(r, pk)
                elif r in vals and len(set(unk)) == 1:
                    u, c = unk[0], len(unk)
                    if op == "add":
                        rem = vals[r] - sum(known)
                        if rem % c == 0:
                            changed |= put(u, rem // c)
                    else:
                        pk = 1
                        for k in known:
                            pk *= k
                        if pk != 0 and vals[r] % pk == 0:
                            q = vals[r] // pk
                            if c == 1:
                                changed |= put(u, q)
                            elif c == 2 and q >= 0 and isqrt(q) ** 2 == q:
                                changed |= put(u, isqrt(q))
    except _Conflict:
        return None, "graph_conflict"
    return vals, None


def graph_verdict(row, key):
    """Three-valued: (True|False|None, detail). True/False = the graph
    mechanically forces a value at query_var that matches/contradicts the
    key; a conflicted graph is False (it cannot be true). None =
    INDETERMINATE — forward propagation with inverse moves cannot reach
    the query (simultaneous-equation graphs); such rows never certify but
    also never raise the alarm bucket."""
    vals, why = propagate_graph(row["factors"])
    if vals is None:
        if why == "graph_conflict":
            return False, why
        return None, why
    gv = vals.get(row["query_var"])
    if gv is None:
        return None, "query_underdetermined"
    return gv == key, f"graph_val={gv}"


def main():
    # --- GPU-side setup: everything below touches the device; __main__ only.
    import numpy as np
    from phase1_algebra_head import (
        T_ALG, build_params, forward, decode, sent_indices, TOKENIZER_JSON)
    from beacon_closing_arm import recompute_states
    from tta_views import permuted_view
    from tta_alg2_dials import solve2
    from tokenizers import Tokenizer
    from tinygrad import Tensor, dtypes
    from tinygrad.nn.state import safe_load

    tok = Tokenizer.from_file(TOKENIZER_JSON)
    p = build_params(0)
    # two-home fix inherited from book8: manifest is the authority,
    # B8_CKPT the explicit bench override.
    ckpt = os.environ.get("B8_CKPT") or json.load(open(MANIFEST))["parser_ckpt"]
    gate = os.path.basename(ckpt)
    print(f"[wild_certify] gate from manifest: {ckpt}")
    sd = safe_load(ckpt)
    assert set(sd.keys()) == set(p.keys()), (
        f"ckpt/param key mismatch (env drift? manifest env = "
        f"{json.load(open(MANIFEST)).get('env')})")
    for k in p:
        p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()

    # decode-complete key superset (book8's list dropped sgn/dig2/y —
    # mis-decodes negative literals and FRAC_OF/OP_APPLY under g41 envs).
    DEC_KEYS = ("pres", "ftype", "op", "islit", "dig", "args", "res",
                "query", "sel", "dup", "dargs", "dig2", "y", "sgn")

    def parse_batch(texts):
        n = len(texts); N = ((n + 7) // 8) * 8
        ids = np.zeros((N, T_ALG), np.int32)
        msk = np.zeros((N, T_ALG), np.float32)
        snt = np.zeros((N, T_ALG), np.int32)
        for i, t in enumerate(texts):
            e = tok.encode(t); L = min(len(e.ids), T_ALG)
            ids[i, :L] = e.ids[:L]; msk[i, :L] = 1.0
            snt[i] = sent_indices(t, list(e.offsets), msk[i])
        st = recompute_states(ids)
        out_r = []
        for s0 in range(0, N, 8):
            out = forward(p, Tensor(st[s0:s0 + 8].astype(np.float32), dtype=dtypes.float),
                          Tensor(msk[s0:s0 + 8].astype(np.float32), dtype=dtypes.float),
                          Tensor(snt[s0:s0 + 8].astype(np.int32), dtype=dtypes.int))
            o = {k: out[k].realize().numpy() for k in DEC_KEYS if k in out}
            for bi in range(8):
                if s0 + bi < n:
                    out_r.append(decode({k: o[k][bi] for k in o}))
        return out_r

    items = load_inputs()
    print(f"[wild_certify] {len(items)} rows from "
          f"{len(set(f for _, f, _ in items))} file(s)")

    res = {"gate": gate, "certified": [], "graph_true_vote_fail": [],
           "vote_pass_graph_false": [], "refused_other": []}
    cert_rows = {"r7": [], "gsm8k": []}

    for i, (src, path, r) in enumerate(items):
        text = r["original"]
        sidx = r.get("src_idx")
        key, why = custody_key(text.strip(), sidx)
        rec = {"source": src, "file": path, "src_idx": sidx}
        if key is None:
            rec.update({"reason": why})
            res["refused_other"].append(rec)
            print(f"  [{i+1}/{len(items)}] {src} src {sidx} REFUSED: {why}",
                  flush=True)
            continue
        gtrue, gwhy = graph_verdict(r, key)

        texts = [text] + [permuted_view(text, 97000 + 10 * i + k)
                          for k in range(1, 5)]
        m = r["m"]   # hard KeyError over silent default (book8 deep clean)
        views = [(f, q, solve2(f, q, {"n_vars": 24, "m": m}))
                 for f, q in parse_batch(texts)]
        votes = [a for _, _, a in views]
        nn = [a for a in votes if a is not None]
        c = Counter(nn).most_common(1)
        plur, cnt = c[0] if c else (None, 0)
        vote_pass = cnt >= 3 and plur == key

        rec.update({"key": key, "votes": votes, "plur": plur, "cnt": cnt,
                    "graph_true": gtrue, "vote_pass": vote_pass,
                    "graph_detail": gwhy})
        if vote_pass and gtrue is True:
            res["certified"].append(rec)
            cert_rows[src].append(dict(
                r, cert={"views": cnt, "gate": gate, "key": key}))
            tag = "CERT"
        elif gtrue is True:
            res["graph_true_vote_fail"].append(rec)
            tag = "WALL"     # register wall: true row the parser can't see
        elif vote_pass and gtrue is False:
            res["vote_pass_graph_false"].append(rec)
            tag = "ALARM"    # vote reaches the key, the graph CONTRADICTS it
        else:
            vwhy = ("vote_pass" if vote_pass else
                    "quorum_wrong_answer" if cnt >= 3 else "no_quorum")
            rec["reason"] = f"{vwhy}|graph:{gwhy}"
            res["refused_other"].append(rec)
            tag = "other"
        gt = {True: "T", False: "F", None: "?"}[gtrue]
        print(f"  [{i+1}/{len(items)}] {src} src {sidx} votes {votes} "
              f"key {key} graph={gt} -> {tag}", flush=True)

    # --- report: per bucket, per source, gsm8k depth split -----------------
    buckets = ("certified", "graph_true_vote_fail", "vote_pass_graph_false",
               "refused_other")
    res["counts"] = {b: len(res[b]) for b in buckets}
    print(f"\n=== WILD CERTIFICATION ({gate}) ===")
    print(" | ".join(f"{b} {len(res[b])}" for b in buckets))
    for src in ("r7", "gsm8k"):
        cs = {b: sum(1 for e in res[b] if e["source"] == src) for b in buckets}
        print(f"  [{src}] " + " | ".join(f"{b} {n}" for b, n in cs.items()))

    gsm_depth = {"deep": Counter(), "shallow": Counter()}
    by_idx = {r.get("src_idx"): (sum(1 for f in r["factors"]
                                     if f["ftype"] == "rel") >= 5)
              for _, _, r in items if r.get("src") == "gsm8k_train"}
    for b in buckets:
        for e in res[b]:
            if e["source"] == "gsm8k":
                gsm_depth["deep" if by_idx.get(e["src_idx"]) else "shallow"][b] += 1
    res["gsm8k_depth_split"] = {k: dict(v) for k, v in gsm_depth.items()}
    print(f"  [gsm8k depth >=5 rels] " + " | ".join(
        f"{b} {gsm_depth['deep'].get(b, 0)}" for b in buckets))
    print(f"  [gsm8k depth <5 rels ] " + " | ".join(
        f"{b} {gsm_depth['shallow'].get(b, 0)}" for b in buckets))

    json.dump(res, open(OUT_RESULTS, "w"), indent=1, default=int)
    for src, path in (("r7", OUT_R7), ("gsm8k", OUT_GSM)):
        with open(path, "w") as fh:
            for row in cert_rows[src]:
                fh.write(json.dumps(row, default=int) + "\n")
        print(f"[wild_certify] {len(cert_rows[src])} certified {src} "
              f"rows -> {path}")
    print(f"[wild_certify] results -> {OUT_RESULTS}")


if __name__ == "__main__":
    main()
