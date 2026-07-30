"""book8_certify.py — the certification pass (2026-07-28): the 29 Sonnet
drafts through the STANDARD chain — g21 parse x 5 views, vote >=3, answer
key, attestation v3. The key judges; the annotator never did.

CUSTODY CORRECTION over book7_certify.py: gold comes from the HARVEST
answer (via .cache/book8_candidates.json, keyed by src_idx), never from
the draft's own solution vector — the certifier's key must be independent
of the annotator's artifact (the row's solution descends from the draft's
own graph; using it as gold would let the pen grade itself)."""
import sys, os, json
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
os.environ.setdefault("ALG2", "1"); os.environ.setdefault("ALG_FTYPES", "8")
os.environ.setdefault("ALG_HW", "512"); os.environ.setdefault("ALG_DUP", "1")
import numpy as np
from collections import Counter
from phase1_algebra_head import T_ALG, build_params, forward, decode, sent_indices, TOKENIZER_JSON
from beacon_closing_arm import recompute_states
from tta_views import permuted_view
from tta_alg2_dials import solve2
from mycelium.attestation import attest_quorum_v3
from tokenizers import Tokenizer
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load

tok = Tokenizer.from_file(TOKENIZER_JSON)
p = build_params(0)
sd = safe_load(".cache/g21.safetensors")
assert set(sd.keys()) == set(p.keys())
for k in p: p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()

def parse_batch(texts):
    n = len(texts); N = ((n+7)//8)*8
    ids = np.zeros((N, T_ALG), np.int32); msk = np.zeros((N, T_ALG), np.float32); snt = np.zeros((N, T_ALG), np.int32)
    for i, t in enumerate(texts):
        e = tok.encode(t); L = min(len(e.ids), T_ALG)
        ids[i, :L] = e.ids[:L]; msk[i, :L] = 1.0
        snt[i] = sent_indices(t, list(e.offsets), msk[i])
    st = recompute_states(ids)
    out_r = []
    for s0 in range(0, N, 8):
        out = forward(p, Tensor(st[s0:s0+8].astype(np.float32), dtype=dtypes.float),
                      Tensor(msk[s0:s0+8].astype(np.float32), dtype=dtypes.float),
                      Tensor(snt[s0:s0+8].astype(np.int32), dtype=dtypes.int))
        keys = ("pres","ftype","op","islit","dig","args","res","query") + (("sel",) if "sel" in out else ()) + (("dup",) if "dup" in out else ())
        o = {k: out[k].realize().numpy() for k in keys}
        for bi in range(8):
            if s0+bi < n: out_r.append(decode({k: o[k][bi] for k in o}))
    return out_r

CANDS = os.environ.get("B8_CANDS", ".cache/book8_candidates.json")
TKEY = os.environ.get("B8_TKEY", "tranche1")
DRAFT = os.environ.get("B8_DRAFT", ".cache/book8_prose_pairs_draft.jsonl")
OUT = os.environ.get("B8_OUT", ".cache/book8_certification.json")
cands = json.load(open(CANDS))
# deep clean 2026-07-30: no silent casts on the gold path — answers must be
# clean ints (custody discipline; a string-formatted "$12" would int() to a
# crash here, but a float-ish 12.0 would silently truncate).
for c in cands[TKEY]:
    assert isinstance(c["answer"], int) or (isinstance(c["answer"], str)
        and c["answer"].strip().lstrip("-").isdigit()), \
        f"non-integer gold for src {c['src_idx']}: {c['answer']!r}"
harvest_gold = {int(c["src_idx"]): int(c["answer"]) for c in cands[TKEY]}

rows = [json.loads(l) for l in open(DRAFT)]
res = {"certified": [], "abstain": [], "wrong": []}
for i, r in enumerate(rows):
    src = int(r["gen"]["src_idx"])
    gold = harvest_gold[src]                       # THE KEY: harvest answer, not the draft's artifact
    if r["solution"][r["query_var"]] != gold:
        # GOLD-AUDIT DUTY (bench 2026-07-28): a pen/harvest divergence HALTS
        # the row for the wheel — never resolved silently in either direction
        # (the pen may be wrong, or the harvest may be dirty: [807]).
        # Mismatches accumulate into the harvest-quality record (the intake
        # filter's differential-pressure gauge; thresholds pinned in the
        # ledger: >=2/tranche or >=1% cumulative -> systematic pool audit).
        rec_path = ".cache/harvest_quality_record.json"
        rec = json.load(open(rec_path)) if os.path.exists(rec_path) else []
        rec.append({"src_idx": src, "harvest_gold": gold,
                    "pen_derived": r["solution"][r["query_var"]],
                    "tranche": r["gen"].get("tranche"), "book": r["gen"].get("book")})
        json.dump(rec, open(rec_path, "w"), indent=1)
        res.setdefault("held_for_wheel", []).append(
            {"i": i, "src_idx": src, "harvest_gold": gold,
             "pen_derived": r["solution"][r["query_var"]]})
        print(f"  [{i+1}/{len(rows)}] src {src} HELD FOR WHEEL: pen {r['solution'][r['query_var']]} vs harvest {gold}", flush=True)
        continue
    m = r["m"]   # deep clean 2026-07-30: hard KeyError over silent 300 default
    dialect = r["gen"]["dialect"]   # schema law: text=SOURCE, gen.dialect=the annotation
    texts = [dialect] + [permuted_view(dialect, 91000 + 10*i + k) for k in range(1, 5)]
    views = [(f, q, solve2(f, q, {"n_vars": 24, "m": m})) for f, q in parse_batch(texts)]
    votes = [a for _, _, a in views]
    nn = [a for a in votes if a is not None]
    c = Counter(nn).most_common(1); plur, cnt = c[0] if c else (None, 0)
    if cnt >= 3 and plur == gold:
        # BASIN TRIPWIRE (2026-07-29, the [1293] panel finding): the
        # sel-basin's signature is text-side detectable — dialect contains
        # division language but the WINNING parses contain no division
        # factor. Substrate-invariant basins are view-invariant AND
        # panel-invariant; this lexical-vs-parse check is the mechanical
        # fence. Flagged quorums hold for the wheel, never certify silently.
        # EXTENDED to the full lexical-trace inventory (2026-07-29): all seven
        # markers measured 100% gold-consistent on 644 marker-rows — the fence
        # template's coverage map, applied. Marker in dialect => factor type
        # in the winning parse, else hold.
        win_parses = [f_ for (f_, q_, a_) in views if a_ == plur]
        def _wf(pred): return any(pred(fa) for f_ in win_parses for fa in f_)
        TRIPS = [
            ("division", (" divided by " in dialect) or ("quotient" in dialect),
             _wf(lambda fa: fa.get("ftype") in ("fdiv", "mod"))),
            ("percent", " percent" in dialect, _wf(lambda fa: fa.get("ftype") == "pct")),
            ("selection", (" the larger of " in dialect) or (" the smaller of " in dialect),
             _wf(lambda fa: fa.get("ftype") == "sel")),
            ("times", " times " in dialect, _wf(lambda fa: fa.get("ftype") == "rel" and fa.get("op") == "mul")),
            ("plus", (" plus " in dialect) or ("The sum of" in dialect),
             _wf(lambda fa: fa.get("ftype") == "rel" and fa.get("op") == "add")),
            ("remainder", "remainder" in dialect, _wf(lambda fa: fa.get("ftype") == "mod")),
        ]
        # THE ONE TRACE LAYER (2026-07-30, deep-clean unification): markers +
        # value consistency now live in mycelium/trace_layer.py — the single
        # fence both this certifier and the wild frontier fixture import.
        # (The deep clean found this file's division marker narrower than the
        # fixture's: no \frac/\div, literal ' divided by ' only. Two fences
        # with the same name must be the same fence.) The local TRIPS list
        # above is superseded; shared trips are authoritative.
        from mycelium.trace_layer import trace_trips
        tripped = sorted(set(
            [name for name, lang, present in TRIPS if lang and not present]
            + trace_trips(dialect, win_parses)))
        if tripped:
            res.setdefault("held_basin_tripwire", []).append({"i": i, "src_idx": src, "votes": votes, "tripped": tripped})
            print(f"  [{i+1}/{len(rows)}] src {src} BASIN TRIPWIRE ({','.join(tripped)}): marker language, no matching factor in winning parse — HELD", flush=True)
            continue
        att, _, _ = attest_quorum_v3(views, plur, dialect, m, solve2)
        res["certified" if att else "abstain"].append({"i": i, "src_idx": src, "votes": votes})
    elif cnt >= 3:
        res["wrong"].append({"i": i, "src_idx": src, "plur": plur, "gold": gold, "votes": votes})
    else:
        res["abstain"].append({"i": i, "src_idx": src, "votes": votes})
    watch = r["gen"].get("watch")
    print(f"  [{i+1}/{len(rows)}] src {src}{' WATCH:'+watch if watch else ''} votes {votes} gold {gold} -> {'CERT' if cnt>=3 and plur==gold else ('WRONG' if cnt>=3 else 'abstain')}", flush=True)
print(f"\n=== BOOK 8 {TKEY.upper()} CERTIFICATION: certified {len(res['certified'])} | abstain {len(res['abstain'])} | wrong {len(res['wrong'])} ===")
json.dump(res, open(OUT, "w"), indent=1, default=int)
