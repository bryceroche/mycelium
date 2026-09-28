"""twin_stamp_mint.py -- THE CARICATURE MARGIN's twin stamp (2026-09-27, zero-GPU,
zero-training; built after THE TWIN-GAP READ, docs/phase1_skeleton_spec.md
"2026-09-27 (15:34)"). Writes, per split, .cache/phase1_alg_twin_<split>.npz:
"twin" int16 (n, L_FAC, 2) and "twin_gold" int16 (n, L_FAC, 2) -- for relation
slot j and argument position a (0/1), the pointer-space VARIABLE id of the
strongest same-sentence SAME-NOUN twin of the gold argument (-1 if none) and
the gold argument's own variable id at that position (-1 alongside it).
scripts/phase1_algebra_head.py's load_alg merges this sidecar into the gold
dict under ALG_CARIC, the same pattern as THE NESTED LADDER's
phase1_alg_hier_<split>.npz (see load_alg's ALG_CARIC block).

Reuses, never reimplements: scripts/lexical_identity_census.py's
build_candidate_tables/same_sentence and scripts/mixed_bucket_census.py's
arg_closure/inherited_keys (KEY-INH) -- the exact same machinery
scripts/twin_gap.py's read used to define "same-noun twin"; SAME-NOUN
membership here is is_twin() below, twin_gap.py's per-competitor predicate
copied verbatim (derived-from-k, or key-set overlap with k's KEY-INH).

SCOPE (matches twin_gap.py's docstring, and THE CARICATURE MARGIN's own bar,
which is pinned on this exact read): only gold ftype=="rel" slots -- the
args-logit bilinear pointer this margin trains. pct/sel/mod/fdiv have
different pointer shapes and are out of scope (not stamped: left at -1 like
everywhere else a twin doesn't apply). A slot/position with no same-sentence
competitor at all, or no competitor that clears the SAME-NOUN bar, is also
left at -1 -- this is the overwhelming case for MINT rows (synthetic,
letter-variable rows carry no literal one-token key almost anywhere, and the
census below states the resulting share explicitly rather than assuming it).

TWIN CHOICE when >1 same-sentence competitor is SAME-NOUN (the "any
deterministic choice" the build brief allows): the competitor whose clause
sits nearest (sentence distance) to k's clause, ties broken by the lower
slot index -- this is a proxy for "the twin actually competing hardest at
read time", not a claim about which twin is objectively hardest; the
per-position, per-slot GAP read (twin_gap.py) is the place that measures
hardness, not this stamp.

Run (any split; DEV is irrelevant -- no tinygrad forward is called, only
phase1_algebra_head's L_FAC/K_VARS module constants are imported):
  .venv/bin/python3 scripts/twin_stamp_mint.py
runs the DEFAULT_SPLITS list below (every split load_alg's ALG_CARIC block
or the CPU gate's tiny64 fixture will ever ask for); pass name=path pairs on
the command line to stamp specific splits instead, e.g.:
  .venv/bin/python3 scripts/twin_stamp_mint.py formpm35c=.cache/form_mix_pm35c.jsonl
"""
import json
import os
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")

import numpy as np

import lexical_identity_census as LIC   # build_candidate_tables, same_sentence
import mixed_bucket_census as MBC       # arg_closure, inherited_keys (KEY-INH)
import stamp_arg_mentions as SAM        # own_var
import phase1_algebra_head as H         # L_FAC / K_VARS module constants only -- no forward, no build_params

L_FAC = H.L_FAC
K_VARS = H.K_VARS

# every split load_alg's ALG_CARIC block or the CPU gate can ask for: the
# train diet, its test fixture, the slice gates' valid slice, the wild
# holdout (READ only -- wild is NEVER trained on; this sidecar is built for
# a load_alg("wildhold") READ path only, never for a train split), and the
# CPU gate's tiny64/testtiny64 mint fixture (train side needs the sidecar to
# exist under ALG_CARIC or load_alg hard-errors).
DEFAULT_SPLITS = [
    ("formpm35c",       ".cache/form_mix_pm35c.jsonl"),           # the train diet
    ("test23",          ".cache/algebra_nl_test.jsonl"),          # its test fixture
    ("pm35cslice",      ".cache/form_pm35c_slice1024.jsonl"),     # the slice gates' TRAIN slice (1,024 rows)
    ("pm35cslicevalid2", ".cache/form_pm35c_slice1024_valid2.jsonl"),  # the slice gates' valid slice
    ("wildhold",        ".cache/wild_admitted_holdout.jsonl"),    # READ only -- never a train split
    ("tiny64",          ".cache/form_tiny64.jsonl"),              # the CPU gate's mint train fixture
    ("testtiny64",      ".cache/test_tiny64.jsonl"),              # the CPU gate's mint test fixture
]


def is_twin(derived, inh_k, comp_inh):
    """per-competitor SAME-NOUN membership -- twin_gap.py's is_twin(), copied
    verbatim (derived-from-k, or KEY-INH overlap with k's own set)."""
    if derived:
        return True
    return bool(inh_k and (comp_inh & inh_k))


def twin_for_row(row):
    (text, factors, solution, bounds, sspans, intro_map, memo,
     cand_key, cand_sent) = LIC.build_candidate_tables(row)
    cmemo = {}
    closure = {i: MBC.arg_closure(i, factors, intro_map, cmemo) for i in range(len(factors))}
    inh = {i: MBC.inherited_keys(i, factors, cand_key, closure) for i in range(len(factors))}
    twin = np.full((L_FAC, 2), -1, dtype=np.int16)
    twin_gold = np.full((L_FAC, 2), -1, dtype=np.int16)
    n_scoreable = 0   # rel-argument instances with >= 1 same-sentence competitor
    n_twinned = 0     # of those, a SAME-NOUN twin was found and stamped
    for j, fac in enumerate(factors):
        if j >= L_FAC or fac.get("ftype") != "rel":
            continue
        args = fac.get("args")
        if not args:
            continue
        for pos, v in enumerate(args):
            if pos >= 2:
                break
            k = intro_map.get(v)
            if k is None or k == j:
                continue
            k_sent = cand_sent.get(k, set())
            competitors = [c for c in range(len(factors))
                           if c != j and c != k and LIC.same_sentence(cand_sent.get(c, set()), k_sent)]
            if not competitors:
                continue
            n_scoreable += 1
            best_c, best_dist = None, None
            for c in competitors:
                derived = k in closure[c]
                if not is_twin(derived, inh[k], inh[c]):
                    continue
                cv = SAM.own_var(factors[c])
                if cv is None or cv == v:   # the competitor's own variable IS the gold arg: not a twin
                    continue
                c_sent = cand_sent.get(c, set())
                dist = (min(abs(a - b) for a in c_sent for b in k_sent)
                        if (c_sent and k_sent) else 10 ** 6)
                if best_dist is None or dist < best_dist or (dist == best_dist and c < best_c):
                    best_dist, best_c = dist, c
            if best_c is None:
                continue
            twin[j, pos] = SAM.own_var(factors[best_c])
            twin_gold[j, pos] = v
            n_twinned += 1
    return twin, twin_gold, n_scoreable, n_twinned


def stamp_split(name, path):
    rows = [json.loads(l) for l in open(path)]
    n = len(rows)
    twins = np.full((n, L_FAC, 2), -1, dtype=np.int16)
    golds = np.full((n, L_FAC, 2), -1, dtype=np.int16)
    n_rows_with_twin = 0
    tot_scoreable = 0
    tot_twinned = 0
    per_row_twins = []
    for i, row in enumerate(rows):
        tw, gd, n_sc, n_tw = twin_for_row(row)
        twins[i] = tw
        golds[i] = gd
        tot_scoreable += n_sc
        tot_twinned += n_tw
        per_row_twins.append(n_tw)
        if n_tw > 0:
            n_rows_with_twin += 1
    out_path = f".cache/phase1_alg_twin_{name}.npz"
    np.savez(out_path, n=np.array(n), twin=twins, twin_gold=golds)
    mean_twins = float(np.mean(per_row_twins)) if per_row_twins else 0.0
    share = tot_twinned / max(tot_scoreable, 1)
    print(f"[twin-stamp] {name:16s} n={n:6d}  rows_with_twin={n_rows_with_twin:5d} "
          f"({n_rows_with_twin / max(n, 1):.3f})  twins/row={mean_twins:.3f}  "
          f"scoreable_rel_args={tot_scoreable:6d}  twinned={tot_twinned:6d}  "
          f"share_of_scoreable_with_twin={share:.3f}  -> {out_path}", flush=True)
    return dict(name=name, path=path, n=n, rows_with_twin=n_rows_with_twin,
                mean_twins=mean_twins, scoreable=tot_scoreable, twinned=tot_twinned, share=share)


def main():
    if len(sys.argv) > 1:
        splits = []
        for a in sys.argv[1:]:
            name, path = a.split("=", 1)
            splits.append((name, path))
    else:
        splits = DEFAULT_SPLITS
    print("=" * 92)
    print("THE CARICATURE MARGIN's TWIN STAMP (2026-09-27)")
    print("=" * 92)
    results = []
    for name, path in splits:
        if not os.path.exists(path):
            print(f"[twin-stamp] SKIP {name}: {path} not found")
            continue
        results.append(stamp_split(name, path))
    print("-" * 92)
    for r in results:
        print(f"  {r['name']:16s} rows_with_twin_share={r['rows_with_twin'] / max(r['n'], 1):.3f}  "
              f"twins/row={r['mean_twins']:.3f}  share_of_scoreable_with_twin={r['share']:.3f}")


if __name__ == "__main__":
    main()
