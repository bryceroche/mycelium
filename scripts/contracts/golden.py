"""golden.py — THE GOLDEN PROBLEMS CONTRACT (hill 5, ledger 2026-10-04 08:09).

A small, FIXED set of problems with known graphs, re-run after every
change, so a regression shows up as "golden24 moved" long before a full
battery would notice. Rows come from the TRAINING diet
(.cache/form_mix_pm35c.jsonl) — never the wild holdout, never
algebra_nl_test (MATH-500's stand-in) — chosen deterministically (seed
241) and NEVER re-chosen; this script refuses to rebuild golden24.jsonl
once it exists (pass --force to deliberately replace it, which also
voids any stored expectation).

THE SIX SHAPES (24 rows, 4 each): givens-only (the shortest shape the
diet actually has — see NOTE below), one relation, two chained
relations (a relation's result feeds a later relation's args), a
percentage (ftype=="pct"), a fraction/division (ftype=="fdiv"), and a
row carrying a same-sentence same-noun twin (scripts/twin_stamp_mint.py's
sidecar .cache/phase1_alg_twin_formpm35c.npz).

NOTE on "givens only": the diet has ZERO rows with no relation at all
(every mint/pen row needs >=1 relation to produce a non-trivial query —
checked directly: 0/50653). The shortest shape the diet actually offers
is exactly one relation over <=3 total factors (one given wired through
one relation, as close to "just the givens" as a solvable row gets);
"one relation" below is kept DISJOINT from it by requiring >=4 factors,
so the two categories are genuinely different shapes, not a relabeling
of the same rows.

THE STATES: load_alg's test path (scripts/phase1_algebra_head.py
~line 2041) needs a cached-states file pair keyed by ALG_TEST_NAME —
.cache/phase1_alg_states_<name>.npz (gold arrays, tokmask, sent) and
_<name>_states.npy (the (n, 256, 2048) trunk memmap from do_precompute,
~1MB/row). Golden rows ARE diet rows (by construction, same text, same
index), so this script SERVES golden24's states by SLICING the diet's
own already-precomputed .cache/phase1_alg_states_formpm35c*.npz/.npy at
the 24 chosen indices — no new trunk forward pass, no GPU, done here on
CPU. (If golden rows ever needed to be NEW text absent from any staged
split, the one-time GPU step would be:
  flock -w 36000 .cache/gpu.lock env DEV=PCI+AMD <FAM env> \\
    ALG_TRAIN=.cache/golden24.jsonl ALG_TRAIN_NAME=golden24 \\
    PRECOMPUTE_ONLY=golden24 .venv/bin/python3 scripts/phase1_algebra_head.py --precompute
  — NOT needed here, and NOT run by this script.)

THE CHECK: reuses scripts/chain_acc.py verbatim (decode -> numeral mask
-> June solver -> row_gold key; CA_ROWS=path dumps the per-row
correct/wrong/refused labels this contract reads). That forward pass
needs the trunk on the card — the one thing in this contract that
touches the GPU. This script never runs it; `golden.py command <ckpt>`
prints the exact line to run WITH THE WORD, and `golden.py check
<rows.json> [--tag NAME]` reads back what that line wrote. The first
`check` for a given tag BANKS the counts into golden24_expect.json (the
pinned expectation); every later check compares.

Subcommands:
  build                 — write golden24.jsonl + golden24_meta.json + the
                           sliced states (idempotent; refuses to rebuild
                           unless --force)
  status                 — print the current golden24 fixture's row/
                           category table and whether states are staged
  command <ckpt> [--mask 0|1] [--tag NAME]
                         — print (not run) the GPU chain_acc.py line
  check <rows.json> --tag NAME
                         — compare a CA_ROWS dump against (or seed)
                           .cache/golden24_expect.json
"""
import argparse
import json
import os
import sys

ROOT = "/home/bryce/mycelium"
DIET_JSONL = os.path.join(ROOT, ".cache/form_mix_pm35c.jsonl")
TWIN_NPZ = os.path.join(ROOT, ".cache/phase1_alg_twin_formpm35c.npz")
STATES_NPZ = os.path.join(ROOT, ".cache/phase1_alg_states_formpm35c.npz")
STATES_NPY = os.path.join(ROOT, ".cache/phase1_alg_states_formpm35c_states.npy")

GOLDEN_JSONL = os.path.join(ROOT, ".cache/golden24.jsonl")
GOLDEN_META = os.path.join(ROOT, ".cache/golden24_meta.json")
GOLDEN_NPZ = os.path.join(ROOT, ".cache/phase1_alg_states_golden24.npz")
GOLDEN_NPY = os.path.join(ROOT, ".cache/phase1_alg_states_golden24_states.npy")
GOLDEN_EXPECT = os.path.join(ROOT, ".cache/golden24_expect.json")

SEED = 241
N_PER_CAT = 4
CATEGORIES = ("givens_only", "one_relation", "two_chained",
              "percentage", "fraction_division", "same_noun_twin")


def classify(rows, has_twin):
    """Pure jsonl read — no gold npz needed (the raw `factors` list
    already carries ftype/args/result in plain integer form)."""
    cats = {c: [] for c in CATEGORIES}
    for i, row in enumerate(rows):
        facs = row["factors"]
        ft = [f["ftype"] for f in facs]
        n_rel = ft.count("rel")
        has_pct = "pct" in ft
        has_fdiv = "fdiv" in ft
        chained = False
        if n_rel >= 2:
            seen_results = set()
            for f in facs:
                if f["ftype"] != "rel":
                    continue
                if any(a in seen_results for a in f.get("args", [])):
                    chained = True
                    break
                seen_results.add(f["result"])
        plain_one_rel = n_rel == 1 and not has_pct and not has_fdiv
        if plain_one_rel and len(facs) <= 3:
            cats["givens_only"].append(i)
        elif plain_one_rel:
            cats["one_relation"].append(i)
        if chained and not has_pct and not has_fdiv:
            cats["two_chained"].append(i)
        if has_pct:
            cats["percentage"].append(i)
        if has_fdiv:
            cats["fraction_division"].append(i)
        if has_twin[i]:
            cats["same_noun_twin"].append(i)
    return cats


def select_golden(rows, has_twin, seed=SEED, n_per_cat=N_PER_CAT):
    """Deterministic (seed 241) pick, 4/category, each row's key verified
    ESTABLISHABLE via custody_gold.row_gold before it is accepted — a
    structural candidate whose key can't be pinned (an unverifiable pen
    row refused by the custody-gold law, or a 'silver'-tier row with no
    solution vector at all) is skipped, never silently stored as a golden
    row with key=null (the whole point of a golden fixture is a KNOWN
    graph and a KNOWN key)."""
    import random
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    sys.path.insert(0, ROOT)
    from mycelium.custody_gold import row_gold
    cats = classify(rows, has_twin)
    rng = random.Random(seed)
    used = set()
    chosen = []   # list of (idx, category)
    skipped = []
    for cat in CATEGORIES:
        pool = [i for i in cats[cat] if i not in used]
        pool.sort()
        rng.shuffle(pool)
        picks = []
        for i in pool:
            try:
                _key_i = int(row_gold(rows[i]))
            except Exception as e:
                skipped.append((i, cat, str(e)[:80]))
                continue
            _sol_i = rows[i].get("solution", []) or []
            _q_i = int(rows[i].get("query_var", 0))
            if _q_i < len(_sol_i) and int(_sol_i[_q_i]) != _key_i:
                skipped.append((i, cat, f"solution[query]={_sol_i[_q_i]} contradicts the custody key {_key_i} (load_alg refuses the fixture)"))
                continue
            if len(rows[i].get("solution", []) or []) <= int(rows[i].get("query_var", 0)):
                skipped.append((i, cat, "solution vector shorter than query_var (load_alg reads solution[query_var])"))
                continue
            picks.append(i)
            if len(picks) == n_per_cat:
                break
        if len(picks) < n_per_cat:
            raise RuntimeError(f"golden: only {len(picks)} key-verifiable candidates for "
                                f"'{cat}', need {n_per_cat} (pool had {len(pool)})")
        for i in sorted(picks):
            chosen.append((i, cat))
            used.add(i)
    if skipped:
        print(f"[golden] {len(skipped)} structurally-matching candidates skipped (key not "
              f"establishable — custody-gold law or missing silver-tier solution), e.g.: "
              + "; ".join(f"idx={i} ({cat}): {msg}" for i, cat, msg in skipped[:3]))
    return chosen


def do_build(force=False):
    if os.path.exists(GOLDEN_JSONL) and not force:
        print(f"[golden] {GOLDEN_JSONL} already exists — refusing to rebuild "
              f"(the fixture is meant to stay FIXED; pass --force to replace it, "
              f"which also deletes any banked golden24_expect.json)")
        return 1
    import numpy as np
    rows = [json.loads(l) for l in open(DIET_JSONL)]
    twin = np.load(TWIN_NPZ)["twin"]
    has_twin = (twin >= 0).any(axis=(1, 2))
    assert len(rows) == len(has_twin), "diet jsonl / twin sidecar row-count mismatch — re-stamp twin_stamp_mint.py"

    chosen = select_golden(rows, has_twin)
    idxs = [i for i, _ in chosen]

    with open(GOLDEN_JSONL, "w") as f:
        for i, _cat in chosen:
            f.write(json.dumps(rows[i]) + "\n")

    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    sys.path.insert(0, ROOT)
    from mycelium.custody_gold import row_gold
    meta = []
    for new_pos, (i, cat) in enumerate(chosen):
        try:
            key = int(row_gold(rows[i]))
        except Exception as e:
            key = None
            print(f"[golden] WARNING: row_gold failed for diet idx {i} ({cat}): {e}")
        meta.append({"golden_pos": new_pos, "diet_idx": i, "category": cat, "key": key,
                     "text": rows[i]["text"][:120]})
    json.dump({"seed": SEED, "n_per_category": N_PER_CAT, "rows": meta},
              open(GOLDEN_META, "w"), indent=1)

    # slice the diet's own precomputed states (CPU only; the diet's trunk
    # forward pass already happened once, at do_precompute() time)
    z = np.load(STATES_NPZ)
    sel = {}
    for k in z.files:
        if k == "mix_sha":
            continue   # test splits aren't SHA-fenced; dropping it avoids a spurious re-check against golden24.jsonl's own (different) content
        sel[k] = z[k][idxs]
    np.savez(GOLDEN_NPZ, **sel)
    big = np.load(STATES_NPY, mmap_mode="r")
    states_sel = np.asarray(big[idxs])   # 24 rows * 1MB — trivial to hold in RAM
    np.save(GOLDEN_NPY, states_sel)
    if os.path.exists(GOLDEN_EXPECT) and force:
        os.remove(GOLDEN_EXPECT)
        print("[golden] --force: cleared the stale golden24_expect.json")

    print(f"[golden] wrote {GOLDEN_JSONL} (24 rows), {GOLDEN_META}, "
          f"{GOLDEN_NPZ}, {GOLDEN_NPY} — sliced from the diet's own cached states, no GPU touched")
    print_status(chosen=chosen)
    return 0


def print_status(chosen=None):
    if not os.path.exists(GOLDEN_META):
        print("[golden] no golden24 fixture yet — run `golden.py build`")
        return 1
    meta = json.load(open(GOLDEN_META))
    print(f"[golden] golden24 (seed {meta['seed']}, {meta['n_per_category']}/category):")
    for r in meta["rows"]:
        print(f"  pos {r['golden_pos']:2d}  diet_idx {r['diet_idx']:6d}  {r['category']:<18} "
              f"key={r['key']!s:<6} {r['text']!r}")
    states_ok = os.path.exists(GOLDEN_NPZ) and os.path.exists(GOLDEN_NPY)
    print(f"[golden] states staged: {states_ok} ({GOLDEN_NPZ}, {GOLDEN_NPY})")
    if os.path.exists(GOLDEN_EXPECT):
        exp = json.load(open(GOLDEN_EXPECT))
        print(f"[golden] expectation banked for tag(s): {sorted(exp.keys())}")
    else:
        print("[golden] no expectation banked yet (the first `check --tag X` writes one)")
    return 0


# The base family env (every chain since 09-18) PLUS the role-signature
# additions (certloop_chain.sh's/caric_chain.sh's "RS8") that PMS8_241 and
# its whole lineage (CL_241, CR_241, CTRLs_241, TR2_241, ...) were trained
# with — this is the default because .cache/sharp_PMS8_241.safetensors is
# the checkpoint every other baseline reference in this campaign is named
# against. A checkpoint from a DIFFERENT lineage (a bare role8-less arm, a
# different dialect) needs --fam pointed at ITS own training env, or
# build_params will mismatch the safetensors keys and chain_acc.py will
# hard-error (by design — "eval-only ckpt loads HARD-ERROR on key
# mismatch", CLAUDE.md §5).
FAM_DEFAULT = ("DEV=PCI+AMD ALG2=1 ALG_FTYPES=9 ALG_DUP=1 ALG_HW=512 ALG_WIDE=1 ALG_BREATH=7 "
               "ALG_NOTEBOOK=1 ALG_SIXWAVE=1 NB_PERSLOT=1 ALG_BINDBUS=7 ALG_BIND_D=512 "
               "ALG_BUSGARAGE=2 ALG_SHELF_CIRCLE=2 ALG_ALTMASK=1 ALG_ALT21=1 ALG_ALT2=1 "
               "ALG_MASKHEAD=1 ALG_FED=1 ALG_POLAR=1 ALG_POLAR_D=128 ALG_POLAR_EM=0.1 "
               "ALG_POLAR_D_INIT=.cache/polar_waist_init_d128u.npz "
               "ALG_PRUNE=pforms,s4,fednl0,lane2 ALG_SLOT_ALL=1 ALG_STELLAR=2 ALG_CLOCK_CANON=1 SC_EVAL=0 "
               "ALG_ROUTER=2 R_GAIN_INIT=1.0 ALG_FREEZE=r_gain ALG_ROUTER_PTR=0.0 ALG_SPAN_ALL=1 "
               "ALG_SPAN_ARGS=1 ALG_SPAN_OP=1 ALG_SPAN_RCUE=1 ALG_SPAN_ARCUE=1 ALG_PTR_SURF=role:add:2.0 "
               "BIND_CODES=.cache/bindbus_codes512r.npz")


def do_command(ckpt, mask, tag, fam):
    if not os.path.exists(GOLDEN_JSONL):
        print("[golden] no golden24.jsonl yet — run `golden.py build` first", file=sys.stderr)
        return 1
    rows_path = f".cache/golden24_rows_{tag}.json"
    cmd = (f'flock -w 36000 .cache/gpu.lock env {fam} '
           f'ALG_TEST=.cache/golden24.jsonl ALG_TEST_NAME=golden24 '
           f'CA_CKPT={ckpt} CA_MASK={mask} CA_ROWS={rows_path} '
           f'.venv/bin/python3 scripts/chain_acc.py > .cache/golden24_check_{tag}.log 2>&1')
    print("# THE ONE GPU STEP in this contract — not run here. Run with the word:")
    print(cmd)
    print(f"# then: .venv/bin/python3 scripts/contracts/golden.py check {rows_path} --tag {tag}")
    return 0


def do_check(rows_path, tag):
    per_row = json.load(open(rows_path))   # {"0": "correct"|"wrong"|"refused", ...} keyed by golden24.jsonl row index
    meta = json.load(open(GOLDEN_META))["rows"] if os.path.exists(GOLDEN_META) else None
    n = len(per_row)
    counts = {"correct": 0, "wrong": 0, "refused": 0}
    detail = []
    for k in sorted(per_row, key=lambda s: int(s)):
        v = per_row[k]
        counts[v] = counts.get(v, 0) + 1
        cat = meta[int(k)]["category"] if meta else "?"
        detail.append((int(k), cat, v))
    print(f"[golden-check] {rows_path} (tag={tag}): {n} rows | "
          f"correct {counts.get('correct', 0)} | wrong {counts.get('wrong', 0)} | refused {counts.get('refused', 0)}")
    for i, cat, v in detail:
        print(f"  pos {i:2d}  {cat:<18} {v}")

    expect = json.load(open(GOLDEN_EXPECT)) if os.path.exists(GOLDEN_EXPECT) else {}
    if tag not in expect:
        expect[tag] = counts
        json.dump(expect, open(GOLDEN_EXPECT, "w"), indent=1)
        print(f"[golden-check] no prior expectation for tag={tag} — BANKED this run as the expectation")
        return 0
    want = expect[tag]
    ok = counts == want
    print(f"[golden-check] expectation for {tag}: {want}")
    print(f"[golden-check] {'PASS' if ok else 'FAIL'} — counts {'match' if ok else 'DIFFER from'} the banked expectation")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build").add_argument("--force", action="store_true")
    sub.add_parser("status")
    pc = sub.add_parser("command")
    pc.add_argument("ckpt")
    pc.add_argument("--mask", default="1")
    pc.add_argument("--tag", default="golden")
    pc.add_argument("--fam", default=FAM_DEFAULT)
    pch = sub.add_parser("check")
    pch.add_argument("rows_json")
    pch.add_argument("--tag", required=True)
    args = ap.parse_args()

    if args.cmd == "build":
        return do_build(force=args.force)
    if args.cmd == "status":
        return print_status()
    if args.cmd == "command":
        return do_command(args.ckpt, args.mask, args.tag, args.fam)
    if args.cmd == "check":
        return do_check(args.rows_json, args.tag)
    return 2


if __name__ == "__main__":
    sys.exit(main())
