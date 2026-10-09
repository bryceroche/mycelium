"""build_inverse_dose_diet.py -- THE INVERSE DOSE, applied to form_mix_pm35c (2026-10-08). Splices a
freshly inverse-dosed mint subsample (SAME row count as pm35c's own mint rows) into pm35c's EXISTING
prose rows (pen + silver, kept byte-identical), then runs the replacement mint rows through the SAME
three-stage stamping pipeline pm35/pm35v/pm35a/pm35c were built with (stamp_value_spans.py ->
stamp_arg_mentions.py -> stamp_given_cues.stamp_row, all three per-row/deterministic, reused not
reimplemented) so the output carries the identical per-factor fields (mentions, arg_spans, cue_spans,
role_cues, role_order, arg_role_cues) pm35c's own mint rows carry.

Selection (the SAME bucket method build_diet_v2.py's --inverse-frac implements, for the standalone
splice case where the historic pm35/pm35c build invocation's exact pen/silver composition and RNG
seed are not reproducible, so a full build_diet_v2.py re-run from the raw sources is not attempted):
mint candidates = form_mix12.jsonl's rows with gen.src is None (THE MINT POOL pm35's own mint rows
were drawn from -- 120,100 rows, confirmed against pm35c's own mint-row count by `gen.src is None`
tally). Classified fwd/inv by polarity_census.classify_row (replaying each row's own variable-
introduction order); bucket A (inv>fwd) and bucket B (the rest) each independently shuffled; x solves
x*shareA + (1-x)*shareB >= target.

usage: .venv/bin/python3 scripts/build_inverse_dose_diet.py
outputs: .cache/form_mix_pm35d.jsonl
"""
import json, sys, os, random, collections, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from polarity_census import classify_row

IN_DIET = ".cache/form_mix_pm35c.jsonl"
POOL = ".cache/form_mix12.jsonl"
OUT = ".cache/form_mix_pm35d.jsonl"
TARGET_INVERSE = 0.58   # aims to comfortably clear the >=0.50 bar after sampling noise
SEED = 243              # pm35/pm35c used 241/242; 243 keeps this dose arm's draw distinct


def G(r):
    return r.get("gen") if isinstance(r.get("gen"), dict) else {}


def inv_fwd(r):
    c = classify_row(r["factors"])
    return sum(1 for v in c if v == "fwd"), sum(1 for v in c if v == "inv")


def dose_law_legs(rows, label):
    af = ai = 0
    for r in rows:
        nf, ni = inv_fwd(r)
        af += nf; ai += ni
    share = ai / max(af + ai, 1)
    texts = collections.Counter(r["text"] for r in rows)
    reps_per_unique = len(rows) / max(len(texts), 1)
    print(f"[dose-law] {label}: {len(rows)} rows, {af} fwd + {ai} inv relation slots "
          f"-> share-of-mix(inverse)={share:.4f}; reps-per-unique={reps_per_unique:.4f} "
          f"({len(rows)} rows / {len(texts)} distinct texts)", flush=True)
    return share, reps_per_unique


def main():
    t0 = time.time()
    rng = random.Random(SEED)
    diet_rows = [json.loads(l) for l in open(IN_DIET)]
    is_mint = [G(r).get("src") is None for r in diet_rows]
    n_mint = sum(is_mint)
    prose_rows = [r for r, m in zip(diet_rows, is_mint) if not m]
    old_mint_rows = [r for r, m in zip(diet_rows, is_mint) if m]
    print(f"[split] {IN_DIET}: {len(diet_rows)} rows -> {len(prose_rows)} prose + "
          f"{n_mint} mint (gen.src is None)", flush=True)
    dose_law_legs(old_mint_rows, "pm35c mint (BEFORE, unbiased)")

    pool = [r for r in (json.loads(l) for l in open(POOL)) if G(r).get("src") is None]
    print(f"[pool] {POOL}: {len(pool)} mint candidates (gen.src is None)", flush=True)
    stats = [inv_fwd(r) for r in pool]
    bucketA = [r for r, (nf, ni) in zip(pool, stats) if ni > nf]
    bucketB = [r for r, (nf, ni) in zip(pool, stats) if ni <= nf]
    statsA = [(nf, ni) for nf, ni in stats if ni > nf]
    statsB = [(nf, ni) for nf, ni in stats if ni <= nf]
    fA = sum(nf + ni for nf, ni in statsA) / max(len(statsA), 1)   # avg relation-slots/row, bucket A
    fB = sum(nf + ni for nf, ni in statsB) / max(len(statsB), 1)   # bucket B
    shareA = sum(ni for _, ni in statsA) / max(sum(nf + ni for nf, ni in statsA), 1)
    shareB = sum(ni for _, ni in statsB) / max(sum(nf + ni for nf, ni in statsB), 1)
    # THE AGGREGATE SHARE IS SLOT-WEIGHTED, NOT ROW-WEIGHTED: bucket A and
    # B have different average relation-slots/row (fA != fB), so mixing by
    # ROW fraction x does NOT land on x*shareA + (1-x)*shareB (a first
    # draft of this script assumed it did and landed 5 points short of a
    # 0.55 target). Solving x*nmint rows from A / (1-x)*nmint from B for
    # the SLOT-weighted achieved share = target:
    #   x*fA*(shareA-target) = (1-x)*fB*(target-shareB)  =>  x = R/(1+R)
    if TARGET_INVERSE <= shareB:
        x = 0.0
    elif TARGET_INVERSE >= shareA:
        x = 1.0
    else:
        R = (fB * (TARGET_INVERSE - shareB)) / (fA * (shareA - TARGET_INVERSE))
        x = R / (1.0 + R)
    nA = min(len(bucketA), round(x * n_mint)); nB = min(len(bucketB), n_mint - nA)
    nA = min(len(bucketA), n_mint - nB)
    rng.shuffle(bucketA); rng.shuffle(bucketB)
    new_mint_rows = bucketA[:nA] + bucketB[:nB]
    rng.shuffle(new_mint_rows)
    print(f"[select] bucketA(inv>fwd) n={len(bucketA)} share={shareA:.4f} x={x:.4f} taken={nA}; "
          f"bucketB n={len(bucketB)} share={shareB:.4f} taken={nB}; total taken={len(new_mint_rows)} "
          f"(target n_mint={n_mint})", flush=True)
    dose_law_legs(new_mint_rows, "pm35d mint (AFTER, inverse-dosed, pre-stamp)")

    # three-stage stamp, applied to the REPLACEMENT mint rows only (the prose rows are carried
    # through untouched -- they are already stamped identically to pm35c's own, and every stamp
    # stage below is a per-row deterministic function of a row's own text/factors/solution, so
    # re-stamping the already-stamped prose rows would be a safe no-op were it done, but is
    # skipped here to make "pm35c's prose rows, kept byte-identical" a literal guarantee, not an
    # idempotence claim).
    import stamp_value_spans as SVS
    for r in new_mint_rows:
        r.setdefault("mentions", {})
    tmp_rows = new_mint_rows
    n_numeral = n_lexicon = n_unresolved = 0
    for r in tmp_rows:
        text = r["text"]; mentions = r["mentions"]
        for fac in r["factors"]:
            if fac.get("ftype") != "given":
                continue
            v = fac["var"]; target = abs(int(fac["value"]))
            spans = SVS.numeral_occurrences(text, target)
            if not spans:
                spans = SVS.lexicon_occurrences(text, target)
            if not spans:
                n_unresolved += 1
                continue
            SVS.add_spans(mentions, v, spans)
    print(f"[stamp-1/3] value spans: {len(tmp_rows)} rows, unresolved givens={n_unresolved}", flush=True)

    import stamp_arg_mentions as SAM
    import args_census as AC
    for r in tmp_rows:
        text = r["text"]; factors = r["factors"]; solution = r.get("solution", [])
        mentions = r.get("mentions", {})
        bounds = AC.sentence_bounds(text); sspans = AC.sentence_spans(text, bounds)
        intro_map = SAM.build_intro_map(factors); memo = {}
        # these rows have no "src" key in gen (is_prose=False, stamp_file's own convention) --
        # the letter fallback is built, matching stamp_file's mint-register branch exactly.
        letter_of = SAM.build_letter_map(text)
        for idx, fac in enumerate(factors):
            if "args" in fac:
                arg_spans, cue_spans, _infos = SAM.process_relation(
                    idx, fac, factors, text, bounds, sspans, solution,
                    intro_map, mentions, memo, letter_of)
                fac["arg_spans"] = arg_spans
                fac["cue_spans"] = cue_spans
            else:
                fac["arg_spans"] = []
                fac["cue_spans"] = []
    print(f"[stamp-2/3] arg mentions: {len(tmp_rows)} rows", flush=True)

    import stamp_given_cues as SGC
    for r in tmp_rows:
        SGC.stamp_row(r)
    print(f"[stamp-3/3] role cues: {len(tmp_rows)} rows", flush=True)

    out_rows = prose_rows + tmp_rows
    rng.shuffle(out_rows)
    with open(OUT, "w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")
    print(f"[build] {len(out_rows)} rows -> {OUT} ({len(prose_rows)} prose + {len(tmp_rows)} mint)",
          flush=True)
    print(f"[timing] {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
