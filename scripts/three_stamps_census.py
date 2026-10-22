"""three_stamps_census.py -- THE THREE-STAMPS CENSUS (zero GPU; gold only).
Part 1 reuses scripts/picker/planning_drop_read.py census A's classification VERBATIM (GIVEN/PRODUCED/SEEN-ARG/FIRST-HERE)
and must reproduce 0.526/0.218/0.076/0.180.  Part 2 measures the 'promise slot' hypothesis (complex_planes 11.4 item 5)
on the third-kind lookups (SEEN-ARG + FIRST-HERE).
DEFINITIONS: step = factor's index in row['factors'] (gold order, givens count). rstep = index among relation factors only.
 introduction step of var v = the first factor index at which v appears as an arg or result of a relation factor
   (for FIRST-HERE that is this very factor; for SEEN-ARG an earlier relation).  (a) is therefore 100% by construction
   except where the var is a GIVEN-var later (cannot happen: classified third-kind only if not given before).
 gap = lookup step - intro step (0 for FIRST-HERE).
 promised var = a var that is the target of >=1 third-kind lookup (row-level distinct).
 produced later = v is the result of some relation at factor index > its introduction step.
 produced after lookup = ... at factor index > the lookup step (per lookup).
Out: .cache/three_stamps_census.txt (refuses to overwrite)."""
import collections, json, os, sys
OUTP = ".cache/three_stamps_census.txt"
if os.path.exists(OUTP):
    sys.exit("refusing to overwrite " + OUTP)
OUT = []
def say(s=""):
    print(s); OUT.append(s)
rows = [json.loads(l) for l in open(".cache/wild_admitted_holdout.jsonl")]
C = collections.Counter(); rows_third = 0
gaps = collections.Counter(); rgaps = collections.Counter(); gaps_seen = collections.Counter()
per_row_promised = collections.Counter(); ex = []
L = collections.Counter(); P = collections.Counter()
for ri, row in enumerate(rows):
    facs = row["factors"]; qv = row.get("query_var")
    given, produced, seen_arg = set(), set(), set(); third = False
    intro = {}; rintro = {}; rk = -1
    res_idx = collections.defaultdict(list)
    for i, f in enumerate(facs):
        if f["ftype"] != "given" and f.get("result") is not None:
            res_idx[f["result"]].append(i)
    given_idx = collections.defaultdict(list)
    for i, f in enumerate(facs):
        if f["ftype"] == "given": given_idx[f["var"]].append(i)
    promised = {}
    for i, f in enumerate(facs):
        if f["ftype"] == "given":
            given.add(f["var"]); continue
        rk += 1
        args = list(f.get("args", [])); res = f.get("result")
        C["rel"] += 1
        for a in args:   # classification: verbatim from planning_drop_read.census
            kind = ("GIVEN" if a in given else "PRODUCED" if a in produced else
                    "SEEN-ARG" if a in seen_arg else "FIRST-HERE")
            C[kind] += 1; C["arg"] += 1
            third |= kind in ("SEEN-ARG", "FIRST-HERE")
            if a not in intro: intro[a] = i; rintro[a] = rk
            if kind in ("SEEN-ARG", "FIRST-HERE"):
                L["third"] += 1
                ok = a in intro and intro[a] <= i
                L["intro_identifiable"] += ok
                g = i - intro[a]; gaps[g] += 1; rgaps[rk - rintro[a]] += 1
                (gaps_seen if kind == "SEEN-ARG" else L).__class__  # no-op
                L[kind] += 1
                if kind == "SEEN-ARG": gaps_seen[g] += 1
                promised.setdefault(a, []).append((i, kind))
                L["prod_after_lookup"] += any(j > i for j in res_idx[a])
                L["given_after_lookup"] += any(j > i for j in given_idx[a])
                L["lookup_is_query"] += (a == qv)
        if res is not None:
            C["res_" + ("CLOSES" if (res in given or res in produced or res in seen_arg) else "NEW")] += 1
            produced.add(res)
            if res not in intro: intro[res] = i; rintro[res] = rk
        seen_arg |= {a for a in args if a not in given and a not in produced}
    rows_third += third
    per_row_promised[len(promised)] += 1
    for v, lk in promised.items():
        P["vars"] += 1
        P["prod_later"] += any(j > intro[v] for j in res_idx[v])
        P["never_produced"] += not res_idx[v]
        P["given_later"] += any(j > intro[v] for j in given_idx[v])
        P["is_query"] += (v == qv)
        P["multi_lookup"] += len(lk) > 1
        P["only_first_here"] += all(k == "FIRST-HERE" for _, k in lk)
        P["with_seen_arg"] += any(k == "SEEN-ARG" for _, k in lk)
        if len(ex) < 400 and any(k == "SEEN-ARG" for _, k in lk) and v != qv and res_idx[v]:
            ex.append((ri, v, intro[v], lk))
n = C["arg"]
say("== 1. REPRODUCTION (planning_drop_read census A classification, verbatim)")
say(f"  {C['rel']} relations, {n} lookups, {len(rows)} rows")
for k in ("GIVEN", "PRODUCED", "SEEN-ARG", "FIRST-HERE"):
    say(f"    {k:11s} {C[k]:5d}  {C[k]/n:.3f}")
say(f"  THIRD KIND {(C['SEEN-ARG']+C['FIRST-HERE'])/n:.3f}; rows needing it {rows_third}/{len(rows)} = {rows_third/len(rows):.3f}")
t = L["third"]
say(f"\n== 2. THIRD-KIND LOOKUPS (n={t}: SEEN-ARG {L['SEEN-ARG']}, FIRST-HERE {L['FIRST-HERE']})")
say(f"(a) introduction step identifiable in gold: {L['intro_identifiable']}/{t} = {L['intro_identifiable']/t:.3f}")
say("(b) gap = lookup step - intro step (factor index, givens counted):")
for g in sorted(gaps): say(f"    gap {g:2d}: {gaps[g]:4d}  {gaps[g]/t:.3f}")
sa = L["SEEN-ARG"]
say(f"  SEEN-ARG only (n={sa}), factor-index gap:")
for g in sorted(gaps_seen): say(f"    gap {g:2d}: {gaps_seen[g]:4d}  {gaps_seen[g]/sa:.3f}")
mean_sa = sum(g*c for g, c in gaps_seen.items())/sa
say(f"  mean gap SEEN-ARG {mean_sa:.2f}; all third-kind {sum(g*c for g,c in gaps.items())/t:.2f}")
say("  gap in RELATION-index units (relations only):")
for g in sorted(rgaps): say(f"    rgap {g:2d}: {rgaps[g]:4d}  {rgaps[g]/t:.3f}")
nr = len(rows); tot = P["vars"]
say(f"(c) distinct promised unknowns per row (rows={nr}; total promised vars {tot}; mean {tot/nr:.2f}):")
for k in sorted(per_row_promised): say(f"    {k} promised: {per_row_promised[k]} rows  {per_row_promised[k]/nr:.3f}")
say(f"    lookups per promised var: {t/tot:.2f}; promised vars with >1 third-kind lookup {P['multi_lookup']} ({P['multi_lookup']/tot:.3f}); "
    f"only-FIRST-HERE vars {P['only_first_here']} ({P['only_first_here']/tot:.3f}); with a SEEN-ARG lookup {P['with_seen_arg']} ({P['with_seen_arg']/tot:.3f})")
say(f"(d) per promised var (n={tot}): produced as a relation result later than its introduction {P['prod_later']} ({P['prod_later']/tot:.3f}); "
    f"never produced {P['never_produced']} ({P['never_produced']/tot:.3f}); (re)given later {P['given_later']} ({P['given_later']/tot:.3f}); "
    f"is query_var {P['is_query']} ({P['is_query']/tot:.3f})")
say(f"    per lookup (n={t}): target produced after the lookup step {L['prod_after_lookup']} ({L['prod_after_lookup']/t:.3f}); "
    f"target is query_var {L['lookup_is_query']} ({L['lookup_is_query']/t:.3f})")
say("\n== 3. EXAMPLES (SEEN-ARG promised var, not query, later produced): row idx, var, intro step, lookups (step, kind)")
for ri, v, i0, lk in ex[:12]:
    say(f"  row {ri} var {v} intro {i0} lookups {lk}: {rows[ri]['text'][:140]}")
    say("     factors: " + " ; ".join(f"{k}:{f['ftype']}" + (f" v{f['var']}={f['value']}" if f['ftype']=='given' else f" {f['op']}{f['args']}->{f['result']}") for k, f in enumerate(rows[ri]['factors'])) + f" ; query {rows[ri].get('query_var')}")
open(OUTP, "w").write("\n".join(OUT) + "\n")
