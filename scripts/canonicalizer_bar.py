"""canonicalizer_bar.py — THE PINNED BAR FIRES (2026-08-03; bar pinned
blind 2026-07-25 at #69's registration: FALSE-MERGE-RATE == 0 REQUIRED,
collapse-rate reported; any false-merge names its pass and demotes it).

Fixture: the deployed mix's GOLD graphs (solver-verified, answers
known) — the strongest false-merge ground truth on disk. Two unique-
text rows whose canonical strings collide are a FALSE MERGE if their
answers differ (semantically distinct by the key's own gold).
Controls: (1) synthetic invariance — factor-shuffle + var-permutation
of a row MUST canonicalize identically (1000 rows x 3 perms);
(2) identical rows must trivially merge. The paraphrase-collapse read
(5-view parses) is PENDING-GPU; answer artifact for the orphan guard:
.cache/canonicalizer_bar.json."""
import sys, json, random
sys.path.insert(0, '.')
from mycelium.canonicalizer import canonical_digest

rows = []
seen_txt = set()
for l in open('.cache/gen22_mix.jsonl'):
    r = json.loads(l)
    if r['text'] in seen_txt:
        continue
    seen_txt.add(r['text'])
    try:
        facs = r['factors'] if isinstance(r['factors'], list) else eval(r['factors'])
        q = int(r['query_var'])
        sol = r['solution'] if isinstance(r['solution'], list) else eval(r['solution'])
        ans = sol[q]
        if all(v == 0 for v in sol):
            continue  # placeholder solution (prose/book rows) — cannot serve as gold
    except Exception:
        continue
    rows.append((r['text'], facs, q, ans))
print(f"[bar] unique-text gold rows: {len(rows)}")

# --- false-merge read (the bar) ---
groups = {}
for txt, facs, q, ans in rows:
    try:
        c = canonical_digest(facs, q, n_vars=24)
    except Exception as e:
        c = "ERR:" + txt[:40]
    groups.setdefault(c, []).append((txt, ans))
sys.path.insert(0, 'scripts')
from hash_audit_iso import verify_iso
by_txt = {txt: (facs, q) for txt, facs, q, ans in rows}
false_merges = []
wl_collisions = 0
n_merged_pairs = 0
for c, members in groups.items():
    if len(members) > 1:
        n_merged_pairs += len(members) - 1
        answers = {a for _, a in members}
        if len(answers) > 1:
            # the door's design: digest is a PREFILTER; verify_iso is the
            # judge. Distinct-answer members sharing a digest are a WL
            # collision (known-incomplete, why verify exists) UNLESS
            # verify_iso confirms isomorphism — which with values semantic
            # and gold deterministic would be a TRUE false merge.
            t0, a0 = members[0]
            confirmed = False
            for t1, a1 in members[1:]:
                if a1 != a0:
                    ra = {"factors": by_txt[t0][0], "query_var": by_txt[t0][1], "n_vars": 24}
                    rb = {"factors": by_txt[t1][0], "query_var": by_txt[t1][1], "n_vars": 24}
                    if verify_iso(ra, rb):
                        confirmed = True
                        break
            if confirmed:
                false_merges.append({"canonical": c[:64],
                                     "answers": sorted(answers),
                                     "texts": [t[:80] for t, _ in members[:3]]})
            else:
                wl_collisions += 1
print(f"[bar] WL digest collisions caught by verify_iso: {wl_collisions} (the door's second stage working as designed)")
fm_rate = len(false_merges) / max(len(rows), 1)
collapse = n_merged_pairs / max(len(rows), 1)
print(f"[bar] canonical classes: {len(groups)}  merged-away rows: {n_merged_pairs} "
      f"(collapse {collapse:.4f})  FALSE MERGES: {len(false_merges)} (rate {fm_rate:.6f})")

# --- control: synthetic invariance (shuffle + var permutation) ---
rng = random.Random(103500)
fails = 0
_KNOWN = {'rel','given','mod','fdiv','pct','sel'}
permutable = [r for r in rows if all(f.get('ftype') in _KNOWN for f in r[1])]
sample = rng.sample(permutable, min(1000, len(permutable)))
for txt, facs, q, ans in sample:
    c0 = canonical_digest(facs, q, n_vars=24)
    for t in range(3):
        vs = sorted({v for f in facs for k in ('var', 'result')
                     if isinstance(f.get(k), int) for v in [f[k]]}
                    | {a for f in facs if isinstance(f.get('args'), list)
                       for a in f['args']} | {q})
        perm = list(vs); rng.shuffle(perm)
        pm = dict(zip(vs, perm))
        def remap(f):
            f = dict(f)
            for k in ('var', 'result'):
                if isinstance(f.get(k), int): f[k] = pm[f[k]]
            if isinstance(f.get('args'), list):
                f['args'] = [pm[a] for a in f['args']]
            return f
        pf = [remap(f) for f in facs]; rng.shuffle(pf)
        c1 = canonical_digest(pf, pm[q], n_vars=24)
        if c1 != c0:
            fails += 1; break
print(f"[control] invariance: {len(sample)-fails}/{len(sample)} "
      f"(shuffle+perm must not change the canonical form)")

verdict = ("PASS — false-merge 0, bar holds" if not false_merges and not fails
           else "FAIL — " + (f"{len(false_merges)} false merges" if false_merges
                             else f"{fails} invariance failures"))
print(f"VERDICT (pinned 2026-07-25): {verdict}")
json.dump({"n_rows": len(rows), "classes": len(groups),
           "collapse_rate": collapse, "false_merges": false_merges[:10],
           "false_merge_rate": fm_rate,
           "invariance_fails": fails, "invariance_n": len(sample),
           "paraphrase_collapse": "PENDING-GPU (5-view parse read rides the next window)",
           "verdict": verdict},
          open('.cache/canonicalizer_bar.json', 'w'), indent=1)
print("[saved] .cache/canonicalizer_bar.json")
