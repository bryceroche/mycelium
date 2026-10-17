"""photobooth_vote_read.py -- THE NAME-SWAP VOTE READ (2026-10-16; bars pinned in the ledger 19:20 before measurement).
Input: .cache/dump_wildPB{0..4}_TAG.pkl (0 = the original re-read in the same chain, 1..4 = the name-swap views) +
.cache/wild_PB_v{1..4}.jsonl (which rows changed). Per the pinned frame:
  DETERMINISM GATE  unchanged rows: every view's predicted graph = view 0's (must be 1.000, else VOID)
  INVARIANCE        changed rows: predicted graph identical to view 0's
  FAC-EXACT         positional per-slot, view k vs view 0 on the changed rows (paired)
  WHOLE-PROBLEM     the predicted graph solves and the gold query slot carries the key: view 0 / each view / the 5-read
                    vote (majority >= 3 of the 5 forced answers)
usage: photobooth_vote_read.py TAG"""
import collections, json, pickle, sys
sys.path.insert(0, "."); sys.path.insert(0, "scripts"); sys.path.insert(0, "scripts/picker")
from step_blame_census import pred_graph, solve_all  # noqa: E402
import matched_read as M  # noqa: E402

tag = sys.argv[1]; K = 4
rows = [json.loads(l) for l in open(".cache/wild_admitted_holdout.jsonl")]
changed = [set(i for i, l in enumerate(open(f".cache/wild_PB_v{v}.jsonl")) if json.loads(l)["gen"]["photobooth"]["names"]) for v in range(1, K + 1)]
D = []
for v in range(K + 1):
    by = collections.defaultdict(list)
    for t in pickle.load(open(f".cache/dump_wildPB{v}_{tag}.pkl", "rb")): by[t[0]].append(t[1:])
    D.append({i: sorted(s, key=lambda s: s[0]) for i, s in by.items()})
sig = lambda slots: tuple((s[6], s[7], tuple(sorted(s[8])), s[9], tuple(s[10]), s[11], s[12]) for s in slots if s[11])
pos = lambda slots: M.positional(slots)


_MEMO = {}


def answer(i, slots):
    k = (i, sig(slots))
    if k not in _MEMO:
        _MEMO[k] = _answer(i, slots)
    return _MEMO[k]


def _answer(i, slots):
    facs, bad = pred_graph(slots)
    if bad: return None
    asg = solve_all(24, facs, int(rows[i].get("m", 300)))
    q = rows[i]["query_var"]
    return int(asg[q]) if asg is not None and q < len(asg) else None


key = [int(r["solution"][r["query_var"]]) if r.get("solution") else None for r in rows]
C = collections.Counter(); ans = {}
for i in range(len(rows)):
    ans[i] = [answer(i, D[v][i]) for v in range(K + 1)]
for v in range(1, K + 1):
    for i in range(len(rows)):
        same = sig(D[v][i]) == sig(D[0][i]); n = len(D[0][i])
        if i in changed[v - 1]:
            C["ch"] += 1; C["inv"] += same; C["slots"] += n; C["fx0"] += pos(D[0][i]); C["fxv"] += pos(D[v][i])
            C["wp0"] += ans[i][0] == key[i] and key[i] is not None; C["wpv"] += ans[i][v] == key[i] and key[i] is not None
        else:
            C["un"] += 1; C["det"] += same
chg_any = set().union(*changed)
for i in chg_any:
    a = [x for x in ans[i] if x is not None]; top = collections.Counter(a).most_common(1)
    vote = top[0][0] if top and top[0][1] >= 3 else None
    C["vrows"] += 1; C["v_orig"] += ans[i][0] == key[i]; C["v_vote"] += vote is not None and vote == key[i]; C["v_abst"] += vote is None
det = C["det"] / max(C["un"], 1); inv = C["inv"] / max(C["ch"], 1); dfx = (C["fxv"] - C["fx0"]) / max(C["slots"], 1)
print(f"== {tag} NAME-SWAP VOTE READ")
print(f"  DETERMINISM GATE (unchanged rows x views, n {C['un']}): {det:.4f} -> {'PASS' if det == 1.0 else 'VOID'}")
print(f"  INVARIANCE (changed rows x views, n {C['ch']}): {inv:.3f}")
print(f"  FAC-EXACT on changed rows: original {C['fx0'] / max(C['slots'], 1):.4f} | views {C['fxv'] / max(C['slots'], 1):.4f} | change {dfx:+.4f} ({C['slots']} slot-reads)")
print(f"  WHOLE-PROBLEM on changed rows: original {C['wp0']}/{C['ch']} per view-pair | views {C['wpv']}/{C['ch']} | 5-read vote on {C['vrows']} rows: "
      f"original {C['v_orig']} right, vote {C['v_vote']} right ({C['v_abst']} abstain)")
verdict = ("NAMES NOT MEMORIZED" if inv >= 0.85 and dfx >= -0.010 else "NAMES MEMORIZED" if inv < 0.70 or dfx <= -0.020 else "INCONCLUSIVE")
print(f"  VERDICT (pinned frame): {verdict if det == 1.0 else 'VOID (determinism gate)'}")
