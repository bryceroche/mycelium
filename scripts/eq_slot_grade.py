"""eq_slot_grade.py -- THE EQ_SLOT GRADER (ledger 2026-10-27; zero GPU). The order table's task: given the problem text and its
LOCATED given mentions, assign each equation given slot (eq_slot = derivation order) one mention. Graded THROUGH THE SOLVER:
the gold factor graph with each given slot's value replaced by its assigned mention's value is solved (solve_forced, unique
answer required) and compared to the CUSTODY answer key (custody_gold.row_gold; never the solution field for pen rows).
An order that differs from gold but yields the key (commuted args, equal values) scores right -- graph equivalence under the
key, not position. Unstated givens (no text mention) keep their gold value: they are outside the task.
Fixture rows = those whose UNTOUCHED gold graph solves to the key (the oracle); the rest are counted, never silently dropped.
  as a module: rows(split) -> task rows; grade(task, assign) -> bool; baselines printed by __main__ (TEXT / RANDOM / ORACLE)."""
import json, os, random, signal, sqlite3, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ".")
from mycelium.custody_gold import row_gold
from repair_replace_swap import solve_forced
import governor_census as G

DB = ".cache/order_table.sqlite"
GRADE_SECS = int(os.environ.get("EG_SECS", "3"))


class _Late(Exception):
    pass


def _late(*_):
    raise _Late()
MIX = {"train": ".cache/form_mix_pm35c.jsonl", "holdout": G.HOLDOUT}


def rows(split, limit=None):
    """One task per mix row: {split,row,smp,gold,slots:[(eq_slot,var,value,char_start|None)], cands:[char_start sorted]}."""
    c = sqlite3.connect(DB); by = {}
    for r, eq, var, val, cs, loc in c.execute(
            "select row,eq_slot,var,value,char_start,locate from order_map where split=? order by row,eq_slot", (split,)):
        by.setdefault(r, []).append((eq, var, val, None if loc == "unstated" else cs))
    out, n_nogold = [], 0
    for i, line in enumerate(open(MIX[split])):
        if limit is not None and len(out) >= limit: break
        if i not in by: continue
        smp = json.loads(line)
        try: gold = row_gold(smp)
        except (KeyError, ValueError, IndexError): n_nogold += 1; continue   # pen text not in the harvest (resampled) / short mint vector
        sl = by[i]; out.append({"split": split, "row": i, "smp": smp, "gold": gold, "slots": sl,
                                "cands": sorted({s[3] for s in sl if s[3] is not None})})
    return out, n_nogold


def grade(t, assign):
    """assign: {eq_slot: char_start of the chosen mention} for every stated slot. True iff the solver returns the key."""
    val_at = {s[3]: s[2] for s in t["slots"] if s[3] is not None}
    newv = {s[1]: (val_at[assign[s[0]]] if s[3] is not None else s[2]) for s in t["slots"]}
    facs = [dict(f, value=newv[f["var"]]) if f["ftype"] == "given" and f["var"] in newv else f for f in t["smp"]["factors"]]
    signal.signal(signal.SIGALRM, _late); signal.alarm(GRADE_SECS)   # a wrong order can send the search wide (109 s seen):
    try: return solve_forced(facs, t["smp"]["query_var"], t["smp"]) == t["gold"]   # the time budget REJECTS, as at mint
    except _Late: return False
    finally: signal.alarm(0)


def text_assign(t):   # the identity reading: the k-th stated slot takes the k-th mention in text order
    st = [s for s in t["slots"] if s[3] is not None]
    return {s[0]: c for s, c in zip(st, t["cands"])}


def gold_assign(t):
    return {s[0]: s[3] for s in t["slots"] if s[3] is not None}


if __name__ == "__main__":
    rng = random.Random(0)
    for split, lim in (("holdout", None), ("train", int(os.environ.get("EG_TRAIN_N", "1000")))):
        T, nog = rows(split, lim)
        ok = [t for t in T if grade(t, gold_assign(t))]
        txt = sum(grade(t, text_assign(t)) for t in ok)
        ident = sum(text_assign(t) == gold_assign(t) for t in ok)
        rnd = 0
        for t in ok:
            st = [s for s in t["slots"] if s[3] is not None]; cs = list(t["cands"]); rng.shuffle(cs)
            rnd += grade(t, {s[0]: c for s, c in zip(st, cs)})
        n = len(ok)
        print(f"{split}: rows {len(T)} (+{nog} without custody gold) | ORACLE solves to key {n}/{len(T)} = {n / max(len(T), 1):.3f} "
              f"| on the fixture: TEXT solver {txt / n:.3f} (exact order {ident / n:.3f}) | RANDOM solver {rnd / n:.3f}", flush=True)
