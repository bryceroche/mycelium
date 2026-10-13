"""edge_read1_goldfree_flag.py -- THE GOLD-FREE DISAGREEMENT FLAG (coordinator's scope extension,
2026-10-12, on top of edge_read1_transfer.py's READ 1). Zero-GPU, zero-model: reads the per-relation
entries edge_read1_transfer.py dumps (.cache/edge_read1_entries_PMS8_241.pkl) -- no forward pass, no
replay, no solver.

THE FLAG (gold-free: the decision to raise it never consults the custody key or score_row_slots'
f_args; only the transfer graph's own ranking and the pointer's own decoded args/dup heads):
  top-1 flag  : raised when the transfer graph's top-1 responder (argmax_j R_sum(i->j)) is NOT one
                of the pointer's own predicted args (pred_args_set: dup-predicted single arg, else
                the model's own unrestricted top-2 of args[j] -- the SAME rule score_row_slots/
                decode use for correctness, applied gold-free here to the model's own output).
  top-2 flag  : raised when NEITHER of the pointer's predicted args appears in the transfer graph's
                top-2 responders.
EVALUATION (gold IS used here, for scoring the flag against truth -- never for raising it):
  recall    = P(flagged | pointer actually wrong)      -- among the 64-row entries where
              score_row_slots' f_args is False (read 1's gold-conditioned "pointer-wrong" set)
  precision = P(pointer actually wrong | flagged)
  coverage  = P(flagged) over ALL relation decisions (not just pointer-wrong)
  proposal  = for the top-1 flag only, among FLAGGED entries, how often the transfer top-1 (the
              value the flag would propose as a replacement) is itself a gold arg (hit1) --
              "how good is the substitute when the flag fires".
BAR (coordinator's pin): recall > 0.19 (READ C's unsat-core solver flagger's own recall) at
precision >= 0.5.

CAVEAT (repeated per the coordinator's instruction): `cands`, the candidate pool the transfer graph
ranks over, is read 1's own present-slot pool, itself derived from the fixture's GOLD factor list
(vs[i]["factors"]) rather than the model's own predicted presence -- a pre-existing choice in READ 1,
unchanged here. The FLAG's decision rule itself (which responder is top-1/top-2, whether it matches
pred_args_set) uses no gold; recall/precision/coverage below ARE the gold-free-deployable numbers
the coordinator asked for, but the slot pool they are computed over is not perfectly gold-free as a
pipeline. Also: READ 1's headline precision@1/@2 table (pointer-wrong vs pointer-right) is itself
GOLD-CONDITIONED (it only ever evaluates the subset already known, via gold, to be wrong) -- the
numbers in THIS file are the deployable ones (no gold needed to decide which relations to inspect).
"""
import os
import sys
import json
import pickle

sys.path.insert(0, "."); sys.path.insert(0, "scripts"); sys.path.insert(0, "scripts/picker")
import numpy as np

ENTRIES_PKL = ".cache/edge_read1_entries_PMS8_241.pkl"
OUT_JSON = ".cache/edge_read1_goldfree_flag_PMS8_241.json"
OUT_TXT = ".cache/edge_read1_goldfree_flag_PMS8_241.txt"


def main():
    d = pickle.load(open(ENTRIES_PKL, "rb"))
    entries = d["entries"]
    n = len(entries)
    for e in entries:
        pas = set(e["pred_args_set"])
        e["flag1"] = bool(e["top1"] not in pas)
        e["flag2"] = bool(len(set(e["top2"]) & pas) == 0)

    def table(flag_key):
        wrong = [e for e in entries if e["pointer_wrong"]]
        flagged = [e for e in entries if e[flag_key]]
        flagged_wrong = [e for e in flagged if e["pointer_wrong"]]
        recall = len(flagged_wrong) / max(len(wrong), 1)
        precision = len(flagged_wrong) / max(len(flagged), 1)
        coverage = len(flagged) / max(n, 1)
        return dict(n=n, n_pointer_wrong=len(wrong), n_flagged=len(flagged),
                   n_flagged_and_wrong=len(flagged_wrong),
                   recall=recall, precision=precision, coverage=coverage)

    t1 = table("flag1")
    t2 = table("flag2")
    flagged1 = [e for e in entries if e["flag1"]]
    proposal_hit1 = float(np.mean([e["hit1"] for e in flagged1])) if flagged1 else float("nan")

    bar_hit1 = t1["recall"] > 0.19 and t1["precision"] >= 0.5
    bar_hit2 = t2["recall"] > 0.19 and t2["precision"] >= 0.5

    report = dict(tag="PMS8_241", n_rows=d.get("n_rows"), n_relation_decisions=n,
                  top1_flag=t1, top2_flag=t2, proposal_correctness_top1=proposal_hit1,
                  bar=dict(recall_gt=0.19, precision_ge=0.5),
                  bar_met=dict(top1=bar_hit1, top2=bar_hit2),
                  caveat="cands (the candidate pool) is gold-presence-derived (read 1's own design); "
                          "the flag's decision rule itself is gold-free")
    json.dump(report, open(OUT_JSON, "w"), indent=1)

    lines = []
    L = lines.append
    L("THE GOLD-FREE DISAGREEMENT FLAG (PMS8_241, wildhold, %d relation decisions across %d rows)"
      % (n, d.get("n_rows", 0)))
    L("")
    L("CAVEAT: the candidate pool (cands) is gold-presence-derived (inherited from READ 1); the flag's")
    L("OWN decision rule (top1/top2 vs pred_args_set) consults no gold. recall/precision/coverage below")
    L("are the deployable, gold-free numbers; READ 1's own precision@1/@2 table is gold-CONDITIONED.")
    L("")
    for name, t in (("TOP-1 FLAG (transfer top-1 not in pointer's own predicted args)", t1),
                   ("TOP-2 FLAG (neither pointer arg in transfer top-2)", t2)):
        L(f"-- {name} --")
        L(f"  n_pointer_wrong={t['n_pointer_wrong']}  n_flagged={t['n_flagged']}  n_flagged_and_wrong={t['n_flagged_and_wrong']}")
        L(f"  RECALL={t['recall']:.3f}  PRECISION={t['precision']:.3f}  COVERAGE={t['coverage']:.3f}")
        L("")
    L(f"PROPOSAL CORRECTNESS (top-1 flag only): among flagged relations, the transfer top-1's own")
    L(f"  gold-arg hit rate (what the flag would substitute) = {proposal_hit1:.3f}")
    L("")
    L(f"VERDICT vs BAR (recall > 0.19 at precision >= 0.5): top-1 flag {'MET' if bar_hit1 else 'NOT MET'}; "
      f"top-2 flag {'MET' if bar_hit2 else 'NOT MET'}.")
    txt = "\n".join(lines)
    open(OUT_TXT, "w").write(txt + "\n")
    print(txt)
    print(f"\n[edge-read1-goldfree] wrote {OUT_JSON} and {OUT_TXT}", flush=True)


if __name__ == "__main__":
    main()
