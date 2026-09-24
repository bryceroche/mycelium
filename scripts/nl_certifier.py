"""nl_certifier.py -- THE NL CERTIFIER, v1 (2026-09-24, word given; THE ALTERNATION SPEC §6 and the
09-24 brainstorm). The solver is the math certifier: it accepts a graph or refuses it. This is the text's
certifier: parameter-free reads of a DECODED graph against its problem text, per slot and per row —
evidence, never pass/fail, never a loss (the Goodhart fence). Zero GPU: it reads a chain_acc CA_RAWDUMP
(every slot's raw heads per row + the text + the key), decodes with the numeral mask exactly as chain_acc
does, runs the solver host-side for the row's label (correct / refused / wrong), and benches every
certificate against those labels (AUROC correct-vs-wrong among solved rows; solved-vs-refused).

Certificates (all in [0, 1], higher = more corroborated):
  value_present   fraction of decoded givens whose value appears as a numeral in the text
  coverage        fraction of the text's numerals (<= 999) claimed by some decoded given
  no_double       1 - (fraction of decoded givens sharing a numeral with another given)
  cue_agree       fraction of decoded relations whose resolved clause carries a cue word consistent
                  with the decoded op (add: total/sum/altogether/combined/more/plus/gained/left/...;
                  mul: times/twice/double/triple/each/per/product/half) — clause_of from the stamper
  chain_reaches   1 if the decoded graph's derived chain reaches the query variable
  derived_stated  fraction of decoded relations whose IMPLIED value (from the solver's assignment when
                  solved) appears as a numeral in the text — a stated total corroborates its chain
  cert            the mean of the above (a first combined score; the bench says which carry)
usage: NLC_DUMP=.cache/rawslots_wild_PMS8_241.pkl NLC_OUT=.cache/nl_certifier_PMS8_241.txt
       [NLC_ROWS=.cache/wild_admitted_holdout.jsonl] .venv/bin/python3 scripts/nl_certifier.py
"""
import os
import re
import sys
import json
import pickle

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
import numpy as np

DUMP = os.environ.get("NLC_DUMP", ".cache/rawslots_wild_PMS8_241.pkl")
OUT = os.environ.get("NLC_OUT", ".cache/nl_certifier_" + os.path.basename(DUMP).replace("rawslots_wild_", "").replace(".pkl", "") + ".txt")
ROWS = os.environ.get("NLC_ROWS", ".cache/wild_admitted_holdout.jsonl")

ADD_CUES = set("total sum altogether combined together more plus gained added increase increased extra additional left remaining fewer less minus lost spent subtract subtracted decrease decreased difference".split())
MUL_CUES = set("times twice double triple each per product multiply multiplied half quarter third rate every".split())


def _solve_row(rec, parse, q):
    """chain_acc's _solve_task, in-process (the row's label)."""
    from admit_annotation import solve_walled
    from alternator_bridge import problem_from_algebra3
    gv = {f["var"]: f["value"] for f in parse if f["ftype"] == "given"}
    used = [f.get("var") for f in parse if f["ftype"] == "given"] + [a for f in parse if f["ftype"] == "rel" for a in list(f["args"]) + [f["result"]]]
    nvv = max([q + 1] + [v + 1 for v in used if v is not None])
    key = rec.get("key")
    if key is None or not parse:
        return "refused", None, None
    gmax = max([int(v) for v in gv.values()] + [1]); m0 = int(min(10001, max(300, 2 * gmax, 2 * int(key))))
    for m in ([m0] + ([10000] if m0 < 10000 else [])):
        try:
            res = solve_walled(problem_from_algebra3(nvv, parse, gv, m), budget=5000, wall=3)
        except Exception:
            return "refused", None, None
        if res.get("status") == "solved":
            asg = res["assignment"]
            return ("correct" if int(asg[q]) == int(key) else "wrong"), int(asg[q]), asg
    return "refused", None, None


def auroc(pos, neg):
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # rank-based, ties at half
    gt = (pos[:, None] > neg[None, :]).mean(); eq = (pos[:, None] == neg[None, :]).mean()
    return float(gt + 0.5 * eq)


def main():
    from phase1_algebra_head import _decode_slots
    from mycelium.rulebook import legal_digit_logits
    import stamp_arg_mentions as SAM
    import args_census as AC
    recs = pickle.load(open(DUMP, "rb"))
    rows = [json.loads(l) for l in open(ROWS)]
    from phase1_algebra_head import N_DIG
    assert N_DIG == np.asarray(recs[0]["dig"]).shape[1], (
        f"the head's N_DIG={N_DIG} but the dump's digits are {np.asarray(recs[0]['dig']).shape[1]} wide — run under the "
        f"FAMILY ENV (ALG_WIDE=1 sets N_DIG=7); the 09-23 facts-census trap")
    lines = []

    def P(s=""):
        print(s); lines.append(s)

    P("=" * 78); P(f"THE NL CERTIFIER v1 — {DUMP} ({len(recs)} rows)"); P("=" * 78)
    per = []
    for rec in recs:
        text = rec["text"]; row = {k: np.asarray(rec[k]).copy() for k in ("pres", "ftype", "op", "dig", "args", "res") + (("dup",) if "dup" in rec else ())}
        for j in range(row["ftype"].shape[0]):   # THE NUMERAL MASK, as chain_acc applies it
            if int(row["ftype"][j].argmax()) == 0:
                continue
            fake = legal_digit_logits(row["dig"][j], text)
            if fake is not None:
                row["dig"][j] = fake
        parse = _decode_slots(row); q = int(np.asarray(rec["q"]).argmax())
        label, val, asg = _solve_row(rec, parse, q)
        givens = [f for f in parse if f["ftype"] == "given"]; rels = [f for f in parse if f["ftype"] == "rel"]
        nums = [int(x) for x in re.findall(r"(?<![\d.])\d{1,3}(?![\d])", text)]
        numset = set(nums)
        gvals = [int(f["value"]) for f in givens if f.get("value") is not None]
        value_present = (np.mean([v in numset for v in gvals]) if gvals else 0.0)
        coverage = (np.mean([n in set(gvals) for n in nums]) if nums else 1.0)
        no_double = (1.0 - (len(gvals) - len(set(gvals))) / len(gvals)) if gvals else 1.0
        # cue agreement per relation, on the stamper's resolved clause
        bounds = AC.sentence_bounds(text); sspans = AC.sentence_spans(text, bounds)
        intro = SAM.build_intro_map(parse); memo = {}
        sol = list(asg) if asg is not None else []
        agree = []
        for j, f in enumerate(parse):
            if f["ftype"] != "rel":
                continue
            cl = SAM.clause_of(j, parse, text, bounds, sspans, sol, intro, memo)
            stems = SAM.content_stem_set(text, SAM.windows_of(cl, sspans)) if cl is not None else set()
            words = set(w.lower() for w in re.findall(r"[a-z']+", " ".join(text[a:b] for a, b in SAM.windows_of(cl, sspans)))) if cl is not None else set()
            cues = ADD_CUES if f.get("op") == "add" else MUL_CUES
            agree.append(1.0 if (words & cues) else 0.0)
        cue_agree = float(np.mean(agree)) if agree else 1.0
        # the derived chain reaches the query
        reach = {f["result"] for f in rels} | {f["var"] for f in givens}
        chain_reaches = 1.0 if q in reach else 0.0
        # derived values stated in the text (needs the solver's assignment)
        if asg is not None and rels:
            derived_stated = float(np.mean([int(asg[f["result"]]) in numset for f in rels if f["result"] < len(asg)]))
        else:
            derived_stated = 0.0
        certs = dict(value_present=value_present, coverage=coverage, no_double=no_double, cue_agree=cue_agree,
                     chain_reaches=chain_reaches, derived_stated=derived_stated)
        certs["cert"] = float(np.mean(list(certs.values())))
        per.append((rec["i"], label, certs, len(givens), len(rels)))
    labels = [p[1] for p in per]
    P(f"labels: correct {labels.count('correct')} | wrong {labels.count('wrong')} | refused {labels.count('refused')}")
    P("")
    P(f"  {'certificate':16s} {'mean(correct)':>14s} {'mean(wrong)':>12s} {'mean(refused)':>14s} {'AUROC c-vs-w':>13s} {'AUROC solved-vs-ref':>20s}")
    for name in ("value_present", "coverage", "no_double", "cue_agree", "chain_reaches", "derived_stated", "cert"):
        c = [p[2][name] for p in per if p[1] == "correct"]; w = [p[2][name] for p in per if p[1] == "wrong"]; r = [p[2][name] for p in per if p[1] == "refused"]
        P(f"  {name:16s} {np.mean(c) if c else float('nan'):14.3f} {np.mean(w) if w else float('nan'):12.3f} {np.mean(r) if r else float('nan'):14.3f} {auroc(c, w):13.3f} {auroc(c + w, r):20.3f}")
    P("")
    P("READING: a certificate carries if its AUROC correct-vs-wrong clears 0.6 with n this small (correct rows are a")
    P("dozen); the combined score is a first form — the perceiver's input and the accept/refuse evidence, never a loss.")
    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")
    pickle.dump(per, open(OUT.replace(".txt", ".pkl"), "wb"))
    print(f"[nl-certifier] wrote {OUT} (+ .pkl per row)")


if __name__ == "__main__":
    main()
