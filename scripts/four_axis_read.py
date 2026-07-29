"""four_axis_read.py — THE FOUR-AXIS EXPERIMENT (registered 2026-07-29).
Fixtures first; full crossing; sub-anatomy readout; verdict per the
pinned rule. Apparatus: the corrected transient-window reader with
per-probe slot resolution (both run-1 faults structurally prevented).

Cells (12 seeded variants each; all solver-verified; letters
consecutive; values <=300). tgt_ord = the dup sentence's ordinal
(counting the preamble as 0) = its factor slot under the deployed
sent-indexed slot convention (verified by the fixtures themselves:
F-HOLD/F-SLIP majorities are the slot-sanity check too).

  F-HOLD  fixture: "a plus a equals b. c is <3A>. b plus a equals c."
          query a — [1173] replica: operand a NEVER engaged, sentence 1.
  F-SLIP  fixture: "a is A. a times a equals b. b plus b equals c.
          c times a equals d." query d — last night's 48/48 shape.
  C1  literal, count-1, EARLY, minimal:   "a is A. a plus a equals b.
      b times a equals c." q c            <- FINGERPOST-1
  C2  derived, count-1, EARLY-ish, minimal: == F-SLIP cell (its dup's
      operand b is result-bound once)     <- fingerpost-2 pairs with C1
  C3  literal, count-2 (prior ARG use), EARLY, minimal: "a is A.
      a times a equals b. a plus a equals c. c times b equals d"? —
      careful: dup at ord 3. Use: "a is A. b is 3. a times b equals c.
      a plus a equals d. d times b equals e." q e — a engaged as given
      + arg before its dup; position mid-early ord 4... stated: C3
      position matches C4's for the count contrast.
  C4  literal, count-1, SAME position as C3, minimal: "f is 9. b is 3.
      f times b equals c. a is A. a plus a equals d. d times b equals
      e." — a given (count-1) directly before its dup at ord 5;
      crowding HIGHER though (3 prior vars) -> honest note: C3/C4
      contrast holds count at matched-position but crowding differs by
      one var; the cleaner count contrast is C1 (count-1) vs C3
      (count-2) at near positions.
  C5  literal, count-1, MID, CROWDED: "b is 3. c is 5. d is 7. a is A.
      a plus a equals e. e times b equals f." q f — dup at ord 5,
      field of 4 bound vars, operand engaged once.
  C6  derived, count-2, MID, CROWDED: "a is A. a times a equals b.
      c is 3. b times c equals d. b plus b equals e. e times a equals
      f"? e=2b<=50, f=2b*A<=250 ok. b: result + arg = count 2, dup at
      ord 5, crowded.
Structural gap stated: derived-at-ordinal-1 is unproducible (the
definition needs a prior sentence).
"""
import sys, os, json
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
os.environ.setdefault("ALG2", "1"); os.environ.setdefault("ALG_FTYPES", "8")
os.environ.setdefault("ALG_HW", "512"); os.environ.setdefault("ALG_DUP", "1")
import numpy as np
from phase1_algebra_head import T_ALG, build_params, forward, sent_indices, TOKENIZER_JSON
from beacon_closing_arm import recompute_states
from tta_alg2_dials import solve2
from tokenizers import Tokenizer
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load

tok = Tokenizer.from_file(TOKENIZER_JSON)
p = build_params(0)
sd = safe_load(".cache/g21.safetensors")
for k in p: p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
rng = np.random.RandomState(23)

def probe_FHOLD(v):
    A = int(rng.choice([5, 6, 7, 8]))
    text = f"Consider the numbers a, b, c. a plus a equals b. c is {3*A}. b plus a equals c. What is a?"
    facs = [{"ftype":"rel","op":"add","args":[0,0],"result":1},
            {"ftype":"given","var":2,"value":3*A},
            {"ftype":"rel","op":"add","args":[1,0],"result":2}]
    return text, facs, 0, A, 1, "a plus a"
def probe_FSLIP(v):
    A = int(rng.choice([2, 3, 4, 5]))
    text = (f"Consider the numbers a, b, c, d. a is {A}. a times a equals b. "
            f"b plus b equals c. c times a equals d. What is d?")
    facs = [{"ftype":"given","var":0,"value":A},
            {"ftype":"rel","op":"mul","args":[0,0],"result":1},
            {"ftype":"rel","op":"add","args":[1,1],"result":2},
            {"ftype":"rel","op":"mul","args":[2,0],"result":3}]
    return text, facs, 3, 2*A*A*A, 3, "b plus b"
def probe_C1(v):
    A = int(rng.choice([3, 4, 5, 6, 7]))
    text = (f"Consider the numbers a, b, c. a is {A}. a plus a equals b. "
            f"b times a equals c. What is c?")
    facs = [{"ftype":"given","var":0,"value":A},
            {"ftype":"rel","op":"add","args":[0,0],"result":1},
            {"ftype":"rel","op":"mul","args":[1,0],"result":2}]
    return text, facs, 2, 2*A*A, 2, "a plus a"
def probe_C3(v):
    A = int(rng.choice([3, 4, 5, 6]))
    text = (f"Consider the numbers a, b, c, d, e. a is {A}. b is 3. "
            f"a times b equals c. a plus a equals d. d times b equals e. What is e?")
    facs = [{"ftype":"given","var":0,"value":A},{"ftype":"given","var":1,"value":3},
            {"ftype":"rel","op":"mul","args":[0,1],"result":2},
            {"ftype":"rel","op":"add","args":[0,0],"result":3},
            {"ftype":"rel","op":"mul","args":[3,1],"result":4}]
    return text, facs, 4, 6*A, 4, "a plus a"
def probe_C5(v):
    A = int(rng.choice([3, 4, 5, 6]))
    text = (f"Consider the numbers a, b, c, d, e, f. b is 3. c is 5. d is 7. "
            f"a is {A}. a plus a equals e. e times b equals f. What is f?")
    facs = [{"ftype":"given","var":1,"value":3},{"ftype":"given","var":2,"value":5},
            {"ftype":"given","var":3,"value":7},{"ftype":"given","var":0,"value":A},
            {"ftype":"rel","op":"add","args":[0,0],"result":4},
            {"ftype":"rel","op":"mul","args":[4,1],"result":5}]
    return text, facs, 5, 6*A, 5, "a plus a"
def probe_C6(v):
    A = int(rng.choice([2, 3, 4, 5]))
    text = (f"Consider the numbers a, b, c, d, e, f. a is {A}. a times a equals b. "
            f"c is 3. b times c equals d. b plus b equals e. e times a equals f. What is f?")
    facs = [{"ftype":"given","var":0,"value":A},
            {"ftype":"rel","op":"mul","args":[0,0],"result":1},
            {"ftype":"given","var":2,"value":3},
            {"ftype":"rel","op":"mul","args":[1,2],"result":3},
            {"ftype":"rel","op":"add","args":[1,1],"result":4},
            {"ftype":"rel","op":"mul","args":[4,0],"result":5}]
    return text, facs, 5, 2*A*A*A, 5, "b plus b"

CELLS = [("F-HOLD", probe_FHOLD, (0,0)), ("F-SLIP", probe_FSLIP, (1,1)),
         ("C1-lit-ct1-early-min", probe_C1, (0,0)),
         ("C3-lit-ct2-early-min", probe_C3, (0,0)),
         ("C5-lit-ct1-mid-crowd", probe_C5, (0,0)),
         ("C6-der-ct2-mid-crowd", probe_C6, (1,1))]
N_VAR = 12

def read_probe(text, tgt_slot, correct, dup_phrase):
    e = tok.encode(text)
    op_char = text.index(dup_phrase) + len(dup_phrase) - 1
    op_tok = next(i for i, (a_, b_) in enumerate(e.offsets) if a_ <= op_char < b_)
    cuts = [op_tok+1+k for k in range(4)] + [len(e.ids)]
    n = len(cuts); N = ((n+7)//8)*8
    ids = np.zeros((N, T_ALG), np.int32); msk = np.zeros((N, T_ALG), np.float32); snt = np.zeros((N, T_ALG), np.int32)
    for i, pl in enumerate(cuts):
        pl = min(pl, len(e.ids))
        ids[i, :pl] = e.ids[:pl]; msk[i, :pl] = 1.0
        snt[i] = sent_indices(text[:e.offsets[pl-1][1]], list(e.offsets[:pl])+[(0,0)]*(T_ALG-pl), msk[i])
    st = recompute_states(ids)
    reads = []
    for s0 in range(0, N, 8):
        out = forward(p, Tensor(st[s0:s0+8].astype(np.float32), dtype=dtypes.float),
                      Tensor(msk[s0:s0+8].astype(np.float32), dtype=dtypes.float),
                      Tensor(snt[s0:s0+8].astype(np.int32), dtype=dtypes.int))
        args = out["args"].realize().numpy(); dup = out["dup"].realize().numpy() if "dup" in out else None
        for bi in range(8):
            if s0+bi < n:
                a = args[bi][tgt_slot]; order = np.argsort(-a)
                is_dup = bool(dup is not None and dup[bi][tgt_slot] > 0)
                b = (int(order[0]),)*2 if is_dup else tuple(sorted((int(order[0]), int(order[1]))))
                reads.append(b)
    t0_ok = any(r == correct for r in reads[:4]); end_ok = reads[-1] == correct
    return ("HOLD" if end_ok else "SLIP") if t0_ok else ("LATE-LOCK" if end_ok else "WANDER")

results = {}
for name, gen, correct in CELLS:
    dist = {"HOLD":0, "SLIP":0, "WANDER":0, "LATE-LOCK":0}
    for v in range(N_VAR):
        text, facs, q, gold, tgt_slot, phrase = gen(v)
        assert solve2(facs, q, {"n_vars":24,"m":300}) == gold, name
        dist[read_probe(text, tgt_slot, correct, phrase)] += 1
    results[name] = dist
    print(f"[{name:24s}] HOLD {dist['HOLD']:2d}  SLIP {dist['SLIP']:2d}  "
          f"WANDER {dist['WANDER']:2d}  LATE-LOCK {dist['LATE-LOCK']:2d}")
    if name == "F-HOLD" and dist["HOLD"] + dist["LATE-LOCK"] < 8:
        print("FIXTURE F-HOLD FAILED — instrument fault; NOTHING BANKS"); sys.exit(2)
    if name == "F-SLIP" and dist["SLIP"] + dist["WANDER"] < 8:
        print("FIXTURE F-SLIP FAILED — instrument fault; NOTHING BANKS"); sys.exit(2)

fp1 = results["C1-lit-ct1-early-min"]
hold1 = fp1["HOLD"] + fp1["LATE-LOCK"] >= 8
slip1 = fp1["SLIP"] + fp1["WANDER"] >= 8
ct = results["C3-lit-ct2-early-min"]
count_effect = (ct["SLIP"]+ct["WANDER"]) - (fp1["SLIP"]+fp1["WANDER"])
print(f"\nFINGERPOST-1 (lit ct1 early min): {'HOLDS' if hold1 else ('SLIPS' if slip1 else 'MIXED')}")
print(f"COUNT CONTRAST (C3 ct2 vs C1 ct1, near-matched field): slip-count delta {count_effect:+d} of 12")
if hold1 and count_effect < 3:
    verdict = "CONFIGURATION — pawl-on-pawl DIES per the standing pin (position/field decide; count adds nothing)"
elif slip1 or count_effect >= 6:
    verdict = "PAWL-ON-PAWL SUPPORTED — prior engagement drives slip beyond field"
else:
    verdict = "MIXED/TEXTURE — mechanism probe; no reinterpretation"
print("VERDICT:", verdict)
json.dump({"cells": results, "fingerpost1_holds": bool(hold1),
           "count_delta": int(count_effect), "verdict": verdict},
          open(".cache/four_axis_read.json", "w"), indent=1)
print("[done] wrote .cache/four_axis_read.json")
