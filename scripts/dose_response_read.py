"""dose_response_read.py — THE DOSE-RESPONSE (registered 2026-07-29,
bars pinned blind; the road's highest-value open experiment fires).

Fixed-position dup-add target on a DERIVED operand; dose = POST-target
distractor load (the erosion axis). Sub-anatomy per probe from a
three-point-plus trace: T0 (target-sentence end), after each subsequent
sentence, full text. Classes: HOLD / SLIP / WANDER / LATE-LOCK.

Probe family (dose d, variant v):
  "Consider the numbers a..<letters>. a is <A>. a times a equals b.
   b plus b equals c."                                   <- FIXED prefix
  [d distractor sentences: "<L> is <small>."]            <- THE DOSE
  "c times a equals <last>. What is <last>?"             <- closing chain
All values <=300, letters consecutive, solver-verified (A in 2..7 so
b=A^2<=49, c=2b<=98, final c*A<=686? -> cap A<=5: c*A<=250). Gold =
solve2 on the intended graph.
"""
import sys, os, json
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
os.environ.setdefault("ALG2", "1"); os.environ.setdefault("ALG_FTYPES", "8")
os.environ.setdefault("ALG_HW", "512"); os.environ.setdefault("ALG_DUP", "1")
import numpy as np
from phase1_algebra_head import T_ALG, L_FAC, build_params, forward, sent_indices, TOKENIZER_JSON
from beacon_closing_arm import recompute_states
from tta_alg2_dials import solve2
from tokenizers import Tokenizer
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load

tok = Tokenizer.from_file(TOKENIZER_JSON)
p = build_params(0)
sd = safe_load(".cache/g21.safetensors")
for k in p: p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()

LETTERS = "abcdefghij"
rng = np.random.RandomState(11)

def make_probe(d, v):
    A = int(rng.choice([2, 3, 4, 5]))
    n_vars = 3 + d + 1
    letters = LETTERS[:n_vars]
    dl = letters[3:3 + d]              # distractor letters
    last = letters[3 + d]
    smalls = rng.choice([2,3,4,5,6,7,8,9], size=d, replace=False) if d else []
    sents = [f"Consider the numbers {', '.join(letters)}.",
             f"a is {A}.", "a times a equals b.", "b plus b equals c."]
    facs = [{"ftype":"given","var":0,"value":A},
            {"ftype":"rel","op":"mul","args":[0,0],"result":1},
            {"ftype":"rel","op":"add","args":[1,1],"result":2}]
    for i, L_ in enumerate(dl):
        sents.append(f"{L_} is {int(smalls[i])}.")
        facs.append({"ftype":"given","var":3+i,"value":int(smalls[i])})
    sents.append(f"c times a equals {last}.")
    facs.append({"ftype":"rel","op":"mul","args":[2,0],"result":3+d})
    text = " ".join(sents) + f" What is {last}?"
    gold = solve2(facs, 3+d, {"n_vars": 24, "m": 300})
    assert gold == 2*A*A*A, (gold, A)
    return text, gold

def parse_prefix_batchable(items):
    """items: list of (text, prefix_len_tokens). Returns per-item (dup, binding) for slot 2."""
    n = len(items); N = ((n+7)//8)*8
    ids = np.zeros((N, T_ALG), np.int32); msk = np.zeros((N, T_ALG), np.float32); snt = np.zeros((N, T_ALG), np.int32)
    for i, (t, pl) in enumerate(items):
        e = tok.encode(t); pl = min(pl, len(e.ids), T_ALG)
        ids[i, :pl] = e.ids[:pl]; msk[i, :pl] = 1.0
        snt[i] = sent_indices(t[:e.offsets[pl-1][1]], list(e.offsets[:pl])+[(0,0)]*(T_ALG-pl), msk[i])
    st = recompute_states(ids)
    outs = []
    for s0 in range(0, N, 8):
        out = forward(p, Tensor(st[s0:s0+8].astype(np.float32), dtype=dtypes.float),
                      Tensor(msk[s0:s0+8].astype(np.float32), dtype=dtypes.float),
                      Tensor(snt[s0:s0+8].astype(np.int32), dtype=dtypes.int))
        args = out["args"].realize().numpy()
        dup  = out["dup"].realize().numpy() if "dup" in out else None
        for bi in range(8):
            if s0+bi < n:
                a = args[bi][TGT_SLOT]                   # fixture-resolved target slot
                order = np.argsort(-a); t1, t2 = int(order[0]), int(order[1])
                is_dup = bool(dup is not None and dup[bi][TGT_SLOT] > 0)
                binding = (t1, t1) if is_dup else tuple(sorted((t1, t2)))
                outs.append((is_dup, binding))
    return outs

CORRECT = (1, 1)   # b plus b: args [1,1]
TGT_SLOT = 3       # FIXTURE-RESOLVED (apparatus fault #1 in run 1: hardcoded 2
                   # read the 'a times a' slot, whose correct (0,0) classified as
                   # wrong -> 48/48 WANDER, an instrument reading not a finding).
                   # Probe sentences: preamble=0, 'a is A'=1, 'a times a'=2,
                   # TARGET 'b plus b'=3. Verified by full-parse fixture below.
DOSES = [0, 1, 2, 3]
N_VAR = 12
results = {d: {"HOLD":0, "SLIP":0, "WANDER":0, "LATE-LOCK":0} for d in DOSES}
detail = []
for d in DOSES:
    for v in range(N_VAR):
        text, gold = make_probe(d, v)
        e = tok.encode(text)
        # prefix points: end of target sentence; after each subsequent sentence; full
        tgt_end_char = text.index("equals c.") + len("equals c.")
        # mid-sentence cuts at operand+1..+4 TOKENS (apparatus fault #2 in run 1:
        # sentence-end T0 is already past the transient the hand-traces showed)
        op_char = text.index("b plus b") + len("b plus b") - 1
        op_tok = next(i for i,(a_,b_) in enumerate(tok.encode(text).offsets) if a_<=op_char<b_)
        mid_toks = [op_tok+1+k for k in range(4)]
        cuts = [tgt_end_char]
        pos = tgt_end_char
        for _ in range(d + 1):                      # distractor sentences + closing chain sentence
            nxt = text.find(".", pos + 1)
            if nxt == -1: break
            cuts.append(nxt + 1); pos = nxt + 1
        cut_toks = list(mid_toks)
        for cc in cuts:
            ct = next(i+1 for i, (a_, b_) in enumerate(e.offsets) if b_ >= cc)
            cut_toks.append(ct)
        cut_toks.append(len(e.ids))                 # full
        reads = parse_prefix_batchable([(text, ct) for ct in cut_toks])
        # engaged = correct at ANY point in the transient window (mid cuts + T0)
        t0_ok  = any(r[1] == CORRECT for r in reads[:5])
        end_ok = reads[-1][1] == CORRECT
        cls = ("HOLD" if end_ok else "SLIP") if t0_ok else ("LATE-LOCK" if end_ok else "WANDER")
        results[d][cls] += 1
        detail.append({"dose": d, "variant": v, "class": cls,
                       "trace": [list(map(int, r[1])) + [int(r[0])] for r in reads]})
    r_ = results[d]
    print(f"[dose {d}] HOLD {r_['HOLD']:2d}  SLIP {r_['SLIP']:2d}  WANDER {r_['WANDER']:2d}  LATE-LOCK {r_['LATE-LOCK']:2d}")

def slip_frac(d):
    eng = results[d]["HOLD"] + results[d]["SLIP"]
    return results[d]["SLIP"] / eng if eng else float("nan")
fracs = [slip_frac(d) for d in DOSES]
print(f"\nslip-fraction (among engaged-at-T0) by dose: " + " ".join(f"D{d}:{f:.2f}" for d, f in zip(DOSES, fracs)))
mono = all(fracs[i+1] >= fracs[i] - 1e-9 for i in range(len(fracs)-1))
delta = fracs[-1] - fracs[0]
if mono and delta >= 0.25:
    verdict = "DOSE-RESPONSE PRESENT — the eroder confirmed; the mechanism watched working"
elif abs(delta) < 0.10:
    verdict = "FLAT — post-target distractor load is NOT the eroder; species re-diagnosis owed"
else:
    verdict = "NON-MONOTONE/INTERMEDIATE — texture; mechanism probe; no reinterpretation"
print("VERDICT:", verdict)
json.dump({"results": {str(d): results[d] for d in DOSES}, "slip_fracs": fracs,
           "delta": delta, "monotone": bool(mono), "verdict": verdict, "detail": detail},
          open(".cache/dose_response_read.json", "w"), indent=1)
print("[done] wrote .cache/dose_response_read.json")
