"""THE CERTIFIER-SIGNAL CENSUS (2026-09-28; Bryce: "each loop's certifier updates the attention masking"; Opus's cheap
check): for the GIVEN slots whose final-breath attention lands in the WRONG SENTENCE, how often would a certifier's
signal point back toward the right one? Zero GPU, on the banked membrane intermediate (attention argmax per breath),
the beam oracle's top-1 masked parse (decoded given values per slot) and the fixture's gold. Signals:
  PULL   the gold numeral is UNCLAIMED by every decoded given (the NL certifier's "unclaimed number in the text")
  PUSH   the slot's decoded value collides with another given's (the Sinkhorn "two slots claim one number")
  UNIQUE the gold numeral occurs exactly once in the text (an implied-value search would be unambiguous)
  ANY    pull or push
Reported for wrong-sentence wrong givens, for same-sentence wrong givens (contrast), and for right givens."""
import os, sys, json, pickle, numpy as np
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
TAG = sys.argv[1] if len(sys.argv) > 1 else "PMS8_241"
import membrane_scale as MS
import phase1_algebra_head as H
tok = H._xcorr_tokenizer()
raw = np.load(f".cache/membrane_raw_wild_{TAG}.npz")
ids, tokmask, sent = raw["ids"], raw["tokmask"], raw["sent"]
pres, ftype, gdig, am = raw["g_presence"], raw["g_ftype"], raw["g_digits"], raw["argmax_breath"]
n, K_B, L_FAC = am.shape
ps = np.load(f".cache/ps_legal_wild_{TAG}.npz"); ok = {(int(r), int(c)): bool(o) for r, c, o in zip(ps["rows"], ps["slots"], ps["ok"])}
recs = {r["row"]: r for r in pickle.load(open(f".cache/beam_oracle_{TAG}.pkl", "rb")) if not r["flips"]}
groups = {"wrong-sentence": [], "same-sentence-wrong": [], "right": []}
for i in range(n):
    runs = MS._digit_runs(tok, ids[i], H.T_ALG)
    parse = recs[i]["parse"] if i in recs else []
    dec_val = {f["_slot"]: f["value"] for f in parse if f["ftype"] == "given" and "_slot" in f}
    from collections import Counter
    claimed = Counter(dec_val.values())
    for j in range(L_FAC):
        if pres[i, j] < 0.5 or int(ftype[i, j]) != 1: continue
        cell = ok.get((i, j));
        if cell is None: continue
        v = int("".join(str(int(x)) for x in gdig[i, j]))
        matches = [(a, b, val) for a, b, val in runs if val == v]
        if not matches: continue
        occ_sent = {int(sent[i, t]) for a, b, _ in matches for t in range(a, b)}
        t_att = int(am[i, K_B - 1, j]); att_sent = int(sent[i, t_att]) if tokmask[i, t_att] else -1
        pull = claimed.get(v, 0) == 0
        push = (j in dec_val) and claimed[dec_val[j]] >= 2
        uniq = len(matches) == 1
        rec = dict(pull=pull, push=push, uniq=uniq, any=pull or push, both=pull and push)
        if cell: groups["right"].append(rec)
        elif att_sent not in occ_sent: groups["wrong-sentence"].append(rec)
        else: groups["same-sentence-wrong"].append(rec)
out = [f"THE CERTIFIER-SIGNAL CENSUS — {TAG}, wild, final-breath attention, given slots"]
out.append(f"{'group':22s} {'n':>5s} {'PULL':>6s} {'PUSH':>6s} {'ANY':>6s} {'BOTH':>6s} {'UNIQUE':>7s}")
for g, L in groups.items():
    if not L: continue
    a = {k: np.mean([r[k] for r in L]) for k in ("pull", "push", "any", "both", "uniq")}
    out.append(f"{g:22s} {len(L):5d} {a['pull']:6.2f} {a['push']:6.2f} {a['any']:6.2f} {a['both']:6.2f} {a['uniq']:7.2f}")
print("\n".join(out)); open(f".cache/certifier_signal_{TAG}.txt", "w").write("\n".join(out) + "\n")
