"""certifier_bench.py — THE CERTIFIER'S BENCH (2026-09-18): among rows whose decoded graph the solver ACCEPTS
(consistent), does any inference-time signal separate the RIGHT ones from the WRONG ones? Signals from the raw
slot heads (CA_RAWDUMP) per row: mean slot confidence (max-prob over present slots), min slot confidence, mean
args margin on relations (top1 - top2 sigmoid), mean res margin, presence margin, n present slots, the share of
given values the numeral mask changed, the head's total log-likelihood of its argmax graph. AUROC (right vs wrong)
per signal, and the coverage at 0.9 precision. Labels: CA_ROWS json from chain_acc (mask=1).
usage: certifier_bench.py rawslots.pkl rows.json"""
import sys, json, pickle, numpy as np
sys.path.insert(0, "."); from mycelium.rulebook import legal_values, digits_of

def sig(x): return 1 / (1 + np.exp(-x))
def sm(x): e = np.exp(x - x.max(-1, keepdims=True)); return e / e.sum(-1, keepdims=True)

def signals(r):
    L = r["ftype"].shape[0]; pres = sig(r["pres"].reshape(L)); on = pres > 0.5
    if on.sum() == 0: return None
    ft = sm(r["ftype"]); res = sm(r["res"]); args = sig(r["args"]); dig = sm(r["dig"])
    conf = []; amarg = []; rmarg = []; changed = 0; givens = 0; ll = 0.0
    vals = legal_values(r["text"], dig.shape[1])
    for j in np.where(on)[0]:
        c = ft[j].max() * res[j].max(); conf.append(c); ll += np.log(pres[j] + 1e-9) + np.log(ft[j].max() + 1e-9) + np.log(res[j].max() + 1e-9)
        srt = np.sort(res[j])[::-1]; rmarg.append(srt[0] - srt[1])
        if ft[j].argmax() == 0:
            a = np.sort(args[j])[::-1]; amarg.append(a[1] - a[2] if len(a) > 2 else a[1])
        else:
            givens += 1; free = dig[j].argmax(-1); nd = dig.shape[1]; free_v = sum(int(d) * 10 ** (nd - 1 - k) for k, d in enumerate(free))
            lg = np.log(dig[j] + 1e-9); best = max(vals, key=lambda v: sum(lg[k, dd] for k, dd in enumerate(digits_of(v, nd)))) if vals else free_v
            changed += int(best != free_v); ll += sum(lg[k, dd] for k, dd in enumerate(digits_of(best, nd)))
    return {"mean_conf": float(np.mean(conf)), "min_conf": float(np.min(conf)), "args_margin": float(np.mean(amarg)) if amarg else 0.0,
            "res_margin": float(np.mean(rmarg)), "pres_margin": float(np.mean(np.abs(pres[on] - 0.5))), "n_slots": float(on.sum()),
            "mask_changed_share": changed / max(givens, 1), "loglik": float(ll), "loglik_per_slot": float(ll / on.sum())}

def auroc(pos, neg):
    pos = np.asarray(pos); neg = np.asarray(neg)
    if len(pos) == 0 or len(neg) == 0: return float("nan")
    return float(((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()))

if __name__ == "__main__":
    D = pickle.load(open(sys.argv[1], "rb")); labels = json.load(open(sys.argv[2]))
    S = {r["i"]: signals(r) for r in D}; right = [i for i, l in labels.items() if l == "correct"]; wrong = [i for i, l in labels.items() if l == "wrong"]
    right = [int(i) for i in right if S.get(int(i))]; wrong = [int(i) for i in wrong if S.get(int(i))]
    print(f"[bench] {sys.argv[1].split('/')[-1]}: consistent rows right {len(right)} wrong {len(wrong)}")
    for k in ("mean_conf", "min_conf", "args_margin", "res_margin", "pres_margin", "loglik_per_slot", "loglik", "n_slots", "mask_changed_share"):
        pos = [S[i][k] for i in right]; neg = [S[i][k] for i in wrong]; a = auroc(pos, neg)
        # coverage at 0.9 precision: the fraction of consistent rows above the threshold where >= 90% are right (signal high = right, or low = right if AUROC < 0.5)
        allv = sorted([(S[i][k], 1) for i in right] + [(S[i][k], 0) for i in wrong], key=lambda t: -t[0] if a >= 0.5 else t[0]); best = 0
        c = 0; n = 0
        for v, y in allv:
            n += 1; c += y
            if c / n >= 0.9 and c >= 3: best = n
        print(f"   {k:<20} AUROC {a:.3f} | coverage at 0.9 precision: {best} rows of {len(allv)} | right mean {np.mean(pos):.3f} wrong mean {np.mean(neg):.3f}")
