"""decode_masks.py — THE DECODE MASKS (2026-09-17, word given; the Jev relay's one transferable idea:
never choose outside the legal set). From a raw-heads dump (LV_DUMP_RAW) and the split's jsonl, the read's
fac-exact under: none | num | pos | num+pos.
  num — a given's value is chosen among the LEGAL values (numerals in the text, lexicon constants, 1) by the
        digit heads' joint log-probability (register-free);
  pos — under the pen convention a slot may only point at variables introduced at or before it: args and
        res logits above the slot index are masked; a slot decoded as a given takes its own index as its
        variable (PROSE ONLY: mint numbers by first mention).
usage: decode_masks.py raw.pkl split.jsonl [N_DIG]"""
import pickle, sys, json, re, collections, numpy as np
sys.path.insert(0, "."); from mycelium.rulebook import legal_values as _legal_values, digits_of as _digits_of

def lsm(x): x = x - x.max(-1, keepdims=True); return x - np.log(np.exp(x).sum(-1, keepdims=True))

def legal_values(text, nd): return _legal_values(text, nd)

def digits_of(v, nd): return _digits_of(v, nd)

def score(D, texts, num=False, pos=False, nd=7):
    n_ok = 0; F = collections.Counter(); cache = {}
    for (i, j, gft, gop, gargs, gres, gdig, pres, ft, op, args, res, dig, dup) in D:
        ft_hat = int(ft.argmax()); args = args.copy(); res = res.copy()
        if pos:
            K = len(res); idx = np.arange(K); args[idx > j] = -1e9; res[idx > j] = -1e9
            if ft_hat != 0: res[:] = -1e9; res[j] = 0.0   # a given's variable is its slot
        res_hat = int(res.argmax())
        if num and ft_hat != 0:
            if i not in cache: cache[i] = legal_values(texts[i], nd)
            ls = lsm(dig); best = None
            for v in cache[i]:
                sc = sum(ls[d, dd] for d, dd in enumerate(digits_of(v, nd)))
                if best is None or sc > best[0]: best = (sc, v)
            dig_hat = digits_of(best[1], nd) if best else dig.argmax(-1).tolist()
        else:
            dig_hat = dig.argmax(-1).tolist()
        ok = pres > 0 and ft_hat == gft and res_hat == gres
        f_args = None
        if gft == 0:
            ok = ok and int(op.argmax()) == gop
            if len(gargs) == 1: f_args = dup > 0 and int(args.argmax()) in gargs
            else: f_args = set(np.argsort(-args)[:2].tolist()) == set(gargs)
            ok = ok and f_args; F["args"] += bool(f_args); F["args_n"] += 1
        else:
            f_dig = dig_hat == gdig; ok = ok and f_dig; F["dig"] += bool(f_dig); F["dig_n"] += 1
        F["res"] += res_hat == gres; F["ftype"] += ft_hat == gft; F["n"] += 1; n_ok += bool(ok)
    return n_ok / F["n"], {"ftype": F["ftype"] / F["n"], "res": F["res"] / F["n"], "args": F["args"] / max(F["args_n"], 1), "dig": F["dig"] / max(F["dig_n"], 1)}

if __name__ == "__main__":
    D = pickle.load(open(sys.argv[1], "rb")); rows = [json.loads(l) for l in open(sys.argv[2])]; texts = {i: r["text"] for i, r in enumerate(rows)}
    nd = int(sys.argv[3]) if len(sys.argv) > 3 else len(D[0][6])
    base = None
    for name, kw in (("none", {}), ("num", {"num": True}), ("pos", {"pos": True}), ("num+pos", {"num": True, "pos": True})):
        acc, f = score(D, texts, nd=nd, **kw); base = acc if base is None else base
        print(f"[dmask] {sys.argv[1].split('/')[-1]:<34} {name:<8} fac-exact {acc:.4f} ({acc - base:+.4f}) | ftype {f['ftype']:.3f} res {f['res']:.3f} args {f['args']:.3f} dig {f['dig']:.3f}")
