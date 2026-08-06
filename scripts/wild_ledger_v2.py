"""wild_ledger.py — THE WILD LEDGER (2026-07-31, registered before
firing). The census-scale deployment survey: every gold-bearing harvest
item under the standard eligibility filter, all levels, through the
deployed chain (manifest gate + manifest mouth) in deployment mode.

PER-ITEM BANKING (the bank-don't-read pattern): raw mouth residual (any
band edge cuts at analysis, no re-run), quorum count, plurality, trace
trips, attest verdict, gold, correct, consumption flag. The artifact is
the actuarial fixture behind any future tier dial. Survey, not bar
experiment — numbers are descriptive by registration. The curve is
measured where gold exists; beyond gold it is inference (stated scope).
"""
import sys, os, json, re, hashlib
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
os.environ.setdefault("ALG2", "1"); os.environ.setdefault("ALG_FTYPES", "8")
os.environ.setdefault("ALG_HW", "512"); os.environ.setdefault("ALG_DUP", "1")
import numpy as np
from collections import Counter, defaultdict
from phase1_algebra_head import T_ALG, build_params, forward, decode, sent_indices, TOKENIZER_JSON
from beacon_closing_arm import recompute_states
from tta_views import permuted_view
from tta_alg2_dials import solve2
from mycelium.attestation import attest_quorum_v3
from mycelium.trace_layer import trace_trips
from tokenizers import Tokenizer
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load

MAN = json.load(open(".cache/GENERATION.json"))
CKPT = MAN["parser_ckpt"]
tok = Tokenizer.from_file(TOKENIZER_JSON)
p = build_params(0)
sd = safe_load(CKPT)
assert set(sd.keys()) == set(p.keys())
for k in p: p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
print(f"[ledger] gate from manifest: {CKPT}")

# --- the mouth (manifest artifact) ---
zc = np.load(MAN["mouth"])
assert "coef" in zc.files, "expected entourage-format mouth (bank+coef+thr_knn)"
coef, mthr = zc["coef"].astype(np.float64), float(zc["thr_knn"])
bank = zc["bank"]
print(f"[ledger] mouth from manifest: {MAN['mouth']} (thr {mthr:.4f})")

def int_answer(a):
    s = str(a).strip().replace("$", "").replace(",", "")
    return int(s) if re.fullmatch(r"-?\d+", s) else None

h = [json.loads(l) for l in open(".cache/math_harvest_v0.jsonl")]
pop = []
for hi, x in enumerate(h):
    t = x["problem"]
    if len(t) >= 300 or "asy]" in t: continue
    if any(int(n) > 300 for n in re.findall(r"\d+", t)): continue
    g = int_answer(x["answer"])
    if g is None or not (0 <= g <= 300): continue
    pop.append((hi, x["level"], t, g))
print(f"[ledger] population: {len(pop)} gold-bearing items "
      f"({dict(Counter(l for _, l, _, _ in pop))})")

# consumption flag: text-identity against every book-candidate record on disk
consumed = set()
import glob
for f in glob.glob(".cache/book*_candidates*.json"):  # deep clean 2026-08-01: the old glob missed per-tranche files (66 vs 426 consumed)
    try:
        d = json.load(open(f))
    except Exception:
        continue
    vals = d.values() if isinstance(d, dict) else [d]
    for v in vals:
        if isinstance(v, list):
            for c in v:
                if isinstance(c, dict):
                    for key in ("problem", "text", "src_text"):
                        if key in c: consumed.add(hashlib.sha256(c[key].encode()).hexdigest())
print(f"[ledger] consumed-source texts found in candidate records: {len(consumed)}")

def parse_batch(texts):
    n = len(texts); N = ((n+7)//8)*8
    ids = np.zeros((N, T_ALG), np.int32); msk = np.zeros((N, T_ALG), np.float32); snt = np.zeros((N, T_ALG), np.int32)
    for i, t in enumerate(texts):
        e = tok.encode(t); L = min(len(e.ids), T_ALG)
        ids[i, :L] = e.ids[:L]; msk[i, :L] = 1.0
        snt[i] = sent_indices(t, list(e.offsets), msk[i])
    st = recompute_states(ids)
    out_r = []
    for s0 in range(0, N, 8):
        out = forward(p, Tensor(st[s0:s0+8].astype(np.float32), dtype=dtypes.float),
                      Tensor(msk[s0:s0+8].astype(np.float32), dtype=dtypes.float),
                      Tensor(snt[s0:s0+8].astype(np.int32), dtype=dtypes.int))
        keys = ("pres","ftype","op","islit","dig","args","res","query") + (("sel",) if "sel" in out else ()) + (("dup",) if "dup" in out else ())
        o = {k: out[k].realize().numpy() for k in keys}
        for bi in range(8):
            if s0+bi < n: out_r.append(decode({k: o[k][bi] for k in o}))
    return out_r

# mouth residuals for the whole population (batched)
V, Ls = [], []
texts_all = [t for _, _, t, _ in pop]
for s0 in range(0, len(texts_all), 8):
    chunk = texts_all[s0:s0+8]
    ids = np.zeros((8, T_ALG), np.int32); msk = np.zeros((8, T_ALG), np.float32)
    for i, t in enumerate(chunk):
        e = tok.encode(t); L = min(len(e.ids), T_ALG)
        ids[i, :L] = e.ids[:L]; msk[i, :L] = 1.0
    s = recompute_states(ids).astype(np.float32)
    v = (s * msk[:, :, None]).sum(1) / np.maximum(msk.sum(1)[:, None], 1)
    V.append(v[:len(chunk)]); Ls.append(msk.sum(1)[:len(chunk)])
    if (s0 // 8) % 40 == 0:
        print(f"[mouth] {s0}/{len(texts_all)}", flush=True)
V = np.concatenate(V); Ls = np.concatenate(Ls)
V /= np.linalg.norm(V, axis=1, keepdims=True)
res_all = np.sort(1.0 - V @ bank.T, axis=1)[:, :8].mean(1) - (coef[0] + coef[1] / Ls)

_norm=lambda t:" ".join(t.split())
_mix=set()
import json as _j
with open(".cache/gen23_mix.jsonl") as _f:
    for _ln in _f: _mix.add(_norm(_j.loads(_ln).get("text","")))
OUT = ".cache/wild_ledger_v2.jsonl"
n_done = 0
with open(OUT, "w") as fout:
    for j, (hi, level, text, gold) in enumerate(pop):
        rec = {"harvest_idx": hi, "level": level, "chars": len(text),
               "text_sha": hashlib.sha256(text.encode()).hexdigest()[:16],
               "gold": gold,
               "consumed": hashlib.sha256(text.encode()).hexdigest() in consumed,
               "mouth_residual": float(res_all[j]),
               "mouth_foreign": bool(res_all[j] > mthr)}
        vt = [text] + [permuted_view(text, 97000 + 10*j + k) for k in range(1, 5)]
        views = [(f, q, solve2(f, q, {"n_vars": 24, "m": 300})) for f, q in parse_batch(vt)]
        nn = [a for _, _, a in views if a is not None]
        c = Counter(nn).most_common(1); plur, cnt = c[0] if c else (None, 0)
        rec["trained"] = _norm(text) in _mix
        rec["quorum"] = cnt; rec["plur"] = plur
        if plur is not None:
            rec["would_be"] = plur; rec["correct"] = (plur == gold)
        if cnt >= 3:
            win = [f_ for (f_, q_, a_) in views if a_ == plur]
            rec["trips"] = trace_trips(text, win)
            att, _, _ = attest_quorum_v3(views, plur, text, 300, solve2)
            rec["attest"] = bool(att)
            if rec["trips"]: rec["tier"] = "tripped"
            elif not att: rec["tier"] = "attest_block"
            else:
                rec["tier"] = "answered"
                rec["correct"] = (plur == gold)
        else:
            rec["tier"] = "no_quorum"
        fout.write(json.dumps(rec) + "\n")
        n_done += 1
        if n_done % 50 == 0:
            print(f"[chain] {n_done}/{len(pop)}", flush=True)

# --- the curve print (descriptive; any edge re-cuttable from the jsonl) ---
recs = [json.loads(l) for l in open(OUT)]
print(f"\n=== THE WILD LEDGER (n={len(recs)}; gate {CKPT}; mouth thr {mthr:.4f}) ===")
for scope_name, scope in (("ALL", recs), ("DISJOINT", [r for r in recs if not r["consumed"]])):
    print(f"--- {scope_name} (n={len(scope)}) ---")
    for level in sorted({r["level"] for r in scope}):
        sub = [r for r in scope if r["level"] == level]
        native = [r for r in sub if not r["mouth_foreign"]]
        ans = [r for r in sub if r["tier"] == "answered"]
        ans_nat = [r for r in ans if not r["mouth_foreign"]]
        ok = sum(1 for r in ans if r.get("correct"))
        ok_nat = sum(1 for r in ans_nat if r.get("correct"))
        print(f"  {level:8s} n={len(sub):4d} mouth-native {len(native):4d} | "
              f"answered {len(ans):4d} (correct {ok}, precision "
              f"{ok/max(len(ans),1):.2f}) | native-answered {len(ans_nat):3d} "
              f"(precision {ok_nat/max(len(ans_nat),1):.2f})")
    # the T1-analog curve: precision among answered, by residual quartile of the FOREIGN answered
    fa = sorted([r for r in scope if r["tier"] == "answered" and r["mouth_foreign"]],
                key=lambda r: r["mouth_residual"])
    if fa:
        qn = max(1, len(fa)//4)
        print(f"  T1-analog (foreign-but-answered, n={len(fa)}), by residual quartile:")
        for qi in range(0, len(fa), qn):
            q = fa[qi:qi+qn]
            okq = sum(1 for r in q if r.get("correct"))
            print(f"    residual [{q[0]['mouth_residual']:+.4f},{q[-1]['mouth_residual']:+.4f}] "
                  f"n={len(q)} precision {okq/len(q):.2f}")
print(f"[saved] {OUT}")
