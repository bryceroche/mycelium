"""neighbor_gain_read.py — where did dry_d12's +25 come from? (2026-07-31,
the battery-hold's read). Per-row bigtest under g22 and dry_d12
(single-view, banked states); GAINED rows classified: dup-adjacent
(gold factors carry args=[a,a] OR text carries 'X plus/times X' same
letter) vs other. Concentrated = in-band transfer (sensible);
spread = unexplained (the hold stands harder)."""
import sys, os, json, re
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
os.environ.setdefault("ALG2", "1"); os.environ.setdefault("ALG_FTYPES", "8")
os.environ.setdefault("ALG_HW", "512"); os.environ.setdefault("ALG_DUP", "1")
import numpy as np
from phase1_algebra_head import T_ALG, build_params, forward, decode
from tta_alg2_dials import solve2
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load

rows = [json.loads(l) for l in open(".cache/algebra_nl_bigtest.jsonl")]
z = np.load(".cache/phase1_alg_states_bigtest.npz")
st = (z["states"] if "states" in z.files
      else np.load(".cache/phase1_alg_states_bigtest_states.npy", mmap_mode="r"))
tokmask, sent = z["tokmask"], z["sent"]
n = len(rows)
print(f"[gain] bigtest n={n}")

def correct_flags(ckpt):
    p = build_params(0)
    sd = safe_load(ckpt)
    for k in p: p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
    ok = np.zeros(n, bool)
    for s0 in range(0, n, 8):
        sl = slice(s0, min(s0+8, n))
        pad = 8 - (sl.stop - sl.start)
        idx = list(range(sl.start, sl.stop)) + [sl.start]*pad
        out = forward(p, Tensor(np.asarray(st[idx], np.float32), dtype=dtypes.float),
                      Tensor(tokmask[idx].astype(np.float32), dtype=dtypes.float),
                      Tensor(sent[idx].astype(np.int32), dtype=dtypes.int))
        keys = ("pres","ftype","op","islit","dig","args","res","query") + (("sel",) if "sel" in out else ()) + (("dup",) if "dup" in out else ())
        o = {k: out[k].realize().numpy() for k in keys}
        for bi in range(sl.stop - sl.start):
            i = s0 + bi
            facs, q = decode({k: o[k][bi] for k in o})
            a = solve2(facs, q, {"n_vars": rows[i]["n_vars"], "m": rows[i].get("m", 300)})
            ok[i] = (a == rows[i]["solution"][rows[i]["query_var"]])
        if (s0 // 8) % 40 == 0: print(f"  [{s0}/{n}]", flush=True)
    return ok

def is_dup_adjacent(r):
    if any(f.get("ftype") == "rel" and len(f.get("args", [])) == 2
           and f["args"][0] == f["args"][1] for f in r["factors"]):
        return True
    return bool(re.search(r"\b(\w+) (?:plus|times) \1\b", r["text"]))

dup_adj = np.array([is_dup_adjacent(r) for r in rows])
print(f"[gain] dup-adjacent rows in bigtest: {dup_adj.sum()}/{n} = {dup_adj.mean():.1%}")
g22 = correct_flags(".cache/g22.safetensors")
arm = correct_flags(".cache/g23_dry_d12.safetensors")
print(f"[gain] single-view correct: g22 {g22.sum()}  dry_d12 {arm.sum()}  (delta {arm.sum()-g22.sum():+d})")
gained = arm & ~g22
lost = g22 & ~arm
gd = dup_adj[gained].sum()
print(f"[gain] GAINED {gained.sum()} (dup-adjacent {gd} = {gd/max(gained.sum(),1):.0%} vs population {dup_adj.mean():.0%})")
print(f"[gain] LOST {lost.sum()} (dup-adjacent {dup_adj[lost].sum()})")
conc = (gd/max(gained.sum(),1)) / max(dup_adj.mean(), 1e-9)
verdict = ("CONCENTRATED — in-band transfer (the gain is the cure reaching bigtest's own dup phrasings); the promotion case can name its mechanism"
           if conc >= 2.0 else
           "SPREAD — the gain is NOT dup-shaped; mechanism unexplained; the hold stands harder"
           if conc <= 1.3 else "PARTIAL concentration — mixed mechanism")
print(f"=== concentration {conc:.1f}x -> VERDICT (pinned): {verdict} ===")
json.dump({"g22": int(g22.sum()), "arm": int(arm.sum()),
           "gained": int(gained.sum()), "gained_dup": int(gd),
           "lost": int(lost.sum()), "pop_dup_share": float(dup_adj.mean()),
           "concentration": conc, "verdict": verdict},
          open(".cache/neighbor_gain_read.json", "w"), indent=1)
print("[saved] .cache/neighbor_gain_read.json")
