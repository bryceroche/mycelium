"""differential_read.py — the hammerhead's instrument (gut #58, fired
from the driver's seat 2026-07-24; bar pinned in the ledger first).

Aperture A: banked per-item routing fidelity (crown_v4 on bigtest).
Aperture B: per-item mean centered-cosine distance of slot fst vectors
to their own kind centroids (gen-16 bank), one forward pass.
Differential = |zA - zB|; populations from banked V4 votes.
LIGHTS: cold mean differential >= correct mean + 0.5*correct std.
FLAT:   otherwise -> cold errors route permanently to guard + books.
Diagnostic only — photographs disagreement, never adjudicates.
"""
import json, sys, os
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
from collections import Counter
os.environ.setdefault("ALG2", "1"); os.environ["ALG_FTYPES"] = "8"
os.environ.setdefault("ALG_DUP", "1")
from phase1_algebra_head import build_params
from waist_abstention_probe import compute_fst
from tinygrad.nn.state import safe_load

z = np.load(".cache/phase1_alg_states_bigtest.npz")
st, tk, se = z["states"], z["tokmask"], z["sent"]
rows = [json.loads(l) for l in open(".cache/algebra_nl_bigtest.jsonl")]
gold = [r["solution"][r["query_var"]] for r in rows]
votes = json.load(open(".cache/lattice_gen16_V4.json"))["bigtest"]
fid = np.array(json.load(open(".cache/routing_fidelity.json"))["fid"])
cb = np.load(".cache/monitor_centroids_gen16.npz")
cents = {k: cb[k] / np.linalg.norm(cb[k]) for k in cb.files}


def kind_of(f):
    t = f["ftype"]
    if t == "rel":
        return "rel_" + f["op"] if f["op"] in ("add", "mul") else None
    return t if t in cents else None


p = build_params(0)
sd = safe_load(".cache/crown_reader_v4.safetensors")
for k in p:
    p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()

n = 1500
coll = np.full(n, np.nan)
for s0 in range(0, n, 8):
    sl = np.arange(s0, min(s0 + 8, n))
    pad = 8 - len(sl)
    sl_p = np.concatenate([sl, sl[:1].repeat(pad)]) if pad else sl
    fst = compute_fst(p, st, tk, se, sl_p)      # (8, L_FAC, 512)
    for bi, i in enumerate(sl):
        i = int(i)
        ds = []
        for j, f in enumerate(rows[i]["factors"][:fst.shape[1]]):
            kd = kind_of(f)
            if kd is None:
                continue
            v = fst[bi, j]
            nv = np.linalg.norm(v)
            if nv < 1e-6:
                continue
            ds.append(1.0 - float(np.dot(v / nv, cents[kd])))
        if ds:
            coll[i] = float(np.mean(ds))

ok = ~np.isnan(coll)
zA = (fid - fid[ok].mean()) / fid[ok].std()
zB = (coll - coll[ok].mean()) / coll[ok].std()
diff = np.abs(zA - zB)

cold, corr = [], []
for i in range(n):
    if not ok[i]:
        continue
    vs = [v for v in votes[i] if v is not None]
    top, cnt = (Counter(vs).most_common(1)[0] if vs else (None, 0))
    if cnt == 5 and top != gold[i]:
        cold.append(i)
    elif cnt >= 3 and top == gold[i]:
        corr.append(i)

d_cold = diff[cold]
d_corr = diff[corr]
bar = d_corr.mean() + 0.5 * d_corr.std()
lights = d_cold.mean() >= bar
print(f"[diff] items scored {int(ok.sum())}/{n} | cold {len(cold)} | correct {len(corr)}")
print(f"[diff] differential: cold mean {d_cold.mean():.3f} vs correct mean "
      f"{d_corr.mean():.3f} (std {d_corr.std():.3f}) | bar {bar:.3f}")
print(f"[diff] aperture means on cold: fid z {zA[cold].mean():+.3f} | "
      f"collapse z {zB[cold].mean():+.3f}")
verdict = ("LIGHTS — deep deceptions fool the apertures DIFFERENTLY; the "
           "differential is the cold-error detector" if lights else
           "FLAT — both apertures deceived identically; cold errors live below "
           "all learned geometry -> route PERMANENTLY to guard + books")
print(f"[diff] VERDICT: {verdict}")
json.dump(dict(cold=[int(i) for i in cold], d_cold_mean=float(d_cold.mean()),
               d_corr_mean=float(d_corr.mean()), d_corr_std=float(d_corr.std()),
               bar=float(bar), zA_cold=float(zA[cold].mean()),
               zB_cold=float(zB[cold].mean()), lights=bool(lights),
               verdict=verdict),
          open(".cache/differential_read.json", "w"), indent=1)
print("[diff] banked -> .cache/differential_read.json")
