"""diet_stage.py -- stage a training split COMPOSED of rows of banked splits (2026-10-17; the two-arm run: the real-only
diet and the real + name-view diet). Trunk states are a function of the TEXT only, so a row's banked state is reused (a
slice), never recomputed; gold / tokmask / sentence indices are sliced the same way.
THE GATES (before the states are written):
  TEXT      out row k's text = its part's source jsonl row idx[k] (the slice is the slice we think it is)
  TOKMASK   tokenize(OUT.jsonl) reproduces the sliced tokmask; SENT the recomputed sentence indices = the sliced ones
  GOLD      build_gold(OUT.jsonl), in row chunks, reproduces EVERY sliced banked g_* array bit-identically (keys mirror the
            first part's banked key set; every part must carry the same set)
Then the npz is saved stamped with mix_sha16(OUT.jsonl) (the SHA-FENCE binds the split to its content).
usage: (the family env) diet_stage.py OUT_SPLIT OUT.jsonl PART [PART ...]   PART = SPLIT@SRC.jsonl[@IDX.npy]
"""
import json, os, shutil, sys
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np  # noqa: E402
import phase1_algebra_head as PH  # noqa: E402
from mycelium.era import mix_sha16  # noqa: E402

out, out_jsonl, specs = sys.argv[1], sys.argv[2], sys.argv[3:]
parts = []
for sp in specs:
    f = sp.split("@")
    split, src = f[0], f[1]
    z = np.load(PH.STATES_NPZ.format(split=split))
    n_src = z["tokmask"].shape[0]
    idx = np.load(f[2]) if len(f) > 2 else np.arange(n_src)
    if "mix_sha" in z.files:
        assert str(z["mix_sha"]) == mix_sha16(src), f"{split} is not staged from {src}"
    else:  # an unstamped part is only admitted by name (the TEXT gate below still binds its rows)
        assert split in os.environ.get("DIET_STAGE_UNSTAMPED", "").split(","), \
            f"{split} carries no mix_sha stamp; admit it explicitly with DIET_STAGE_UNSTAMPED={split}"
    parts.append((split, src, idx, z))
keys = sorted(k for k in parts[0][3].files if k.startswith("g_"))
for p in parts:
    assert sorted(k for k in p[3].files if k.startswith("g_")) == keys, f"part {p[0]} carries a different gold key set"
rows = [json.loads(l) for l in open(out_jsonl)]
N = sum(len(p[2]) for p in parts)
assert len(rows) == N, f"{out_jsonl} has {len(rows)} rows, the parts {N}"
# TEXT
k0 = 0
for split, src, idx, z in parts:
    st = [json.loads(l)["text"] for l in open(src)]
    for j, i in enumerate(idx):
        assert rows[k0 + j]["text"] == st[i], f"TEXT GATE FAIL: out row {k0 + j} != {split} row {i}"
    k0 += len(idx)
print(f"[diet-stage] TEXT gate PASS: {N} rows from {[(p[0], len(p[2])) for p in parts]}", flush=True)
# THE SIBLING FENCE (2026-10-18): no training text within 0.90 similarity of a MATH-500 problem (mycelium/sibling_fence.py)
from mycelium.sibling_fence import siblings  # noqa: E402
sib = siblings([r["text"] for r in rows])
assert not sib, (f"SIBLING FENCE: {len(sib)} rows are MATH-500 template siblings (e.g. row {sib[0][0]} ~ MATH-500 #{sib[0][2]} "
                 f"at {sib[0][1]:.3f}); drop them from {out_jsonl}")
print("[diet-stage] SIBLING FENCE PASS", flush=True)
cat = lambda k: np.concatenate([p[3][k][p[2]] for p in parts])
# TOKMASK + SENT
s1, _, m1, o1 = PH.tokenize(out_jsonl)
tm, se = cat("tokmask"), cat("sent")
assert np.array_equal(m1.astype(np.uint8), tm.astype(np.uint8)), "TOKMASK GATE FAIL"
sv = np.stack([PH.sent_indices(r["text"], o1[i], m1[i]) for i, r in enumerate(s1)])
assert np.array_equal(sv.astype(se.dtype), se), "SENT GATE FAIL"
print("[diet-stage] TOKMASK + SENT gates PASS", flush=True)
# GOLD (sliced arrays land on disk-backed memmaps; the gate runs in row chunks)
tmp = f".cache/diet_stage_tmp_{out}"; os.makedirs(tmp, exist_ok=True)
G = {}
for k in keys:
    a = cat(k)
    G[k] = np.lib.format.open_memmap(f"{tmp}/{k}.npy", "w+", a.dtype, a.shape); G[k][:] = a; G[k].flush(); del a
CH = 4000
for a in range(0, N, CH):
    g1 = PH.build_gold(s1[a:a + CH], o1[a:a + CH])
    for k in keys:
        assert k[2:] in g1, f"GOLD GATE FAIL: build_gold no longer emits {k}"
        assert np.array_equal(g1[k[2:]], G[k][a:a + CH]), f"GOLD GATE FAIL: {k} rows {a}..{a + CH}"
    del g1
print(f"[diet-stage] GOLD gate PASS: {len(keys)} arrays bit-identical over {N} rows", flush=True)
# STATES
S = np.lib.format.open_memmap(PH.STATES_NPY.format(split=out), "w+", np.float16, (N, PH.T_ALG, PH.H_TRUNK))
k0 = 0
for split, src, idx, z in parts:
    src_s = np.load(PH.STATES_NPY.format(split=split), mmap_mode="r")
    for a in range(0, len(idx), 1000):
        S[k0 + a:k0 + a + len(idx[a:a + 1000])] = src_s[idx[a:a + 1000]]
    k0 += len(idx)
S.flush(); del S
np.savez(PH.STATES_NPZ.format(split=out), tokmask=tm, sent=se, mix_sha=mix_sha16(out_jsonl), **G)
shutil.rmtree(tmp)
print(f"[diet-stage] STAGED {out}: {N} rows, states sliced, stamped {mix_sha16(out_jsonl)}", flush=True)
