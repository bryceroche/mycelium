"""commit_xout_smoke.py — ORGAN-2's wire pull (2026-08-05). The reverse
gear's release dynamics under ALG_XOUT=1, all three arms. Checks:
(1) INIT-CLOSED: revoke=None + dump arm == rings v1 bit-for-bit;
(2) DUMP releases (mass ledger nonzero, revoked slot released);
(3) #150 ACCEPTANCE at smoke grade: graded arm's trajectory beats
    rate-matched pure decay (the resisting term is real, the fork is
    three arms not two);
(4) ELASTIC leaks without a trigger (self-resetting);
(5) two-terminal law: backward through the release path, W_cmt live."""
import os, sys
os.environ["ALG_BREATH"] = "3"; os.environ["ALG_RINGS"] = "1"
os.environ.setdefault("ALG2", "1"); os.environ.setdefault("ALG_FTYPES", "8")
os.environ.setdefault("ALG_HW", "512"); os.environ.setdefault("ALG_DUP", "1")
os.environ.setdefault("ALG_WIDE", "1")
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
from tinygrad import Tensor, dtypes
from phase1_algebra_head import build_params, forward, loss_fn, L_FAC, K_VARS, T_ALG, N_DIG

rng = np.random.default_rng(103850)
p = build_params(0)
p["W_cmt_b"].assign(Tensor(np.full(1, 2.0, np.float32))).realize()  # commit
for t in p.values(): t.requires_grad = True                # pressure ON (σ≈.88)
B = 1
st = Tensor(rng.standard_normal((B, T_ALG, 2048)).astype(np.float32))
msk = Tensor(np.ones((B, T_ALG), np.float32))
snt = Tensor(np.zeros((B, T_ALG), np.int32), dtype=dtypes.int)
sm = Tensor(np.ones((B, L_FAC, L_FAC), np.float32))
rv_np = np.zeros((B, L_FAC), np.float32); rv_np[0, 0] = 1.0  # revoke slot 0
rv = Tensor(rv_np)

def run(xout, arm, revoke):
    os.environ["ALG_XOUT"] = str(xout); os.environ["ALG_XARM"] = arm
    o = forward(p, st, msk, snt, slot_mask=sm, revoke=revoke)
    return {k: o[k].realize().numpy() for k in o if k in ("cmt_m", "xrel")}

base = run(0, "dump", None)
closed = run(1, "dump", None)
assert np.array_equal(base["cmt_m"], closed["cmt_m"]), "INIT-CLOSED VIOLATED"
assert float(closed["xrel"].max()) == 0.0
print(f"[smoke] init-closed PASS (mass p50 {np.median(base['cmt_m']):.4f}, xrel 0)")

d = run(1, "dump", rv)
assert float(d["xrel"][0, 0]) > 0, "dump released nothing"
assert float(d["xrel"][0, 1]) == 0.0, "dump leaked to unrevoked slot"
print(f"[smoke] dump: slot0 released ledger {float(d['xrel'][0,0]):.4f}, "
      f"final {float(d['cmt_m'][0,0]):.4f} (re-commit pressure re-adds)")

g = run(1, "graded", rv)
# rate-matched pure decay of what dump committed by the same breath: the
# resisting term (same-breath commit pressure) must beat it (#150)
m1 = float(base["cmt_m"][0, 0])           # unreleased final ≈ committed level
rate_matched = m1 * 0.5                    # one revoked breath, no resistance
gm = float(g["cmt_m"][0, 0])
assert gm > 1.2 * rate_matched, f"graded {gm:.3f} !> rate-matched {rate_matched:.3f}"
print(f"[smoke] graded: final {gm:.4f} vs rate-matched dumping {rate_matched:.4f} "
      f"— resisting term REAL (fork is three arms)")

e = run(1, "elastic", None)
assert float(e["xrel"].min()) > 0, "elastic leaked nothing without trigger"
assert float(np.median(e["cmt_m"])) < float(np.median(base["cmt_m"]))
print(f"[smoke] elastic: standing leak live (xrel p50 {np.median(e['xrel']):.4f}, "
      f"mass p50 {np.median(e['cmt_m']):.4f} < rings {np.median(base['cmt_m']):.4f})")

os.environ["ALG_XOUT"] = "1"; os.environ["ALG_XARM"] = "graded"
o = forward(p, st, msk, snt, slot_mask=sm, revoke=rv)
gold = {"presence": Tensor(np.ones((B, L_FAC), np.float32)),
        "is_lit_f": Tensor(np.zeros((B, L_FAC), np.float32)),
        "is_mod": Tensor(np.zeros((B, L_FAC), np.float32)),
        "is_sel": Tensor(np.zeros((B, L_FAC), np.float32)),
        "is_pct": Tensor(np.zeros((B, L_FAC), np.float32)),
        "is_fdiv": Tensor(np.zeros((B, L_FAC), np.float32)),
        "args": Tensor(np.zeros((B, L_FAC, K_VARS), np.float32)),
        "fspan": Tensor(np.ones((B, L_FAC, T_ALG), np.float32)),
        "vspan": Tensor(np.ones((B, K_VARS, T_ALG), np.float32)),
        "ftype": Tensor(np.zeros((B, L_FAC), np.int32), dtype=dtypes.int),
        "op": Tensor(np.zeros((B, L_FAC), np.int32), dtype=dtypes.int),
        "res": Tensor(np.zeros((B, L_FAC), np.int32), dtype=dtypes.int),
        "digits": Tensor(np.zeros((B, L_FAC, N_DIG), np.int32), dtype=dtypes.int),
        "sign": Tensor(np.zeros((B, L_FAC), np.float32)),
        "query": Tensor(np.zeros((B,), np.int32), dtype=dtypes.int)}
l = loss_fn(o, gold)
lv = float(l.numpy()); assert np.isfinite(lv)
l.backward()
gc = float(p["W_cmt"].grad.abs().max().numpy())
assert gc > 0, "COMMIT TERMINAL INERT THROUGH RELEASE PATH"
print(f"[smoke] loss={lv:.4f} |W_cmt.grad|max={gc:.3e} through the release path")
print("XOUT SMOKE: PASS — the reverse gear's dynamics are live, all three arms")
