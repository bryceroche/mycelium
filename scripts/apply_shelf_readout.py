"""THE SHELF READOUT (2026-09-12, the word; "the graph is time"): the readout
reads an attention over EVERY breath's state (per slot; query from the final
state; a learned per-breath logit bias) instead of the final state alone — the
shallow-wide edge from the one loss to every breath (Karpathy's hop-distance
argument applied to the loop; never a per-breath loss, the deep-supervision
grave). A ROAD (all readout traffic passes through the mix; sever door
"shelf"); birth ~= the final state: ALG_SHELF_B0 logits on the last breath,
the content term ~0 (small q/k init). Idempotent; HEAD_PATH env for a copy."""
import os, sys
p = os.environ.get("HEAD_PATH", "scripts/phase1_algebra_head.py"); s = open(p).read()
if "_shelf_readout" in s:
    print("[apply] already applied"); sys.exit(0)
def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:80]); s = s.replace(old, new)
rep('''                           "s5", "nb2", "t1"))   # the balanced generation (+ T1, 2026-09-12)
''', '''                           "s5", "nb2", "t1", "shelf"))   # the balanced generation (+ T1, the shelf readout, 2026-09-12)
# THE SHELF READOUT (2026-09-12, the word): ALG_SHELF=1 reads the answer from
# an attention over every breath's state (per slot, query = the final state,
# + a learned per-breath logit bias) — the shallow-wide edge from the one
# loss to every breath. ALG_SHELF_D = the q/k width; ALG_SHELF_B0 = the
# last breath's logit at birth (the birth read picks it: 3/5/7).
ALG_SHELF = int(os.environ.get("ALG_SHELF", "0"))
ALG_SHELF_D = int(os.environ.get("ALG_SHELF_D", "64"))
ALG_SHELF_B0 = float(os.environ.get("ALG_SHELF_B0", "5"))
''')
rep('''    if ALG_T1:
        assert ALG_T1 % 2 == 1, "ALG_T1 = an odd tap count"
''', '''    if ALG_SHELF:
        _rS = np.random.RandomState(seed + 7331)
        p["sh_q"] = t((_rS.randn(H_W, ALG_SHELF_D) * 0.01 / math.sqrt(H_W)).astype(np.float32))
        p["sh_k"] = t((_rS.randn(H_W, ALG_SHELF_D) * 0.01 / math.sqrt(H_W)).astype(np.float32))
        _sb = np.zeros(K_B, np.float32); _sb[-1] = ALG_SHELF_B0
        p["sh_b"] = t(_sb)                                    # the last breath preferred at birth
    if ALG_T1:
        assert ALG_T1 % 2 == 1, "ALG_T1 = an odd tap count"
''')
rep('''    _s_final = breaths[-1]
    if _GTAP is not None and "final" in _GTAP:
        _s_final = _s_final + _GTAP["final"]
''', '''    _s_final = breaths[-1]
    if ALG_SHELF and "sh_q" in p and "shelf" not in _SEVER and len(breaths) > 1:
        _s_final = _shelf_readout(p, breaths)    # THE SHELF READOUT: every breath on the road to the loss
    if _GTAP is not None and "final" in _GTAP:
        _s_final = _s_final + _GTAP["final"]
''')
rep('''def _t1_conv(p, x, tokmask):
''', '''def _shelf_readout(p, breaths):
    """THE SHELF READOUT (2026-09-12): per slot, softmax over the K breath
    states of (q(final) . k(state_k) / sqrt(D) + b_k); the read = the
    weighted sum of the states themselves (no value map: the states ARE
    the shelf). A road; birth ~= the final state (b_last = ALG_SHELF_B0)."""
    S = breaths[0].stack(*breaths[1:], dim=1)               # (B, K, LT, HW)
    K = int(S.shape[1])
    assert int(p["sh_b"].shape[0]) == K, f"shelf bias sized {p['sh_b'].shape[0]} for {K} breaths"
    q = breaths[-1] @ p["sh_q"]                             # (B, LT, D)
    k = S @ p["sh_k"]                                       # (B, K, LT, D)
    e = (q.unsqueeze(1) * k).sum(-1) / math.sqrt(float(p["sh_q"].shape[1])) + p["sh_b"].reshape(1, K, 1)
    a = e.softmax(1)                                        # (B, K, LT)
    return (a.unsqueeze(-1) * S).sum(1)                     # (B, LT, HW)


def _t1_conv(p, x, tokmask):
''')
open(p, "w").write(s); print("[apply] the shelf readout applied to", p)
