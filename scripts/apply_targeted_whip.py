"""THE TARGETED WHIP / THE MELT (2026-09-13, the wheel's third form): on a
certified refusal, the minimal unsatisfiable core names the slots that cannot
coexist; their CONTENT planes are melted (replaced by amp x the slot's norm of
seeded unit noise; amp 0 = zeroed; the clock planes kept) at the state entering
the next breath, so the machine must RE-DERIVE them from the sentence (the
grounding roads re-read every breath). ALG_WHEEL_MELT=<amp> (unset = bit-identical).
Read time: _wheel_turn writes state["wheel_melt"]; training: the walker's melt
bank. Idempotent; HEAD_PATH / ST_PATH env for copies."""
import os, sys
hp = os.environ.get("HEAD_PATH", "scripts/phase1_algebra_head.py"); s = open(hp).read()
sp = os.environ.get("ST_PATH", "scripts/step_trainer.py"); t = open(sp).read()
def rep(src, old, new):
    assert src.count(old) == 1, (src.count(old), old[:80]); return src.replace(old, new)
if "_WHEEL_MELT" not in s:
    s = rep(s, '''_WHEEL = None     # THE STEERING WHEEL at read time (apply_wheel_read.py): None = no wheel
''', '''_WHEEL = None     # THE STEERING WHEEL at read time (apply_wheel_read.py): None = no wheel
# THE MELT (2026-09-13, the targeted whip): ALG_WHEEL_MELT=<amp> — the core's
# slots have their content planes replaced (amp x slot norm of seeded unit
# noise; 0 = zeroed; the clock kept) at the state entering the next breath,
# so they are RE-DERIVED from the sentence. None = off (bit-identical).
_WHEEL_MELT = (float(os.environ["ALG_WHEEL_MELT"]) if os.environ.get("ALG_WHEEL_MELT") not in (None, "") else None)


def _melt(cur, m):
    """cur: (B, LT, HW); m: (B, LT) 1.0 on the slots to melt. On those slots the
    content dims become _WHEEL_MELT x ||slot|| x unit seeded noise (0 -> zeros);
    the clock dims are untouched; other slots are untouched."""
    B, LT, HW = (int(x) for x in cur.shape)
    n = _whip_noise(B, LT, HW)
    g = _polar_sink()[2] if ALG_POLAR else 1.0
    n = n * g
    nn = (n * n).sum(-1, keepdim=True).sqrt() + 1e-6
    nc = (cur * cur).sum(-1, keepdim=True).sqrt()
    fresh = n / nn * nc * _WHEEL_MELT                 # the melted content (amp 0 -> zeros)
    mm = m.reshape(B, LT, 1)
    return cur + mm * (fresh - cur * g)               # content replaced on melted slots; clock (g=0) kept
''')
    s = rep(s, '''    if _WHIP_K and kb == _WHIP_K:
        cur = _whip_kick(cur)            # THE WHIP: the state entering breath kb, kicked
''', '''    if _WHIP_K and kb == _WHIP_K:
        cur = _whip_kick(cur)            # THE WHIP: the state entering breath kb, kicked
    _wm = state.get("wheel_melt")
    if _WHEEL_MELT is not None and _wm is not None:
        cur = _melt(cur, _wm)            # THE MELT: the core's slots, re-derived this breath
        if _WHEEL is not None:
            state["wheel_melt"] = None   # consumed (the read-time wheel rewrites it per breath)
''')
    s = rep(s, '''    bias = _np.zeros((B, 1, L_TOT, T), _np.float32)
    beta = float(_WHEEL.get("beta", 3.0)); mode = _WHEEL.get("mode", "union")
''', '''    bias = _np.zeros((B, 1, L_TOT, T), _np.float32)
    melt = _np.zeros((B, L_TOT), _np.float32)          # THE MELT's mask for the next breath
    beta = float(_WHEEL.get("beta", 3.0)); mode = _WHEEL.get("mode", "union")
''')
    s = rep(s, '''        turned += 1
        core_slots = [parse[k]["_slot"] for k in r["core"]]
        sents = {}
''', '''        turned += 1
        core_slots = [parse[k]["_slot"] for k in r["core"]]
        for j in core_slots:
            melt[b, j] = 1.0
        sents = {}
''')
    s = rep(s, '''    _WHEEL.setdefault("turned", []).append((kb, turned, B))
    return _Tw(bias, dtype=_dw.float) if turned else None
''', '''    _WHEEL.setdefault("turned", []).append((kb, turned, B))
    state["wheel_melt"] = (_Tw(melt, dtype=_dw.float) if (turned and _WHEEL_MELT is not None) else None)
    return _Tw(bias, dtype=_dw.float) if (turned and beta != 0.0) else None
''')
    open(hp, "w").write(s); print("[apply] the melt applied to", hp)
else:
    print("[apply] head already applied")
if "melt_bank" not in t:
    t = rep(t, '''        self.wheel_bank = ([fix((B, 1, LT, T)) for _ in range(self.K_B - 2)]
                           if self.wheel else [])
''', '''        self.wheel_bank = ([fix((B, 1, LT, T)) for _ in range(self.K_B - 2)]
                           if self.wheel else [])
        self.melt_bank = ([fix((B, LT)) for _ in range(self.K_B - 2)]
                          if self.wheel else [])          # THE MELT's per-breath mask (2026-09-13)
''')
    t = rep(t, '''                "wheel_bias": (self.wheel_bank[k - 2]
                               if (self.wheel and k >= 2) else None),
''', '''                "wheel_bias": (self.wheel_bank[k - 2]
                               if (self.wheel and k >= 2) else None),
                "wheel_melt": (self.melt_bank[k - 2]
                               if (self.wheel and k >= 2 and getattr(self.H, "_WHEEL_MELT", None) is not None) else None),
''')
    t = rep(t, '''                bias, turned = wheel_bias(H, onp, self.fat_np, se_np, nv, ma,
                                          self.wheel_beta, self.wheel_mode,
                                          self.workers, self.LT)
                self.put(self.wheel_bank[k - 1], bias)
''', '''                bias, turned, melt = wheel_bias(H, onp, self.fat_np, se_np, nv, ma,
                                                self.wheel_beta, self.wheel_mode,
                                                self.workers, self.LT)
                self.put(self.wheel_bank[k - 1], bias)
                self.put(self.melt_bank[k - 1], melt)
''')
    t = rep(t, '''    bias = np.zeros((B, 1, LT, T), np.float32); turned = 0
''', '''    bias = np.zeros((B, 1, LT, T), np.float32); turned = 0
    melt = np.zeros((B, LT), np.float32)                 # THE MELT's mask
''')
    t = rep(t, '''        turned += 1
        parse = parses[b]
        slots = [parse[k]["_slot"] for k in core]
''', '''        turned += 1
        parse = parses[b]
        slots = [parse[k]["_slot"] for k in core]
        for j in slots:
            melt[b, j] = 1.0
''')
    t = rep(t, '''            bias[b, 0, j, :] = np.where(np.isin(se_np[b], list(want)), beta, 0.0)
    return bias, turned
''', '''            bias[b, 0, j, :] = np.where(np.isin(se_np[b], list(want)), beta, 0.0)
    return bias, turned, melt
''')
    open(sp, "w").write(t); print("[apply] the melt bank applied to", sp)
else:
    print("[apply] walker already applied")
