"""apply_census_hooks.py — the staged hook patch for the port census
(2026-09-01). RUN ONLY AFTER the router-rescue chain exits (the
running-chain law). Adds the inert-by-default _CENSUS capture to
forward(): state baseline + per-organ q_extra injections per breath.
Asserts on every anchor; ast-checks; prints the diff summary.
"""
import ast

fn = 'scripts/phase1_algebra_head.py'
s = open(fn).read()

# 1. the census global, beside the impulse hook's init
old = '''    global _IMP                     # the impulse hook (systems-ID probe;'''
new = '''    global _CENSUS                  # the port census hook (inert unless
    try: _CENSUS                    # port_census.py arms it — same
    except NameError: _CENSUS = None  # pattern as _IMP below)
    global _IMP                     # the impulse hook (systems-ID probe;'''
assert old in s, "anchor 1"
s = s.replace(old, new, 1)

# 2. state baseline + breath_emb capture at q_extra birth
old = '''            q_extra = cur + p["breath_emb"][kb].reshape(1, 1, -1)'''
new = '''            q_extra = cur + p["breath_emb"][kb].reshape(1, 1, -1)
            if _CENSUS is not None:
                _CENSUS.append((kb, "state", cur.realize().numpy()))
                _CENSUS.append((kb, "breath_emb",
                                p["breath_emb"][kb].realize().numpy()
                                .reshape(1, 1, -1)))'''
assert old in s, "anchor 2"
s = s.replace(old, new, 1)

# 3. notebook read capture (per-slot lane)
old = '''                    _rd = sum(_at[:, :, j:j + 1] * _nb[j] for j in range(len(_nb)))
                    q_extra = q_extra + _rd           # (B, L, H) — no blur'''
new = '''                    _rd = sum(_at[:, :, j:j + 1] * _nb[j] for j in range(len(_nb)))
                    q_extra = q_extra + _rd           # (B, L, H) — no blur
                    if _CENSUS is not None:
                        _CENSUS.append((kb, "notebook", _rd.realize().numpy()))'''
assert old in s, "anchor 3"
s = s.replace(old, new, 1)

# 4. garage read capture (gated value; _inj4 computed once, branches after)
old = '''                _inj4 = Tensor.cat(*_rds4, dim=-1) @ p["W_busr"]'''
new = '''                _inj4 = Tensor.cat(*_rds4, dim=-1) @ p["W_busr"]
                if _CENSUS is not None:
                    _CENSUS.append((kb, "garage",
                                    (_inj4 * p["bus_g"].reshape(1, 1, 1))
                                    .realize().numpy()))'''
assert old in s, "anchor 4"
s = s.replace(old, new, 1)

# 5. detwave feature capture (gated value)
old = '''                q_extra = q_extra + (_fe6 @ p["W_det"]) \\
                    * p["det_g"].reshape(1, 1, 1)'''
new = '''                _dinj = (_fe6 @ p["W_det"]) * p["det_g"].reshape(1, 1, 1)
                q_extra = q_extra + _dinj
                if _CENSUS is not None:
                    _CENSUS.append((kb, "detwave", _dinj.realize().numpy()))'''
assert old in s, "anchor 5"
s = s.replace(old, new, 1)

# 6. router bias capture (bank-side transmitter, gated)
old = '''                _rb7 = ((_cq7 @ p["W_ra"])
                        @ (waist @ p["W_rb"]).transpose(-2, -1)) / 8.0
                _rb_last = _rb7'''
new = '''                _rb7 = ((_cq7 @ p["W_ra"])
                        @ (waist @ p["W_rb"]).transpose(-2, -1)) / 8.0
                _rb_last = _rb7
                if _CENSUS is not None:
                    _CENSUS.append((kb, "router(bank)",
                                    (_rb7 * p["r_gain"].reshape(1, 1, 1))
                                    .realize().numpy()))'''
assert old in s, "anchor 6"
s = s.replace(old, new, 1)

open(fn, 'w').write(s)
ast.parse(s)
print("census hooks applied (6 anchors); ast OK — run eq_check before trusting")
