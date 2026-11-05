"""UNIT-BIND (ledger 2026-11-05, Bryce: "yes for unit bind"): gsm_breath_reader.py gains BT_UNIT. The tinygrad engine needs no edit: the
unit block rides two existing linear roads (role, cfp) whose widths it reads off the parameter dict.

  BT_UNIT=1  every number token's unit (scripts/gsm_units.py, the OP-DIAG heuristic) is a product of base words; per problem, per batch,
             each base word draws a FRESH random phase vector theta (PU = 32 planes), as the name IDs do: a unit's identity is
             unlearnable, only sameness is. A unit's phase = sum of exponent x theta(word) (a rate X/Y = theta_X - theta_Y, "1" = 0).
             Tokens: the role road carries [role feats | cos | sin | kind one-hot (none, unknown, dimensionless, unit)].
             Slots: the executor's write-back computes each committed slot's unit by the same algebra (x adds, / subtracts, + / - keep
             A's), so X/Y x Y = X lands on theta_X exactly; the certificate road carries [cert | the same block].
  BT_UNIT=2  the placebo: every unit-bearing token and slot draws an INDEPENDENT phase (kinds kept): same features, no sameness.
The unit phases come from a generator seeded by the batch rng's STATE (read, never consumed), so the batches and masks are the control's.
BT_UNIT=0 leaves every tensor and every rng draw as before. Asserts each anchor once. Usage: unit_bind.py ROOT  (dry runs pass a copy)
"""
import os, sys

R = os.path.join(sys.argv[1], "scripts")


def edit(f, pairs):
    p = os.path.join(R, f); s = open(p).read()
    for a, b in pairs:
        assert s.count(a) == 1, f"{f}: anchor not found exactly once: {a[:80]!r}"
        s = s.replace(a, b)
    open(p, "w").write(s); print(f"[ub] {f}: {len(pairs)} edit(s)")


edit("gsm_breath_reader.py", [
    ('from mycelium import rotor_clock as CK\n',
     'from mycelium import rotor_clock as CK\nimport gsm_units as GU\n'),
    ('T6OFF = os.environ.get("BT_T6OFF") == "1"',
     'UNIT = os.environ.get("BT_UNIT", "0")                # (ledger 11-05 UNIT-BIND) 1 = units phase-bound per problem; 2 = the placebo\n'
     'PU = 32; NUF = 2 * PU + 4 if UNIT != "0" else 0     # the unit block: cos | sin | kind one-hot (none, unknown, dimensionless, unit)\n'
     'assert UNIT == "0" or ROLE, "UNIT-BIND rides the role road (BT_ROLE=1)"\n'
     'T6OFF = os.environ.get("BT_T6OFF") == "1"'),
    ('def step_type(op, a, b):',
     'def unit_feats(U, KD, th, rnd=None):\n'
     '    """(UNIT-BIND) per row -> (n, NUF): cos | sin of the unit phase (zeros unless known) | kind one-hot; th = word -> phase vector;\n'
     '    rnd (n, PU): the placebo\'s independent phases for unit-bearing rows"""\n'
     '    f = np.zeros((len(U), NUF), np.float32)\n'
     '    for i, (u, kd) in enumerate(zip(U, KD)):\n'
     '        f[i, 2 * PU + kd] = 1\n'
     '        if kd < 2: continue\n'
     '        a = rnd[i] if rnd is not None and kd == 3 else sum((e * th[w] for w, e in u.items()), np.zeros(PU))\n'
     '        f[i, :PU] = np.cos(a); f[i, PU:2 * PU] = np.sin(a)\n'
     '    return f\n\n\n'
     'def step_type(op, a, b):'),
    ('cf = np.zeros((B, K, 3), np.float32)', 'cf = np.zeros((B, K, 3 + NUF), np.float32)'),
    ('        n, vals = e["n"], e["vals"]; sv = [None] * K; sl = [0] * K\n',
     '        n, vals = e["n"], e["vals"]; sv = [None] * K; sl = [0] * K; su = [None] * K   # su: (UNIT) each slot\'s unit\n'),
    ('            cf[b, k] = TR.cert(sv[k])\n',
     '            cf[b, k, :3] = TR.cert(sv[k])\n'
     '            if NUF:   # (UNIT-BIND) the slot\'s unit by the phase algebra, from its operands\' units (tokens, earlier slots, constants = 1)\n'
     '                ua = [(e["uU"][i] if i < n else None) if kd_ == "t" else (su[i] if committed[b, i] else None) if kd_ == "s" else {}\n'
     '                      for kd_, i in (A, Bo)]\n'
     '                su[k] = GU.combine(op, *ua); kd = 1 if su[k] is None else 2 if not su[k] else 3\n'
     '                cf[b, k, 3:] = unit_feats([su[k]], [kd], e["uth"], e["urs"][k:k + 1] if UNIT == "2" else None)[0]\n'),
    ('            s.role = nn.Linear(14, D, bias=False) if ROLE else None',
     '            s.role = nn.Linear(14 + NUF, D, bias=False) if ROLE else None'),
    ('s.cfp = nn.Linear(3, D)', 's.cfp = nn.Linear(3 + NUF, D)'),
    ('        t["role"] = torch.as_tensor(r)\n',
     '        t["role"] = torch.as_tensor(r)\n'
     '    if NUF:   # (UNIT-BIND) fresh word phases per problem from a generator seeded by the batch rng\'s state (read, never consumed)\n'
     '        ur = np.random.default_rng((rng.bit_generator.state["state"]["state"], 1105))\n'
     '        ux = np.zeros(t["pad"].shape + (NUF,), np.float32)\n'
     '        for b, it in enumerate(its):\n'
     '            U, KD = GU.token_units(it); n = len(it["toks"])\n'
     '            th = {w: ur.random(PU) * 2 * math.pi for w in sorted({w for u in U if u for w in u})}\n'
     '            rt, rs = ur.random((n, PU)) * 2 * math.pi, ur.random((K, PU)) * 2 * math.pi   # (placebo) independent phases\n'
     '            ux[b, :n] = unit_feats(U, KD, th, rt if UNIT == "2" else None)\n'
     '            E[b]["uU"], E[b]["uth"], E[b]["urs"] = U, th, rs\n'
     '        t["role"] = torch.cat([t["role"], torch.as_tensor(ux)], -1)\n'),
])
