"""junction_bus.py — PJ (ledger 2026-11-12): each step's volatile junction (the class path L1/L2/L3 of its two
operand sides) encoded into complex phase-bound registers and recovered by CONJUGATE unbinding + codebook cleanup
(never 1/z). Register = sum_{role r, level l} role[r]*level[l]*code_l[class]  +  sum_r role[r]*level[3]*PRESENT
(the presence marker; a None side binds no class terms and no marker). Whole-problem mode multiplies each step's
register by slot[j] and superposes. Leaf strings are never encoded: decode passes them through from the template.
Presence rule: Re<unbound, PRESENT>/P is ~1 when the side is present, ~0 (noise ~sqrt(K/P)) when absent; threshold = 0.5
of the clean signal."""
import numpy as np
from mycelium.complex_tensor import phasor, bind, unbind, cleanup

PRESENCE_THR = 0.5
MAX_SLOTS = 64

class JunctionBus:
    def __init__(self, P, seed=0, levels=(2048, 256, 32)):
        rng = np.random.default_rng(seed); self.P, self.levels = P, tuple(levels)
        ph = lambda *s: phasor(rng.uniform(0, 2 * np.pi, s))
        self.code = [ph(n, P) for n in levels]
        self.role, self.level, self.slot = ph(2, P), ph(len(levels) + 1, P), ph(MAX_SLOTS, P)
        self.present = ph(P)

    def encode_step(self, ab):
        z = np.zeros(self.P, np.complex64)
        for r, side in enumerate(ab):
            if side is None: continue
            for l in range(len(self.levels)):
                z += bind(bind(self.code[l][side[1 + l]], self.level[l]), self.role[r])
            z += bind(bind(self.present, self.level[-1]), self.role[r])
        return z

    def decode_step(self, z, ab_template=None):
        out = []
        for r in range(2):
            u = unbind(z, self.role[r])
            pres = (unbind(u, self.level[-1]) * np.conj(self.present)).real.mean()
            if pres < PRESENCE_THR: out.append(None); continue
            cls = [int(cleanup(unbind(u, self.level[l]), self.code[l])) for l in range(len(self.levels))]
            leaf = ab_template[r][0] if ab_template is not None and ab_template[r] is not None else None
            out.append([leaf] + cls)
        return out

    def encode_problem(self, abs_list):
        return sum(bind(self.encode_step(ab), self.slot[j]) for j, ab in enumerate(abs_list)).astype(np.complex64)

    def decode_problem(self, z, n_steps, templates):
        return [self.decode_step(unbind(z, self.slot[j]), templates[j]) for j in range(n_steps)]

def roundtrip(J, P, mode="register", seed=0, n_steps=1):
    bus = JunctionBus(P, seed)
    if mode == "register": return [bus.decode_step(bus.encode_step(ab), ab) for ab in J]
    out = []
    for i in range(0, len(J), n_steps):
        g = J[i:i + n_steps]; out += bus.decode_problem(bus.encode_problem(g), len(g), g)
    return out
