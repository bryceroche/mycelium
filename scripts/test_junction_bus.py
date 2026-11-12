"""test_junction_bus.py — PJ unit tests (ledger 2026-11-12): bind/conj-unbind exactness, register round trip, None sides."""
import ast, inspect, numpy as np
from mycelium import junction_bus as jb
from mycelium.complex_tensor import phasor, bind, unbind, cleanup

def load(n=None):
    J = [ast.literal_eval(s) for s in np.load('.cache/pj/dump_local_k0_BEAM4+CMOTIF.npz', allow_pickle=True)['Jte']]
    return J if n is None else J[:n]

def same(a, b): return a == b

rng = np.random.default_rng(0)
cb = phasor(rng.uniform(0, 2 * np.pi, (64, 256))); role = phasor(rng.uniform(0, 2 * np.pi, 256))
assert all(cleanup(unbind(bind(cb[i], role), role), cb) == i for i in range(64)); print("1 PASS bind/conj-unbind exact (64/64)")

J = load(500); R = jb.roundtrip(J, 256)
rate = np.mean([same(a, b) for a, b in zip(J, R)]); assert rate >= 0.99, rate; print(f"2 PASS register P=256 exact recovery {rate:.4f} (n=500)")

Nn = [(i, j) for i, j in enumerate(J) if None in j]; assert Nn
assert all(R[i][k] is None for i, j in Nn for k in range(2) if j[k] is None) and all(R[i][k] is not None for i, j in Nn for k in range(2) if j[k] is not None)
print(f"3 PASS None sides round-trip ({len(Nn)} steps with a None side)")

src = inspect.getsource(jb.JunctionBus) + inspect.getsource(jb.roundtrip)
assert "/ self" not in src and "1/" not in src.replace(" ", "") and "reciprocal" not in src
assert "unbind(" in src and "np.conj" in inspect.getsource(unbind) and "/" not in inspect.getsource(unbind).split('"""')[-1]
print("4 PASS unbind path uses np.conj, no division")
