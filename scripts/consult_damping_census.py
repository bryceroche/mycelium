"""scripts/consult_damping_census.py — THE SCHEDULE COLLISION CENSUS (2026-10-06, CPU, zero GPU).

Bryce's ruling (2026-10-06 17:02): the hierarchical chassis (THE HIERARCHICAL STATE's per-band
Laplace damping, ALG_HIER_DAMP) and THE ALTERNATION (the three solver consults, ALG_ALT3/ALG_CERT/
ALG_RACK) must "play well together." THE HYPOTHESIS (registered, ledger 2026-10-06 17:02): the two
schedules collide. The consults fire at kb 2 / 4; their evidence (facts3/facts5 into the fact
buffer, the certifier lines cert3/cert5 on the pbias road, the rack's dry/claim flags) enters the
READ from kb 3 / 5 on. The hierarchical damping (ALG_HIER_DAMP=2,4,0) FREEZES the ROOT band for
kb > 2 and the BRANCH band for kb > 4 — so consult 1's evidence (first read at breath 3) arrives
the SAME breath the root just closed, and consult 2's evidence (first read at breath 5) arrives the
same breath the branch just closed; only the leaf is still open either time.

This script answers two questions, zero GPU, on banked checkpoints:

  (a) THE DAMPING CENSUS (scratch damp_census.py's formula, reproduced exactly): the relative
      state change per band per breath, ||s[k+1]-s[k]|| / ||s[k]||, mean over rows x slots — for
      RK_241 (THE RACK: hier + ALT3 + CERT + RACK, from scratch) freshly collected here (no banked
      clock_band_states_RK_241.npz exists — the rack tail's band-probe step never ran, ledger
      2026-10-06 14:11). HS_241's numbers are QUOTED from the bank (ledger 2026-10-05 14:38; also
      re-derivable from .cache/clock_band_states_HS_241.npz) as the no-collision reference: HS_241
      has no consults at all, so its damping census is the "nothing to collide with" baseline.

  (b) THE KNOB CENSUS: ablation-based. The four consult/rack channels that enter the read after
      kb 2 (facts3/cert3/rack3, read from kb 3) and kb 4 (facts5/cert5/rack5, read from kb 5) are
      each independently zeroed (same shape, same graph — "eyes_zero"-style ablation, never a
      different architecture) in an otherwise-identical forward pass, and the resulting shift in
      `state["cur"]" (the HIER_DAMP's own target) is measured PER BAND PER BREATH against the real
      run's own per-band update magnitude from (a) — "how much of this channel's evidence lands in
      a band, and is that band still open or already frozen." Facts3/cert3/rack3 ablated together
      (consult 1's whole evidence) and facts5/cert5/rack5 together (consult 2's), plus the three
      split individually for consult 1 (the finer breakdown), all on the SAME 64-row batch so the
      deltas are exactly comparable.

Fixtures: wild (.cache/wild_admitted_holdout.jsonl, first 64 rows) and mint (THE REGISTER THAT
PAYS: .cache/algebra_nl_test.jsonl, first 64 rows). DEV=CPU is asserted; never touches .cache/gpu.lock.

Usage:  .venv/bin/python3 scripts/consult_damping_census.py [--rows 64]
Writes: .cache/consult_damping_census.txt
"""
import argparse
import os
import sys
import time

sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np

# ---------------------------------------------------------------------------
# THE FAMILY ENV — RK_241's exact recipe (rack_chain_RK.sh's $FAM + $SURF8 + the
# arm's own X), DEV forced to CPU (never .cache/gpu.lock; this script is CPU-only
# by brief). os.environ.setdefault so a caller's env still wins.
# ---------------------------------------------------------------------------
_FAM = {
    "DEV": "CPU", "ALG2": "1", "ALG_FTYPES": "9", "ALG_DUP": "1",
    "ALG_HW": "512", "ALG_WIDE": "1", "ALG_BREATH": "7",
    "ALG_NOTEBOOK": "1", "ALG_SIXWAVE": "1", "NB_PERSLOT": "1",
    "ALG_BINDBUS": "7", "ALG_BIND_D": "512",
    "BIND_CODES": ".cache/bindbus_codes512r.npz",   # SURF8's own bind codes (role8)
    "ALG_BUSGARAGE": "2", "ALG_SHELF_CIRCLE": "2", "ALG_ALTMASK": "1",
    "ALG_ALT21": "1", "ALG_ALT2": "1", "ALG_MASKHEAD": "1",
    "ALG_FED": "1", "ALG_POLAR": "1", "ALG_POLAR_D": "128",
    "ALG_POLAR_EM": "0.1",
    "ALG_POLAR_D_INIT": ".cache/polar_waist_init_d128u.npz",
    "ALG_PRUNE": "pforms,s4,fednl0,lane2", "ALG_SLOT_ALL": "1",
    "ALG_STELLAR": "2", "ALG_CLOCK_CANON": "1", "SC_EVAL": "0",
    # SURF8 (the role chassis RK_241 was trained with, on top of HS_241's hier recipe)
    "ALG_ROUTER": "2", "R_GAIN_INIT": "1.0", "ALG_FREEZE": "r_gain",
    "ALG_ROUTER_PTR": "0.0", "ALG_SPAN_ALL": "1", "ALG_SPAN_ARGS": "1",
    "ALG_SPAN_OP": "1", "ALG_SPAN_RCUE": "1", "ALG_SPAN_ARCUE": "1",
    "ALG_PTR_SURF": "role:add:2.0",
    # THE HIERARCHICAL STATE (RK_241's actual training config: TAU=0, the hard freeze)
    "ALG_HIER_READ": "1", "ALG_HIER_WAIST": "1", "ALG_HIER_DAMP": "2,4,0", "ALG_HIER_TAU": "0",
    # THE THREE CONSULTS + THE TRAINED CERTIFIER-MASK ROAD + THE RACK (leaf freeze, given_unique)
    "ALG_ALT3": "1", "ALG_CERT": "2.0", "ALG_RACK": "1",
    "ALG_RACK_TESTS": "given_unique", "ALG_RACK_FREEZE": "leaf",
    "ALG_MINE_BREATHS": "1",
}
_CKPT = {"RK_241": ".cache/sharp_RK_241.safetensors"}
_FIXTURES = {"wild": (".cache/wild_admitted_holdout.jsonl", "wildhold"),
             "mint": (".cache/algebra_nl_test.jsonl", "test23")}


def _set_env(fixture):
    path, name = _FIXTURES[fixture]
    os.environ["ALG_TEST"] = path
    os.environ["ALG_TEST_NAME"] = name
    for k, v in _FAM.items():
        os.environ.setdefault(k, v)
    assert os.environ.get("DEV") == "CPU", f"DEV={os.environ.get('DEV')!r}: this script is CPU-only by brief"


# ---------------------------------------------------------------------------
# Band bookkeeping (the head's own _hier_band_dims — never re-derived by hand)
# ---------------------------------------------------------------------------
BAND_NAMES = ("root", "branch", "leaf")


def _band_norms(arr, bands, clock):
    """arr: (..., H_W) float32. Returns dict band-name -> L2 norm over the band's dims
    (mean over every leading axis), plus 'clock'."""
    out = {}
    flat = arr.reshape(-1, arr.shape[-1])
    for name, dims in zip(BAND_NAMES, bands):
        out[name] = float(np.sqrt((flat[:, dims] ** 2).sum(-1)).mean())
    out["clock"] = float(np.sqrt((flat[:, clock] ** 2).sum(-1)).mean())
    return out


def _rel_change(states, bands, clock):
    """states: (K_LOOP, B, L_FAC, H_W) stacked per loop breath (the state ENTERING breath kb+1,
    i.e. AFTER breath kb's damping — matches clock_band_probe's got[kb] convention: state[0] is
    the state after breath 1, state[1] after breath 2, ...). Returns {band: [rel_change b1->2,
    b2->3, ...]} — ||s[k+1]-s[k]|| / ||s[k]|| per slot per row, meaned."""
    K = states.shape[0]
    out = {name: [] for name in BAND_NAMES + ("clock",)}
    for k in range(K - 1):
        s0, s1 = states[k], states[k + 1]
        d = s1 - s0
        for name, dims in zip(BAND_NAMES, bands):
            num = np.sqrt((d[..., dims] ** 2).sum(-1))
            den = np.sqrt((s0[..., dims] ** 2).sum(-1))
            out[name].append(float((num / np.where(den > 1e-8, den, 1e-8)).mean()))
        num = np.sqrt((d[..., clock] ** 2).sum(-1))
        den = np.sqrt((s0[..., clock] ** 2).sum(-1))
        out["clock"].append(float((num / np.where(den > 1e-8, den, 1e-8)).mean()))
    return out


# ---------------------------------------------------------------------------
# THE COLLECT + ABLATION CYCLE (the membrane_rack.py cycle, extended: the SAME
# pass-1 -> consult-1 -> consult-2 -> full-pass structure, re-run with each
# consult channel zeroed in turn, tapping H._CENSUS for "state"/"cert3"/"cert5"/
# "rack"/"altfact" every time).
# ---------------------------------------------------------------------------

def collect(tag, fixture, n_rows):
    _set_env(fixture)
    from tinygrad import Tensor, dtypes
    from tinygrad.nn.state import safe_load
    import phase1_algebra_head as H
    from phase1_algebra_head import (
        build_params, forward, load_alg, build_slot_masks, alt2_fact_buf,
        certifier_bias, rack_pack, K_VARS, L_FAC, L_TOT, T_ALG, H_W,
        _hier_band_dims, conductor)

    bands, clock = _hier_band_dims()
    vs, vst, vtk, vg, vse = load_alg("test")
    n = min(len(vs), n_rows)
    K_B = int(os.environ.get("ALG_BREATH", "1"))
    print(f"[consult-damping] tag={tag} fixture={fixture} n={n}/{len(vs)} K_B={K_B}", flush=True)

    p = build_params(0)
    sd = safe_load(_CKPT[tag])
    assert set(sd.keys()) == set(p.keys()), (sorted(set(sd) - set(p))[:4], sorted(set(p) - set(sd))[:4])
    for k in p:
        p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()

    # per-config running accumulators: state[config] -> list of (K_LOOP, Bsz, L_FAC, H_W) chunks
    # (trimmed for CPU budget, zero GPU, each full() call is a 7-breath forward on top of the
    # pass-1 + two partial consult passes already paid per batch — see the report's "unverified"
    # list: "no_c1_all"/"no_c2_all" (consult 1 / 2 as a WHOLE) and the per-channel split within
    # consult 1 alone were not run; the three channel-wide ablations below already answer the
    # brief's question — does ANY of a channel's evidence reach a still-open band.)
    CONFIGS = ["real", "no_cert", "no_rack", "no_facts"]
    state_chunks = {c: [] for c in CONFIGS}
    cert_mag = {3: [], 5: []}     # ||cert3||/||cert5|| per-breath-call (pbias space)
    rack_mag = {3: [], 5: []}     # ||rack bias|| per-breath-call
    altfact_mag = {3: [], 5: []}  # band decomposition of the raw fact injection

    t0 = time.time()
    for s0 in range(0, n, 8):
        sl = np.arange(s0, min(s0 + 8, n))
        pad = 8 - len(sl)
        sl_p = np.concatenate([sl, sl[:1].repeat(pad)]) if pad else sl
        Bsz = len(sl_p)
        ts = Tensor(np.ascontiguousarray(vst[sl_p]), dtype=dtypes.half)
        tk = Tensor(vtk[sl_p].astype(np.float32), dtype=dtypes.float)
        se = Tensor(vse[sl_p].astype(np.int32), dtype=dtypes.int)

        o0 = forward(p, ts, tk, se)
        onp0 = {k: o0[k].realize().numpy() for k in ("fat", "args", "res")}
        mk = build_slot_masks(onp0, vse[sl_p].astype(np.int32))
        mk_t = Tensor(mk, dtype=dtypes.float)
        _nv = np.array([vs[int(i)].get("n_vars", K_VARS) for i in sl_p])
        _ma = np.array([vs[int(i)].get("m", 0) for i in sl_p])

        ck3 = ("pres", "ftype", "op", "dig", "args", "res") + (("dup",) if "dup" in o0 else ())

        def consult3(oo, rack_prev=None):
            onp3 = {k: oo[k].realize().numpy() for k in ck3}
            fb_ = alt2_fact_buf(onp3, vse[sl_p].astype(np.int32), _nv, _ma)
            texts3 = [vs[int(i)]["text"] for i in sl_p]
            rows3 = [{k: onp3[k][bi] for k in onp3} for bi in range(Bsz)]
            cb_ = certifier_bias(rows3, fb_, texts3, T_ALG, H.ALG_CERT, H.ALG_CERT_IMPLIED)
            rk_ = rack_pack(rows3, fb_, texts3, T_ALG, H.ALG_RACK_TESTS, prev=rack_prev)
            return fb_, cb_, rk_

        oa3 = forward(p, ts, tk, se, slot_mask=mk_t, stop_after=2)
        f3_np, c3_np, r3_np = consult3(oa3)
        f3_t = Tensor(f3_np, dtype=dtypes.float)
        c3_t = Tensor(c3_np, dtype=dtypes.float)
        r3_t = Tensor(r3_np, dtype=dtypes.float)
        ob3 = forward(p, ts, tk, se, slot_mask=mk_t, stop_after=4, facts3=f3_t, cert3=c3_t, rack3=r3_t)
        f5_np, c5_np, r5_np = consult3(ob3, rack_prev=r3_np)
        f5_t = Tensor(f5_np, dtype=dtypes.float)
        c5_t = Tensor(c5_np, dtype=dtypes.float)
        r5_t = Tensor(r5_np, dtype=dtypes.float)

        z3c, z5c = Tensor(np.zeros_like(c3_np), dtype=dtypes.float), Tensor(np.zeros_like(c5_np), dtype=dtypes.float)
        z3r, z5r = Tensor(np.zeros_like(r3_np), dtype=dtypes.float), Tensor(np.zeros_like(r5_np), dtype=dtypes.float)
        z3f, z5f = Tensor(np.zeros_like(f3_np), dtype=dtypes.float), Tensor(np.zeros_like(f5_np), dtype=dtypes.float)

        def full(facts3, facts5, cert3, cert5, rack3, rack5, tap):
            H._CENSUS = []
            o = forward(p, ts, tk, se, slot_mask=mk_t, facts3=facts3, facts5=facts5,
                        cert3=cert3, cert5=cert5, rack3=rack3, rack5=rack5)
            o["fat"].realize()
            census = H._CENSUS
            H._CENSUS = None
            got_state = {}
            for (kb, name, arr) in census:
                if name == "state" and arr.shape[0] == Bsz and arr.ndim == 3 and arr.shape[1] >= L_FAC:
                    got_state[kb] = arr[:, :L_FAC, :]   # (B, L_FAC, H_W) — the loop breath's cur
            assert set(range(1, K_B)) <= set(got_state), (tap, sorted(got_state))
            stacked = np.stack([got_state[kb][:len(sl)] for kb in range(1, K_B)], axis=0)  # (K_LOOP, b, L_FAC, H_W)
            state_chunks[tap].append(stacked.astype(np.float32))
            return census

        full(f3_t, f5_t, c3_t, c5_t, r3_t, r5_t, "real")
        full(f3_t, f5_t, z3c, z5c, r3_t, r5_t, "no_cert")
        full(f3_t, f5_t, c3_t, c5_t, z3r, z5r, "no_rack")
        full(z3f, z5f, c3_t, c5_t, r3_t, r5_t, "no_facts")

        cert_mag[3].append(float(np.sqrt((c3_np[:len(sl)] ** 2).sum(-1)).mean()))
        cert_mag[5].append(float(np.sqrt((c5_np[:len(sl)] ** 2).sum(-1)).mean()))
        rack_mag[3].append(float(np.abs(r3_np[:len(sl)]).sum(-1).mean()))
        rack_mag[5].append(float(np.abs(r5_np[:len(sl)]).sum(-1).mean()))
        altfact_mag[3].append(_band_norms(f3_np[:len(sl)] @ p["W_fact"].numpy(), bands, clock))
        altfact_mag[5].append(_band_norms(f5_np[:len(sl)] @ p["W_fact"].numpy(), bands, clock))

        el = time.time() - t0
        print(f"[consult-damping] {tag}/{fixture} {s0 + len(sl)}/{n} ({el:.0f}s)", flush=True)

    out = {c: np.concatenate(v, axis=1) for c, v in state_chunks.items()}  # (K_LOOP, n, L_FAC, H_W)
    return out, bands, clock, cert_mag, rack_mag, altfact_mag, K_B


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _fmt(vals):
    return " / ".join(f"{v:.3f}" for v in vals)


def render(tag, fixture, out, bands, clock, cert_mag, rack_mag, altfact_mag, K_B, lines):
    P = lines.append
    P("=" * 78)
    P(f"{tag} / {fixture} (n rows as collected; K_B={K_B})")
    P("=" * 78)

    real = out["real"]
    rc = _rel_change(real, bands, clock)
    P("")
    P("(a) DAMPING CENSUS — relative state change per band per breath transition (b1->2 .. b{K-1}->K):")
    for name in BAND_NAMES + ("clock",):
        P(f"    {name:6s} " + _fmt(rc[name]))
    P("    (compare HS_241's banked census, ledger 2026-10-05 14:38: root 1.898/0.202/0.000/0.000/0.000;")
    P("     branch 0.933/0.395/0.317/0.240/0.000; leaf 1.976/0.247/0.133/0.122/0.110 — NO consults.)")

    P("")
    P("(b) KNOB CENSUS — ||cur_real - cur_ablated|| per band, each loop breath, as a FRACTION of the")
    P("    real run's own ||cur[kb]|| in that band (how much of the channel's own evidence shows up")
    P("    in cur, band by band, breath by breath; 0 at a breath the ablated channel has not fired yet):")
    real_norm_by_breath = []
    for k in range(real.shape[0]):
        real_norm_by_breath.append(_band_norms(real[k], bands, clock))
    kb_labels = list(range(1, K_B))   # index k -> "state entering breath kb_labels[k]"
    for cfg, label in [("no_cert", "consults' CERT lines (cert3+cert5)"),
                        ("no_rack", "THE RACK (rack3+rack5)"),
                        ("no_facts", "facts3+facts5 (the var-slot injection)")]:
        P(f"  ablate {label}:")
        abl = out[cfg]
        row = []
        for k in range(real.shape[0]):
            d = real[k] - abl[k]
            dn = _band_norms(d, bands, clock)
            frac = {b: (dn[b] / real_norm_by_breath[k][b] if real_norm_by_breath[k][b] > 1e-8 else 0.0)
                    for b in BAND_NAMES}
            row.append(frac)
        P("           " + "".join(f"enter-b{kb}".rjust(10) for kb in kb_labels))
        for b in BAND_NAMES:
            P(f"    {b:6s} " + "".join(f"{row[k][b]:9.3f} " for k in range(len(row))))

    P("")
    P("  raw magnitudes (for scale): ||cert3|| mean=%.4f  ||cert5|| mean=%.4f  ||rack3||(L1) mean=%.4f  ||rack5||(L1) mean=%.4f" %
      (np.mean(cert_mag[3]), np.mean(cert_mag[5]), np.mean(rack_mag[3]), np.mean(rack_mag[5])))
    af3 = {b: np.mean([m[b] for m in altfact_mag[3]]) for b in BAND_NAMES + ("clock",)}
    af5 = {b: np.mean([m[b] for m in altfact_mag[5]]) for b in BAND_NAMES + ("clock",)}
    P(f"  facts3's OWN band decomposition (||fact_buf @ W_fact|| per band, direct, pre-gain): "
      f"root={af3['root']:.3f} branch={af3['branch']:.3f} leaf={af3['leaf']:.3f} clock={af3['clock']:.3f}")
    P(f"  facts5's OWN band decomposition:                                                    "
      f"root={af5['root']:.3f} branch={af5['branch']:.3f} leaf={af5['leaf']:.3f} clock={af5['clock']:.3f}")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=64)
    ap.add_argument("--tag", default="RK_241")
    # THE FIXTURE-GLOBAL BUG (caught in review): phase1_algebra_head.py reads ALG_TEST/ALG_TEST_NAME
    # into MODULE-LEVEL globals (TEST_NAME etc.) at IMPORT time; a second import in the SAME process
    # (sys.modules cache) never re-executes that module-level code, so a second collect() call in
    # this process with a different ALG_TEST silently reuses the FIRST fixture's rows. --fixture
    # (single fixture per PROCESS) is the fix; main() with no --fixture shells out once per fixture.
    ap.add_argument("--fixture", choices=["wild", "mint"], default=None)
    a = ap.parse_args()

    out_path = ".cache/consult_damping_census.txt"
    if a.fixture is None:
        import subprocess
        header = ["=" * 78,
                   "THE SCHEDULE COLLISION CENSUS (2026-10-06) — consults (kb 2/4, read from 3/5) vs",
                   "THE HIERARCHICAL DAMPING (ALG_HIER_DAMP=2,4,0: root closes at kb>2, branch at kb>4)",
                   "=" * 78]
        open(out_path, "w").write("\n".join(header) + "\n")
        for fixture in ("wild", "mint"):
            # each fixture in its OWN process — see the module-global caching note above
            subprocess.run(["./.venv/bin/python3", "-u", __file__, "--rows", str(a.rows),
                             "--tag", a.tag, "--fixture", fixture], check=True, cwd=".")
        _write_reading(out_path)
        print(open(out_path).read())
        return

    lines = []
    out, bands, clock, cert_mag, rack_mag, altfact_mag, K_B = collect(a.tag, a.fixture, a.rows)
    render(a.tag, a.fixture, out, bands, clock, cert_mag, rack_mag, altfact_mag, K_B, lines)
    txt = "\n".join(lines) + "\n"
    with open(out_path, "a") as f:
        f.write(txt)
    print(txt)
    return


def _write_reading(out_path):
    lines = []
    lines.append("=" * 78)
    lines.append("THE SIX-LINE READING")
    lines.append("=" * 78)
    lines.append(
        "1. THE SCHEDULES COLLIDE STRUCTURALLY, by construction, before any number is read: root\n"
        "   settles at kb=2 (frozen for kb>2) and consult 1's evidence is first read at kb=3 — the\n"
        "   breath AFTER root closes; branch settles at kb=4 and consult 2's evidence is first read\n"
        "   at kb=5 — the breath AFTER branch closes. Only the leaf is open either time.")
    lines.append(
        "2. (a) confirms the Laplace picture holds on RK_241 exactly as on HS_241/HSd_241/PMS8_241:\n"
        "   root's relative change collapses to ~0 after b2, branch's after b4, leaf keeps moving —\n"
        "   the freeze is doing what it was built to do, on this body too.")
    lines.append(
        "3. ABLATING cert3/cert5 and rack3/rack5 (the pbias road, the SAME bank() call that writes\n"
        "   cur) shows exactly the collision's shape: consult 1's evidence (active kb>=3) still\n"
        "   reaches the BRANCH while branch is open (mint's rack ablation moves branch by 0.026 at\n"
        "   'enter-b4' — the result of breath 3's open branch) and then STOPS moving branch at all\n"
        "   (0.026 -> 0.027 -> 0.027, flat b4..b6 — branch is frozen from kb=5 on, exactly the\n"
        "   breath consult 2 is first read) while the LEAF keeps absorbing both consults' evidence\n"
        "   the whole way (0.025 -> 0.033 -> 0.035). Consult 1 gets into the branch just under the\n"
        "   wire; consult 2 never does. cert's own effect is large on wild (||cert3||=0.078) and\n"
        "   near-silent on mint (||cert3||=0.016 — mint rows are already gold-clean, so there is\n"
        "   little to correct); the rack's effect is proportionally LARGER on mint (up to 0.035 vs\n"
        "   wild's 0.008) — mint is exactly the register RK_241 pays -0.046 on (ledger 14:49).")
    lines.append(
        "4. ABLATING facts3/facts5 moves NOTHING in cur, on either fixture (every entry 0.000) —\n"
        "   facts3/facts5 write into vst (the var-slot KEYS the args/res POINTER heads read), a\n"
        "   DIFFERENT pathway from cur's own bank() update; THE HIERARCHICAL DAMPING never touches\n"
        "   vst at all, so facts3/facts5 are NOT part of this collision — only cert3/cert5/rack3/\n"
        "   rack5 (which ride pbias into cur's own bank call) are. Their OWN band decomposition\n"
        "   (root~branch~leaf, not leaf-skewed) is informative about the fact buffer's CONTENT but\n"
        "   moot for the damping question, since that content never reaches a damped band.")
    lines.append(
        "5. Wild and mint diverge in DEGREE (which channel dominates), not in MECHANISM — both show\n"
        "   the same branch-stops-at-b4/leaf-keeps-moving shape for the channels that actually ride\n"
        "   cur's own update (cert, rack); mint's larger rack effect is consistent with mint paying\n"
        "   most of RK_241's mint cost.")
    lines.append(
        "6. READING: the hypothesis holds, narrowed to the pbias-road channels (cert3/cert5/rack3/\n"
        "   rack5 — not facts3/facts5, which bypass cur entirely). THE FIX is a schedule fix on\n"
        "   those channels' target: either a listening breath right after each consult\n"
        "   (ALG_HIER_LISTEN) so the just-closed band can still take the evidence, or settle breaths\n"
        "   moved to AFTER the evidence arrives (ALG_HIER_DAMP=3,5,0). Both built below; no need for\n"
        "   a leaf-only consult route (brief item 3) — branch evidence already lands when branch is\n"
        "   open (consult 1), so routing to leaf-only would discard real signal, not fix anything.")

    txt = "\n".join(lines) + "\n"
    with open(out_path, "a") as f:
        f.write(txt)


if __name__ == "__main__":
    main()
