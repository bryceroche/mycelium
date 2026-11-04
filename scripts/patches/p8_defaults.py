"""P8 DEFAULTS (ledger 2026-11-04, P8 PASS; pinned 11-03: "on PASS, switch the bench defaults in the same commit"): the GSM8K benches
read the 8M reader's states (MBR-8M, distill_reader_mbr2x4) by default: gsmb_mbr8_states.npy + gsmb_role_feats_mbr8.npy.
Staged, not edited in place: lfbt8/lfw128/lfbt9 re-read these files (all three pin RD_STATES explicitly, so the switch is
behavior-neutral for them, but the law is the law). Applied by unit lfp8def once lfbt9 has applied bt9_idskip.py and launched
both seeds. T9-NS states stay reachable by RD_STATES=gsmb_t9ns_states.npy (pair kept). Asserts each anchor once.
"""
import os

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def edit(f, pairs):
    p = os.path.join(R, f); s = open(p).read()
    for a, b in pairs:
        assert s.count(a) == 1, f"{f}: anchor not found exactly once: {a[:80]!r}"
        s = s.replace(a, b)
    open(p, "w").write(s); print(f"[p8] {f}: {len(pairs)} edit(s)")


edit("gsm_trunk_reader.py", [
    ('    st = os.environ.get("RD_STATES", "gsmb_t9ns_states.npy")\n    if st in',
     '    # P8 (ledger 11-04): the default is the 8M reader MBR-8M (ALL IN ON 8M; BT4 TEST step parity 0.1535 vs 0.1590, bar -0.010)\n'
     '    st = os.environ.get("RD_STATES", "gsmb_mbr8_states.npy")\n    if st in'),
])
edit("gsm_breath_reader.py", [
    ('    st = os.environ.get("RD_STATES", "gsmb_t9ns_states.npy")\n    pair = {"gsmb_t9ns_states.npy": "gsmb_role_feats_t9ns.npy", ',
     '    st = os.environ.get("RD_STATES", "gsmb_mbr8_states.npy")                                # P8 (ledger 11-04): the 8M reader\n'
     '    pair = {"gsmb_mbr8_states.npy": "gsmb_role_feats_mbr8.npy", "gsmb_t9ns_states.npy": "gsmb_role_feats_t9ns.npy", '),
])
