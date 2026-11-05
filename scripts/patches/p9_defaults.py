"""P9 DEFAULTS (ledger 2026-11-05, P9 PASS; pinned 11-04: "on PASS the default switch is staged as a patch, applied between chains"): the
GSM8K benches read the 8M Tree-descending reader's states (RB8, distill_reader rb8x4) by default: gsmb_rb8_states.npy +
gsmb_role_feats_rb8.npy. Staged, not edited in place: lfgq re-reads these files. MBR-8M states stay reachable by
RD_STATES=gsmb_mbr8_states.npy (pair kept). Asserts each anchor once. Usage: p9_defaults.py ROOT  (ROOT holds scripts/; dry runs pass a copy)
"""
import os, sys

R = os.path.join(sys.argv[1], "scripts")


def edit(f, pairs):
    p = os.path.join(R, f); s = open(p).read()
    for a, b in pairs:
        assert s.count(a) == 1, f"{f}: anchor not found exactly once: {a[:80]!r}"
        s = s.replace(a, b)
    open(p, "w").write(s); print(f"[p9] {f}: {len(pairs)} edit(s)")


edit("gsm_trunk_reader.py", [
    ('    st = os.environ.get("RD_STATES", "gsmb_mbr8_states.npy")\n    if st in',
     '    # P9 (ledger 11-05): the default is the 8M Tree-descending reader RB8 (BT4 TEST step 0.1649 vs 0.1532, bar -0.010)\n'
     '    st = os.environ.get("RD_STATES", "gsmb_rb8_states.npy")\n    if st in'),
])
edit("gsm_breath_reader.py", [
    ('    st = os.environ.get("RD_STATES", "gsmb_mbr8_states.npy")                                # P8 (ledger 11-04): the 8M reader\n'
     '    pair = {"gsmb_mbr8_states.npy": "gsmb_role_feats_mbr8.npy", ',
     '    st = os.environ.get("RD_STATES", "gsmb_rb8_states.npy")                                 # P9 (ledger 11-05): the RB8 reader\n'
     '    pair = {"gsmb_rb8_states.npy": "gsmb_role_feats_rb8.npy", "gsmb_mbr8_states.npy": "gsmb_role_feats_mbr8.npy", '),
])
