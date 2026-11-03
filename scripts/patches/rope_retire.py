"""THE RoPE RETIREMENT RULING (Bryce 2026-11-03, ledger): the machines move with the prose.

  gsm_trunk_reader.load : RD_STATES default -> gsmb_t9ns_states.npy (the T9-NS reader's GSM8K states); the legacy
                          states (Llama trunk / NMDB, RoPE) load only with RD_LEGACY=1 (banked comparisons, controls)
  gsm_breath_reader     : the role feats follow the states (rolef_name()); a states/role-feats mismatch fails loudly
  gsm_bt6_read / bt7    : use rolef_name()
  gsm_role_feats        : RF_STATES / RF_OUT defaults -> the T9-NS pair
  distill_reader        : a RoPE form (no DR_T6, or DR_T6=2 HY) runs only with DR_LEGACY_ROPE=1
Applied by unit lfrope after lfmbr22 and lfbt7b (both pinned on the old defaults) are done. Idempotence: asserts each anchor once.
"""
import os

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def edit(f, pairs):
    p = os.path.join(R, f); s = open(p).read()
    for a, b in pairs:
        assert s.count(a) == 1, f"{f}: anchor not found exactly once: {a[:80]!r}"
        s = s.replace(a, b)
    open(p, "w").write(s); print(f"[rope_retire] {f}: {len(pairs)} edit(s)")


edit("gsm_trunk_reader.py", [(
    '    return Dt, np.load(GD.V1.C(os.environ.get("RD_STATES", "gsmb_trunk_states.npy")), mmap_mode="r")',
    '    # THE RoPE RETIREMENT RULING (ledger 11-03): the reader is the T9-NS student; RoPE-era states need RD_LEGACY=1\n'
    '    st = os.environ.get("RD_STATES", "gsmb_t9ns_states.npy")\n'
    '    if st in ("gsmb_trunk_states.npy", "gsmb_student_states.npy") and os.environ.get("RD_LEGACY") != "1":\n'
    '        raise SystemExit(f"RD_STATES={st} is a RoPE-era reader (retired 11-03); set RD_LEGACY=1 for a banked comparison or control")\n'
    '    return Dt, np.load(GD.V1.C(st), mmap_mode="r")')])

edit("gsm_breath_reader.py", [
    ('SPLITS = TR.SPLITS\n',
     'SPLITS = TR.SPLITS\n\n\n'
     'def rolef_name():\n'
     '    """the role feats built on the same reader states (RoPE retirement ruling 11-03): a mismatch fails loudly"""\n'
     '    st = os.environ.get("RD_STATES", "gsmb_t9ns_states.npy")\n'
     '    pair = {"gsmb_t9ns_states.npy": "gsmb_role_feats_t9ns.npy", "gsmb_student_states.npy": "gsmb_role_feats.npy"}\n'
     '    rf = os.environ.get("BT_ROLEF", pair.get(st, ""))\n'
     '    assert rf and (st not in pair or rf == pair[st]), f"role feats {rf!r} were not built on the reader states {st!r}"\n'
     '    return rf\n'),
    ('    RF = np.load(GD.V1.C(os.environ.get("BT_ROLEF", "gsmb_role_feats.npy")), mmap_mode="r") if ROLE else None   # BT_ROLEF: ledger 11-02 CY',
     '    RF = np.load(GD.V1.C(rolef_name()), mmap_mode="r") if ROLE else None   # the role feats follow the states (ledger 11-03)')])

for f in ("gsm_bt6_read.py", "gsm_bt7_read.py"):
    edit(f, [('BR.RF = np.load(BR.GD.V1.C(os.environ.get("BT_ROLEF", "gsmb_role_feats.npy")), mmap_mode="r")',
              'BR.RF = np.load(BR.GD.V1.C(BR.rolef_name()), mmap_mode="r")')])

edit("gsm_role_feats.py", [
    ('os.environ.get("RF_OUT", "gsmb_role_feats.npy")', 'os.environ.get("RF_OUT", "gsmb_role_feats_t9ns.npy")'),
    ('os.environ.get("RF_STATES", "gsmb_student_states.npy")', 'os.environ.get("RF_STATES", "gsmb_t9ns_states.npy")')])

edit("distill_reader.py", [(
    'T9 = os.environ.get("DR_T6") == "3"',
    'assert (os.environ.get("DR_T6") in ("1", "3")) or os.environ.get("DR_LEGACY_ROPE") == "1", \\\n'
    '    "RoPE retired (ledger 11-03): the reader runs on the T9 clock (DR_T6=3); RoPE forms (no DR_T6 / DR_T6=2) need DR_LEGACY_ROPE=1"\n'
    'T9 = os.environ.get("DR_T6") == "3"')])
