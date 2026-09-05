"""apply_mass_thread.py — LIGHT THE MASK HEAD'S TWO DARK SENSES
(2026-09-05, mask-head round 2). Round 1's verdict (ledger): +0.0044
wild for the learner WITH the atlas and domain-mass ports zero-fallback
— two of four designed senses dark. This patch threads BOTH; C (the
atlas feed) is FOLDED INTO B (the mass thread) because every anchor
overlaps (forward signature, _bs_ctx, the fixed-buffer block, the
batch-feed site, _quick_val, loop_val). Two independent env gates:

ALG_MH_MASS=1 — THE MASS THREAD. alt2_fact_buf already computes per-var
domain-mass inside its ping call and DISCARDS it (only the contradiction
check reads it). Approach chosen: EXTEND, not re-ping — v0/v1/dispatcher
gain an optional `mass_out=None` (an (B, K_VARS) array filled in place;
None = byte-identical legacy path, dispatcher passes it through so the
seam-vector A/B stays symmetric). A second ping pass was the cheaper
patch but a worse organ: two pings can disagree mid-refactor (meter-
divergence law) and double the measured CPU seam cost for nothing.
Semantics: default fill = m+1 (the full 0..m domain — "nothing known"),
overwritten with GAC-propagated domain sizes for vars the ping covers;
contradiction/exception rows keep the full-domain fill (the bridge's
silence convention). Banked at mask-prep beside FACTS (frozen, same
vintage, same parse), normalized mass/301 -> [0,1] (values<=300 law),
fed as a fixed (batch, K_VARS, 1) buffer -> forward(mh_mass=...) ->
ctx["mh_mass"] (the port already reads it, zero-fallback).

ALG_MH_ATLAS=1 — THE ATLAS FEED. consult()-by-similarity needs realized
per-breath states; the FUSED trainer never materializes them mid-graph
(states live inside the jitted walk). HONEST SIMPLIFICATION (the
mission's sanctioned fallback): feed each row its OWN CLASS's per-step
trajectory page — class known at mask-prep time from gen metadata via
mycelium.step_atlas.atlas_class (single-source labeler; the miner uses
the same organ). We bank a per-row INDEX into the small (C+1, K_STEPS,
H_W) page table (zero page = class absent from the atlas), assemble
(batch, K_STEPS, H_W) per batch, and breath_step slices its own page
statically per kb (the loop is unrolled; kb is a python int — JIT-safe)
-> ctx["mh_atlas_traj"], falling back to the existing ctx["mh_atlas"]
per-breath port for seam drivers. Consult-by-similarity is the
READ-TIME UPGRADE (the step trainer realizes states per breath and can
call consult at seams); a dark port lit with the true class's
trajectory is exactly the conditioning the spec wants for TRAINING.
Atlas loads go through the LOUD DOOR (load_atlas + era stamp) against
the RESEARCH manifest (env MH_ATLAS_MANIFEST, default
.cache/RESEARCH_MANIFEST.json — the deployed GENERATION.json is not
this artifact's era anchor). ALG_MH_ATLAS=1 with the atlas file missing
is a HARD ERROR, not a silent skip (no-silent-fallbacks law).

loop_val.py is patched in the SAME transaction (prose-promotions law:
the chain's read legs toggle these envs with the arm — a reader that
ignored them would measure the ON arm with dark ports and lie).

BOTH ENVS UNSET = byte-identical: mass_out defaults None, MASSB/ATLAS
stay None, forward's new kwargs default None, ctx values None — every
new branch collapses to today's exact graph (the chain's eq gate
verifies A/B/C pre/post).

--check: builds both would-be results, ast-parses them, writes NOTHING.
"""
import ast
import sys

CHECK = '--check' in sys.argv
ANCHORS = []


def note(desc):
    ANCHORS.append(desc)


def sub(s, old, new, n=1, desc=""):
    assert s.count(old) == n, \
        f"anchor MISSING/NOT-UNIQUE (want {n}, have {s.count(old)}): {desc}"
    note(f"{desc} (x{n})")
    return s.replace(old, new, n)


# ===========================================================================
# FILE 1: scripts/phase1_algebra_head.py
# ===========================================================================
fn1 = 'scripts/phase1_algebra_head.py'
s = open(fn1).read()
n_lines0 = s.count('\n')

assert 'mass_out' not in s and 'ALG_MH_MASS' not in s \
    and 'mh_atlas_traj' not in s, \
    "mass/atlas thread already present — refuse (idempotence guard)"

# --- 1. alt2_fact_buf family: optional mass_out (fill-in-place) -----------
s = sub(s, "def _alt2_fact_buf_v0(onp, se_np, n_vars_arr, m_arr, theta=0.9):",
        "def _alt2_fact_buf_v0(onp, se_np, n_vars_arr, m_arr, theta=0.9,\n"
        "                      mass_out=None):",
        1, "v0 signature")
s = sub(s, "def _alt2_fact_buf_v1(onp, se_np, n_vars_arr, m_arr, theta=0.9):",
        "def _alt2_fact_buf_v1(onp, se_np, n_vars_arr, m_arr, theta=0.9,\n"
        "                      mass_out=None):",
        1, "v1 signature")

MASS_INIT = (
    "    if mass_out is not None:\n"
    "        # THE MASS THREAD (apply_mass_thread.py, 2026-09-05): default\n"
    "        # fill = m+1 (full 0..m domain — nothing known); ping rows\n"
    "        # overwrite below. Contradiction keeps the fill (silence).\n"
    "        mass_out[:] = np.asarray(m_arr, np.float64)[:, None] + 1.0\n")
s = sub(s, '    buf = np.zeros((B, K_VARS, 4), np.float32)\n\n    def _sig(x):',
        '    buf = np.zeros((B, K_VARS, 4), np.float32)\n'
        + MASS_INIT + '\n    def _sig(x):',
        1, "v0 mass_out default fill")
s = sub(s, '    buf = np.zeros((B, K_VARS, 4), np.float32)\n    has_dup = "dup" in onp',
        '    buf = np.zeros((B, K_VARS, 4), np.float32)\n'
        + MASS_INIT + '    has_dup = "dup" in onp',
        1, "v1 mass_out default fill")

s = sub(s,
        '            facts, mass, _r = ping(nv, facs, int(m_arr[bi]))\n'
        '            if mass is None:                       # contradiction: silence\n'
        '                continue\n',
        '            facts, mass, _r = ping(nv, facs, int(m_arr[bi]))\n'
        '            if mass is None:                       # contradiction: silence\n'
        '                continue\n'
        '            if mass_out is not None:               # the mass thread:\n'
        '                _nmv = min(len(mass), mass_out.shape[1])\n'
        '                mass_out[bi, :_nmv] = mass[:_nmv]  # post-GAC domain sizes\n',
        2, "ping sites gain mass capture (v0+v1, identical text)")

s = sub(s,
        '        except Exception:\n'
        '            buf[bi] = 0.0                          # per-item silence\n',
        '        except Exception:\n'
        '            buf[bi] = 0.0                          # per-item silence\n'
        '            if mass_out is not None:               # silence for mass too\n'
        '                mass_out[bi] = float(m_arr[bi]) + 1.0\n',
        2, "exception sites reset mass to full-domain (v0+v1)")

s = sub(s, "def alt2_fact_buf(onp, se_np, n_vars_arr, m_arr, theta=0.9):",
        "def alt2_fact_buf(onp, se_np, n_vars_arr, m_arr, theta=0.9,\n"
        "                  mass_out=None):",
        1, "dispatcher signature")
s = sub(s, "    return fn(onp, se_np, n_vars_arr, m_arr, theta=theta)",
        "    return fn(onp, se_np, n_vars_arr, m_arr, theta=theta,\n"
        "              mass_out=mass_out)",
        1, "dispatcher pass-through (seam-vector A/B stays symmetric)")

# --- 2. breath_step: the trajectory page port ------------------------------
s = sub(s,
        '        _mh_ap = ctx.get("mh_atlas")\n'
        '        if _mh_ap is None:\n'
        '            _mh_ap = (cur * 0.0).detach()\n',
        '        _mh_ap = ctx.get("mh_atlas")\n'
        '        if _mh_ap is None and ctx.get("mh_atlas_traj") is not None:\n'
        '            # ATLAS TRAJECTORY PORT (apply_mass_thread.py,\n'
        '            # 2026-09-05): (B, K_STEPS, H_W) per-row class pages;\n'
        '            # kb is a python int (the breath loop is unrolled) so\n'
        '            # this slice is static per jitted step. Page kb feeds\n'
        '            # breath kb (page 0 = intake, never consumed here —\n'
        '            # breath_step runs kb>=1). Consult-by-similarity is\n'
        '            # the read-time upgrade (seam drivers set "mh_atlas").\n'
        '            _mh_ap = ctx["mh_atlas_traj"][:, kb:kb + 1, :]\n'
        '        if _mh_ap is None:\n'
        '            _mh_ap = (cur * 0.0).detach()\n',
        1, "breath_step: mh_atlas_traj static per-kb page slice")

# --- 3. forward: two new kwargs, threaded into _bs_ctx ---------------------
s = sub(s,
        "def forward(p, trunk, tokmask, sent, slot_mask=None, revoke=None, "
        "tail=None, drop=None, anchor=None, amask=None, gmod=None, "
        "pmask=None, lsent=None, reg=None, fact_buf=None):",
        "def forward(p, trunk, tokmask, sent, slot_mask=None, revoke=None, "
        "tail=None, drop=None, anchor=None, amask=None, gmod=None, "
        "pmask=None, lsent=None, reg=None, fact_buf=None, mh_mass=None, "
        "mh_atlas_traj=None):",
        1, "forward signature (+mh_mass, +mh_atlas_traj; default None)")
s = sub(s,
        '                   "fact_buf": fact_buf,\n',
        '                   "fact_buf": fact_buf, "mh_mass": mh_mass,\n'
        '                   "mh_atlas_traj": mh_atlas_traj,\n',
        1, "_bs_ctx carries the two ports (None when unset = today)")

# --- 4. do_train: bank MASS + atlas index beside FACTS ---------------------
s = sub(s,
        '    MASKS = None\n'
        '    ALT2 = int(os.environ.get("ALG_ALT2", "0"))\n'
        '    FACTS = np.zeros((n, K_VARS, 4), np.float32) if ALT2 else None\n',
        '    MASKS = None\n'
        '    ALT2 = int(os.environ.get("ALG_ALT2", "0"))\n'
        '    FACTS = np.zeros((n, K_VARS, 4), np.float32) if ALT2 else None\n'
        '    # MASK HEAD round 2 (apply_mass_thread.py, 2026-09-05): the two\n'
        '    # dark senses. MASSB = per-var domain-mass banked at mask-prep\n'
        '    # (same vintage as FACTS), normalized /301 -> [0,1].\n'
        '    MH_MASS = int(os.environ.get("ALG_MH_MASS", "0"))\n'
        '    MASSB = np.zeros((n, K_VARS), np.float32) \\\n'
        '        if (ALT2 and MH_MASS) else None\n'
        '    ATLAS_TAB = ATLAS_IDX = None\n'
        '    if int(os.environ.get("ALG_MH_ATLAS", "0")):\n'
        '        # THE ATLAS FEED: per-row class trajectory pages (the\n'
        '        # research-manifest loud door; missing file = HARD error,\n'
        '        # never a silent dark port). Zero page for absent classes.\n'
        '        from mycelium.step_atlas import load_atlas, atlas_class\n'
        '        _amp = os.environ.get("MH_ATLAS_MANIFEST",\n'
        '                              ".cache/RESEARCH_MANIFEST.json")\n'
        '        _apath = os.environ.get("MH_ATLAS",\n'
        '                                ".cache/step_atlas_current.npz")\n'
        '        _atl = load_atlas(_apath, manifest_path=_amp)\n'
        '        _acls = {c: i for i, c in enumerate(_atl["classes"])}\n'
        '        _tab = np.ascontiguousarray(\n'
        '            _atl["means"].transpose(1, 0, 2)).astype(np.float32)\n'
        '        assert _tab.shape[1] >= K_B and _tab.shape[2] == H_W, \\\n'
        '            (_tab.shape, K_B, H_W)\n'
        '        ATLAS_TAB = np.concatenate(\n'
        '            [_tab, np.zeros((1,) + _tab.shape[1:], np.float32)])\n'
        '        ATLAS_IDX = np.array(\n'
        '            [_acls.get(atlas_class(smp.get("gen")), len(_acls))\n'
        '             for smp in samples], np.int64)\n'
        '        print(f"[mh-atlas] trajectory feed live: {_apath} "\n'
        '              f"classes={sorted(_acls)} zero-page rows="\n'
        '              f"{int((ATLAS_IDX == len(_acls)).sum())}/{n}",\n'
        '              flush=True)\n',
        1, "do_train: MASSB + ATLAS_TAB/ATLAS_IDX init (env-gated)")

s = sub(s,
        '                FACTS[sl] = alt2_fact_buf(_oa2, sent[sl_p], _nv2,\n'
        '                                          _ma2)[:len(sl)]\n',
        '                _mo2 = (np.zeros((len(sl_p), K_VARS), np.float32)\n'
        '                        if MASSB is not None else None)\n'
        '                FACTS[sl] = alt2_fact_buf(_oa2, sent[sl_p], _nv2,\n'
        '                                          _ma2,\n'
        '                                          mass_out=_mo2)[:len(sl)]\n'
        '                if MASSB is not None:\n'
        '                    # normalize [0,1]: /301 (values<=300 law)\n'
        '                    MASSB[sl] = np.clip(_mo2[:len(sl)] / 301.0,\n'
        '                                        0.0, 1.0)\n',
        1, "mask-prep: MASS banked beside FACTS (same ping, no re-ping)")

# --- 5. fixed buffers + batch feed + forward calls -------------------------
s = sub(s,
        '    b_fact = fix(np.zeros((batch, K_VARS, 4), np.float32), dtypes.float) \\\n'
        '        if ALT2 else None   # ALT2: fixed shape, ALWAYS fed (zeros when no\n'
        '                            # facts) — the jitted step\'s signature is stable\n',
        '    b_fact = fix(np.zeros((batch, K_VARS, 4), np.float32), dtypes.float) \\\n'
        '        if ALT2 else None   # ALT2: fixed shape, ALWAYS fed (zeros when no\n'
        '                            # facts) — the jitted step\'s signature is stable\n'
        '    b_mhm = fix(np.zeros((batch, K_VARS, 1), np.float32), dtypes.float) \\\n'
        '        if MASSB is not None else None   # mask-head mass port (b_fact idiom)\n'
        '    b_mha = fix(np.zeros((batch, ATLAS_TAB.shape[1], H_W), np.float32),\n'
        '                dtypes.float) if ATLAS_TAB is not None else None\n',
        1, "fixed buffers b_mhm/b_mha (the b_fact assign idiom)")

s = sub(s, 'fact_buf=b_fact)',
        'fact_buf=b_fact,\n'
        '                        mh_mass=b_mhm, mh_atlas_traj=b_mha)',
        3, "the three training forward calls feed both ports")

s = sub(s,
        '        if b_fact is not None:\n'
        '            b_fact.assign(Tensor(FACTS[idx], dtype=dtypes.float).contiguous()).realize()\n',
        '        if b_fact is not None:\n'
        '            b_fact.assign(Tensor(FACTS[idx], dtype=dtypes.float).contiguous()).realize()\n'
        '        if b_mhm is not None:\n'
        '            b_mhm.assign(Tensor(MASSB[idx][:, :, None],\n'
        '                                dtype=dtypes.float).contiguous()).realize()\n'
        '        if b_mha is not None:\n'
        '            b_mha.assign(Tensor(ATLAS_TAB[ATLAS_IDX[idx]],\n'
        '                                dtype=dtypes.float).contiguous()).realize()\n',
        1, "per-batch assigns (mirror b_fact exactly)")

# --- 6. _quick_val: the deployable cycle reads with lit ports --------------
s = sub(s,
        '    def _quick_val():\n'
        '        vs, vst, vtk, vg, vse = load_split_val\n',
        '    def _quick_val():\n'
        '        vs, vst, vtk, vg, vse = load_split_val\n'
        '        _vaidx = (np.array(\n'
        '            [_acls.get(atlas_class(smp.get("gen")), len(_acls))\n'
        '             for smp in vs], np.int64)\n'
        '            if ATLAS_TAB is not None else None)\n',
        1, "_quick_val: val-row atlas indices (same labeler organ)")

s = sub(s,
        '                _fbv = alt2_fact_buf(_ov, vse[sl_p], _nvv, _mav)\n'
        '                o = forward(p, _t1, _t2, _t3,\n'
        '                            slot_mask=Tensor(_mkv, dtype=dtypes.float),\n'
        '                            fact_buf=Tensor(_fbv, dtype=dtypes.float))\n',
        '                _mov = (np.zeros((len(sl_p), K_VARS), np.float32)\n'
        '                        if MASSB is not None else None)\n'
        '                _fbv = alt2_fact_buf(_ov, vse[sl_p], _nvv, _mav,\n'
        '                                     mass_out=_mov)\n'
        '                _vmh = (Tensor(np.clip(_mov / 301.0, 0.0, 1.0)\n'
        '                               [:, :, None].astype(np.float32),\n'
        '                               dtype=dtypes.float)\n'
        '                        if _mov is not None else None)\n'
        '                _vat = (Tensor(ATLAS_TAB[_vaidx[sl_p]],\n'
        '                               dtype=dtypes.float)\n'
        '                        if ATLAS_TAB is not None else None)\n'
        '                o = forward(p, _t1, _t2, _t3,\n'
        '                            slot_mask=Tensor(_mkv, dtype=dtypes.float),\n'
        '                            fact_buf=Tensor(_fbv, dtype=dtypes.float),\n'
        '                            mh_mass=_vmh, mh_atlas_traj=_vat)\n',
        1, "_quick_val: live mass + class pages on the val cycle")

ast.parse(s)
assert s.count('mass_out=None') == 3           # v0 + v1 + dispatcher
assert s.count('mh_atlas_traj') >= 5
assert s.count('ALG_MH_MASS') == 1 and s.count('ALG_MH_ATLAS') == 1

# ===========================================================================
# FILE 2: scripts/loop_val.py (same transaction — the read legs must see
# the envs the chain toggles, or the ON arm is read with dark ports)
# ===========================================================================
fn2 = 'scripts/loop_val.py'
s2 = open(fn2).read()
n2_lines0 = s2.count('\n')
assert 'ALG_MH_MASS' not in s2 and 'mh_atlas_traj' not in s2, \
    "loop_val already threaded — refuse (idempotence guard)"

s2 = sub(s2, 'n_ok = n_tot = 0\n',
         '# MASK HEAD round 2 (apply_mass_thread.py, 2026-09-05): the read\n'
         '# legs light the same two ports the trainer lit — toggled by the\n'
         '# SAME envs the chain sets per arm (trained-env law). Atlas loads\n'
         '# via the research-manifest loud door; env set + file missing =\n'
         '# hard error (no silent dark ports).\n'
         '_ATAB = _AIDX = None\n'
         'if int(os.environ.get("ALG_MH_ATLAS", "0")):\n'
         '    from mycelium.step_atlas import load_atlas, atlas_class\n'
         '    _atl = load_atlas(\n'
         '        os.environ.get("MH_ATLAS", ".cache/step_atlas_current.npz"),\n'
         '        manifest_path=os.environ.get("MH_ATLAS_MANIFEST",\n'
         '                                     ".cache/RESEARCH_MANIFEST.json"))\n'
         '    _acls = {c: i for i, c in enumerate(_atl["classes"])}\n'
         '    _tab = np.ascontiguousarray(\n'
         '        _atl["means"].transpose(1, 0, 2)).astype(np.float32)\n'
         '    _ATAB = np.concatenate(\n'
         '        [_tab, np.zeros((1,) + _tab.shape[1:], np.float32)])\n'
         '    _AIDX = np.array(\n'
         '        [_acls.get(atlas_class(s.get("gen")), len(_acls))\n'
         '         for s in vs], np.int64)\n'
         'n_ok = n_tot = 0\n',
         1, "loop_val: atlas preload + per-row class indices")

s2 = sub(s2, '    fact_t = None\n', '    fact_t = mass_t = None\n',
         1, "loop_val: mass_t default")

s2 = sub(s2,
         '        fb = alt2_fact_buf(_oa, vse[sl_p].astype(np.int32), _nv, _ma)\n'
         '        fact_t = Tensor(fb, dtype=dtypes.float)\n',
         '        _mo = (np.zeros((len(sl_p), K_VARS), np.float32)\n'
         '               if int(os.environ.get("ALG_MH_MASS", "0")) else None)\n'
         '        fb = alt2_fact_buf(_oa, vse[sl_p].astype(np.int32), _nv, _ma,\n'
         '                           mass_out=_mo)\n'
         '        fact_t = Tensor(fb, dtype=dtypes.float)\n'
         '        if _mo is not None:\n'
         '            mass_t = Tensor(np.clip(_mo / 301.0, 0.0, 1.0)\n'
         '                            [:, :, None].astype(np.float32),\n'
         '                            dtype=dtypes.float)\n',
         1, "loop_val: live mass from the same ping (no re-ping)")

s2 = sub(s2,
         '    o = forward(p, ts, tk, se, slot_mask=Tensor(mk, dtype=dtypes.float),\n'
         '                fact_buf=fact_t)\n',
         '    _mha_t = (Tensor(_ATAB[_AIDX[sl_p]], dtype=dtypes.float)\n'
         '              if _ATAB is not None else None)\n'
         '    o = forward(p, ts, tk, se, slot_mask=Tensor(mk, dtype=dtypes.float),\n'
         '                fact_buf=fact_t, mh_mass=mass_t, mh_atlas_traj=_mha_t)\n',
         1, "loop_val: masked pass feeds both ports")

ast.parse(s2)

print(f"[mass+atlas thread] {len(ANCHORS)} anchors OK "
      f"(phase1 +{s.count(chr(10)) - n_lines0} lines, "
      f"loop_val +{s2.count(chr(10)) - n2_lines0} lines):")
for i, desc in enumerate(ANCHORS, 1):
    print(f"  {i:2d}. {desc}")
if CHECK:
    print("[mass+atlas thread] --check: ast OK on both would-be results; "
          "NOTHING written")
else:
    open(fn1, 'w').write(s)
    open(fn2, 'w').write(s2)
    print("[mass+atlas thread] APPLIED (phase1_algebra_head.py + "
          "loop_val.py); run the eq gate (A/B/C, envs unset) before "
          "trusting — bit-identity is the bar")
