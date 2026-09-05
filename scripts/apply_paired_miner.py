"""apply_paired_miner.py — THE SEVEN-PAGE NL ATLAS, miner-side
(2026-09-05, the paired atlas). STAGED patch to
scripts/mine_step_atlas.py (the file belongs to the round-2b chain's
world — running-invocation law: applied by the staged paired-atlas
chain after round-2b exits, then the chain re-mines).

WHAT IT ADDS: beside the slot-state (math) Welford bank, a SECOND bank
accumulates each item's per-breath READING — out["nl_all"] from the NL
tap (apply_nl_tap.py, same ALG_MINE_BREATHS env, same two-pass cycle):
the factor bank's head-averaged token attention, slot-averaged, pooled
over the ckpt's own 512-d waisted token states (waist = gelu(trunk @
waist_w + b) + sent_emb — in-graph; no re-projection). Same (breath,
class) keys via the SAME atlas_class organ, saved via save_atlas
nl_cells= into the SAME npz (nl_means/nl_vars/nl_counts) — one atlas
file, two charts, one class index, one era stamp.

DEPENDS ON: apply_paired_atlas.py (save_atlas nl_cells) and
apply_nl_tap.py (out["nl_all"]) — both asserted at apply time.

--check: builds the would-be result, ast-parses it, writes NOTHING.
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


fn = 'scripts/mine_step_atlas.py'
s = open(fn).read()
n_lines0 = s.count('\n')

assert 'cells_nl' not in s and 'nl_all' not in s, \
    "paired miner already present — refuse (idempotence guard)"
if not CHECK:   # dependency gates (at --check time the others may still
    # be staged; at APPLY time they must already be in the files)
    _head = open('scripts/phase1_algebra_head.py').read()
    assert 'nl_all' in _head, \
        "apply_nl_tap.py must be applied FIRST (out['nl_all'] missing)"
    _atlmod = open('mycelium/step_atlas.py').read()
    assert 'nl_cells' in _atlmod, \
        "apply_paired_atlas.py must be applied FIRST (save_atlas nl_cells)"

# --- 1. docstring: the NL chart clause -------------------------------------
s = sub(s,
        'CLASS LABEL: mycelium.step_atlas.atlas_class(row["gen"]) — the single-\n',
        'THE NL CHART (apply_paired_miner.py, 2026-09-05, the paired atlas):\n'
        'beside the slot-state pooling, each item\'s per-breath READING is\n'
        'banked: the factor bank\'s head-averaged token attention (slot-\n'
        'averaged to one distribution over tokens) pools the ckpt\'s OWN\n'
        'waisted token states -> (H_W,) per (item, breath). The NL tap\n'
        '(apply_nl_tap.py) computes it in-graph under the SAME env\n'
        '(out["nl_all"]); this miner only accumulates. Second Welford bank,\n'
        'same classes, same npz (nl_means/nl_vars/nl_counts), same era\n'
        'stamp — one file, two charts. The asymmetry the registration\n'
        'dissolved: the trunk ran once, but the READING evolves per breath\n'
        '(same rack, different pose).\n'
        '\n'
        'CLASS LABEL: mycelium.step_atlas.atlas_class(row["gen"]) — the single-\n',
        1, "docstring gains the NL-chart clause")

# --- 2. the second accumulator ---------------------------------------------
s = sub(s,
        '    cells = {}\n'
        '    n_done = 0\n',
        '    cells = {}\n'
        '    cells_nl = {}      # the second chart (the reading)\n'
        '    n_done = 0\n',
        1, "cells_nl accumulator")

# --- 3. realize the tap beside breaths_all ---------------------------------
s = sub(s,
        '        br = [b.realize().numpy() for b in o["breaths_all"]]\n'
        '        assert len(br) == K_STEPS, \\\n'
        '            f"breaths_all has {len(br)} steps, atlas wants {K_STEPS}"\n',
        '        br = [b.realize().numpy() for b in o["breaths_all"]]\n'
        '        assert len(br) == K_STEPS, \\\n'
        '            f"breaths_all has {len(br)} steps, atlas wants {K_STEPS}"\n'
        '        nl = [t.realize().numpy() for t in o["nl_all"]]\n'
        '        assert len(nl) == K_STEPS, \\\n'
        '            f"nl_all has {len(nl)} steps, atlas wants {K_STEPS}"\n',
        1, "realize nl_all (the NL tap) beside breaths_all")

# --- 4. accumulate both charts under one key -------------------------------
s = sub(s,
        '                # POOLING (see module docstring): mean over the 24 slots\n'
        '                cells[key].add(br[s_id][bi].mean(0).astype(np.float64))\n',
        '                # POOLING (see module docstring): mean over the 24 slots\n'
        '                cells[key].add(br[s_id][bi].mean(0).astype(np.float64))\n'
        '                if key not in cells_nl:\n'
        '                    cells_nl[key] = StepWelford(H_W)\n'
        '                # NL: attention-pooled waisted token state (the tap\n'
        '                # already pooled in-graph; accumulate as-is)\n'
        '                cells_nl[key].add(nl[s_id][bi].astype(np.float64))\n',
        1, "Welford both charts per (breath, class)")

# --- 5. save both charts into the one npz ----------------------------------
s = sub(s,
        '    path = save_atlas(cells, H_W, path=ATLAS_OUT,\n'
        '                      manifest_path=RESEARCH_MANIFEST)\n',
        '    path = save_atlas(cells, H_W, path=ATLAS_OUT,\n'
        '                      manifest_path=RESEARCH_MANIFEST,\n'
        '                      nl_cells=cells_nl)\n',
        1, "save_atlas carries nl_cells (one file, two charts)")

# --- 6. the per-class report shows both banks ------------------------------
s = sub(s,
        '        counts = [cells[(s, cls)].n if (s, cls) in cells else 0\n'
        '                  for s in range(K_STEPS)]\n'
        '        print(f"[mine-atlas]   {cls:12s} n/step={counts}")\n',
        '        counts = [cells[(s, cls)].n if (s, cls) in cells else 0\n'
        '                  for s in range(K_STEPS)]\n'
        '        print(f"[mine-atlas]   {cls:12s} n/step={counts}")\n'
        '        ncounts = [cells_nl[(s, cls)].n if (s, cls) in cells_nl\n'
        '                   else 0 for s in range(K_STEPS)]\n'
        '        assert ncounts == counts, \\\n'
        '            f"paired charts disagree on {cls}: {ncounts} vs {counts}"\n'
        '        print(f"[mine-atlas]   {cls:12s} nl n/step={ncounts}")\n',
        1, "report + paired-count assert (one pass, two charts, same n)")

ast.parse(s)
assert s.count('cells_nl') >= 5

print(f"[paired miner] {len(ANCHORS)} anchors OK "
      f"(miner +{s.count(chr(10)) - n_lines0} lines):")
for i, desc in enumerate(ANCHORS, 1):
    print(f"  {i:2d}. {desc}")
if CHECK:
    print("[paired miner] --check: ast OK on the would-be result; "
          "NOTHING written")
else:
    open(fn, 'w').write(s)
    print("[paired miner] APPLIED (scripts/mine_step_atlas.py); re-mine "
          "(MSA_N=4096) to bank the two-chart atlas")
