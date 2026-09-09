"""apply_tokgate_census.py — THE PORT CENSUS learns to read THE TOKEN GATE
(2026-09-09; the reader half of scripts/apply_tok_cook.py, spec
docs/token_cooker_spec.md S3 / the twin's bar 2). A STAGED patch on the
READER, not the head: it anchors into scripts/port_census.py and under
--check writes NOTHING and PRINTS A PREVIEW of the section it adds, on a
synthetic accumulator, so the shape of the reading can be judged before
anything is applied. PC_TARGET may point at a copy (rehearsal).

WHAT THE HEAD RECORDS (apply_tok_cook.py, behind the _CENSUS hook, at
every LOOP breath on which the gate ran):

  tokgate      (B, L_TOT, T)  the gate itself — the mask head's per-slot
                              distribution over the prompt's real tokens.
                              TOKEN band: no baseline is recorded in that
                              band, so its `rel` column reads n/a exactly
                              as the router's does (never divide across
                              bands), and the reader prints its rms.
  tokgate_kl   (B, 1, 1)      the organ's OWN mean KL from uniform per
                              slot, one scalar per row. The reader QUOTES
                              it; it does not rebuild it (the
                              meter-divergence law).

WHAT THIS PATCH ADDS TO THE READER:

  (1) `tokgate_kl` is kept OUT of the organ table and out of the pairwise
      cosines. It is a DIVERGENCE, not an injection: its rms is not an
      amplitude, and a (1,)-vector cosine against another (1,)-vector is
      +-1 by construction — a number that would look like a finding and
      be an artifact. It gets its own section instead.
  (2) `tokgate` joins NO_GAIN: the gate rides no scalar gain (it REPLACES
      an attention rather than adding into a band), so its empty row in
      the gain ledger reads as a fact, not an omission — the alt21_s3/s4
      precedent.
  (3) A NEW SECTION, "TOKEN GATE — THE AIM", per breath:
        KL         the organ's mean KL from uniform per slot (nats)
        log n_tok  the uniform gate's own entropy, DERIVED FROM THE
                   RECORDED GATE (n_tok = the count of strictly positive
                   entries per slot: pads are EXACTLY 0.0 by
                   construction, the -1e4 addend underflows) — so the
                   reference is measured on the same rows, not assumed
        aim        KL / log n_tok in [0, 1): 0.000 = the uniform floor
                   the severance probe measured (wild 0.1536), 1.000
                   would be a delta on one token. THE COOKER'S JOB IS TO
                   MOVE THIS NUMBER, and the twin's bar 2 is read here.
        rms        the gate tensor's rms (token band, bare)

  Nothing else in the reader moves: the three organs the ledger quotes
  keep their accumulation, their rel column and their cosine grammar.

--check: asserts every anchor present and unique, builds the would-be
result, ast-parses it, runs the structural asserts, PRINTS THE PREVIEW,
and writes NOTHING.
"""
import ast
import os
import sys

fn = os.environ.get("PC_TARGET", 'scripts/port_census.py')
CHECK = '--check' in sys.argv
s = open(fn).read()
n_lines0 = s.count('\n')

assert 'tokgate' not in s, \
    "the token gate is already in the reader — refuse (idempotence)"
assert 'NO_GAIN = ("alt21_s3", "alt21_s4")' in s, \
    "this reader predates the organ pass (apply_census_organs2.py) — " \
    "anchor into port_census.py AS IT IS"

PATCHES = []


def patch(num, desc, old, new):
    PATCHES.append((num, desc, old, new))


patch(1, "the two token-gate accumulators, beside base_mag/base_shape",
      '''base_shape = {}   # baseline name -> kb -> shape   (the band signature)''',
      '''base_shape = {}   # baseline name -> kb -> shape   (the band signature)
tg_kl = {}        # kb -> [mean KL from uniform per slot]  (THE TOKEN GATE)
tg_ref = {}       # kb -> [log n_tok]  the uniform gate's own entropy,
                  # DERIVED from the recorded gate (pads are exactly 0.0)''')

patch(2, "collect the gate's KL and its uniform reference",
      '''        if organ in ("state", "state_slot", "state_hslot"):
            base_mag.setdefault(organ, {}).setdefault(kb, []).append(m)
            base_shape.setdefault(organ, {})[kb] = arr.shape''',
      '''        if organ in ("state", "state_slot", "state_hslot"):
            base_mag.setdefault(organ, {}).setdefault(kb, []).append(m)
            base_shape.setdefault(organ, {})[kb] = arr.shape
        if organ == "tokgate":
            # THE TOKEN GATE (apply_tok_cook.py, 2026-09-09): the
            # uniform reference, MEASURED on the same rows rather than
            # assumed — n_tok is the count of strictly positive entries
            # per slot, and pads are EXACTLY 0.0 by construction (the
            # -1e4 pad addend underflows in the softmax).
            _nt = (arr > 0.0).sum(-1)
            tg_ref.setdefault(kb, []).append(
                float(np.log(np.maximum(_nt, 1)).mean()))
        if organ == "tokgate_kl":
            tg_kl.setdefault(kb, []).append(float(arr.mean()))''')

patch(3, "keep tokgate_kl out of the organ table and the cosines",
      '''organs = sorted({o2 for (_, o2) in acc if o2 not in BASELINES})''',
      '''# THE TOKEN GATE's KL is a DIVERGENCE, not an injection: its rms is not
# an amplitude and a (1,)-vector cosine is +-1 by construction. It is read
# in its own section below, never in the injection table.
DIVERGENCES = ("tokgate_kl",)
organs = sorted({o2 for (_, o2) in acc
                 if o2 not in BASELINES and o2 not in DIVERGENCES})''')

patch(4, "tokgate joins NO_GAIN (it replaces an attention, it adds nothing)",
      '''NO_GAIN = ("alt21_s3", "alt21_s4")''',
      '''NO_GAIN = ("alt21_s3", "alt21_s4", "tokgate")
# `tokgate` (THE TOKEN COOKER, 2026-09-09) rides NO scalar gain at all —
# it does not add into a band, it REPLACES the slot->token attention on
# sealed rows. There is nothing to divide out and no pre-gain form to
# report; its empty gain-ledger row is a fact, like stations 3-4's.''')

patch(5, "the TOKEN GATE section (KL, its reference, the aim, the rms)",
      '''# ------------------------------------------- the 2026-09-01 reading, kept''',
      '''# ------------------------------------------------------- the token gate
if tg_kl:
    print("TOKEN GATE — THE AIM (apply_tok_cook.py, 2026-09-09). KL = the "
          "organ's OWN mean KL from uniform per slot (nats, quoted not "
          "rebuilt); log n_tok = the uniform gate's entropy, DERIVED from "
          "the recorded gate on the same rows; aim = KL / log n_tok in "
          "[0, 1): 0.000 IS the uniform floor the severance probe "
          "measured, 1.000 would be a delta on one token. The cooker's "
          "job is to move `aim`.")
    for kb in sorted(tg_kl):
        _kl = float(np.mean(tg_kl[kb]))
        _rf = float(np.mean(tg_ref.get(kb, [0.0]))) or float("nan")
        _rm = (float(np.mean(acc[(kb, "tokgate")][1]))
               if (kb, "tokgate") in acc else float("nan"))
        print(f"  b{kb}:  KL={_kl:>9.5f}  log n_tok={_rf:>7.4f}  "
              f"aim={_kl / _rf:>7.5f}  rms={_rm:.5g}")
    print("  [grammar] the gate is a DISTRIBUTION over the prompt's real "
          "tokens (exactly 0 on pads), not an injection: it has no band "
          "baseline, no gain and no pre-form. Read `aim`, not `rms`.")

# ------------------------------------------- the 2026-09-01 reading, kept''')

for num, desc, old, new in PATCHES:
    assert old in s, f"anchor {num} MISSING ({desc}) — read the file, adjust"
    assert s.count(old) == 1, f"anchor {num} NOT UNIQUE ({desc})"
    s = s.replace(old, new, 1)

ast.parse(s)

# ------------------------------------------------------ structural asserts
assert s.count('DIVERGENCES = ("tokgate_kl",)') == 1 and \
    'o2 not in DIVERGENCES' in s, \
    "the KL must be excluded from `organs` in exactly one place"
assert s.count('if organ == "tokgate":') == 1 and \
    s.count('if organ == "tokgate_kl":') == 1, \
    "each record is collected exactly once"
assert s.index('tg_kl = {}') < s.index('tg_kl.setdefault') < \
    s.index('if tg_kl:'), "declare, collect, then print"
assert '(arr > 0.0).sum(-1)' in s, \
    "n_tok must be DERIVED from the gate (pads are exactly 0.0), not assumed"
assert 'aim={_kl / _rf:>7.5f}' in s, "the aim column is the reading"
assert s.count('NO_GAIN = (') == 1 and '"tokgate")' in s, \
    "tokgate must join NO_GAIN"
assert 'tokgate' not in s[s.index('organs_c = [o2 for o2 in organs'):], \
    "the gate must not enter the cosine section by name (it rides in " \
    "`organs` for the rms table only; the KL never rides at all)"

print(f"[tokgate census] {len(PATCHES)} anchors OK "
      f"(+{s.count(chr(10)) - n_lines0} lines, 0 deleted):")
for num, desc, _o, _n in PATCHES:
    print(f"  {num}. {desc}")

# ------------------------------------------------------------- the PREVIEW
print("[tokgate census] PREVIEW of the new section (synthetic accumulator: "
      "a gate that is exactly uniform at b1 and progressively sharper, on "
      "a 100-token row) —")
import numpy as np                                             # noqa: E402
_T, _L = 100, 32
for _kb, _tau in ((1, 0.0), (3, 0.5), (6, 2.0)):
    _rs = np.random.RandomState(_kb)
    _g = np.exp(_tau * _rs.randn(8, _L, _T))
    _g = (_g / _g.sum(-1, keepdims=True)).astype(np.float32)
    _kl = float((np.log(_T) + (_g * np.log(np.maximum(_g, 1e-300))).sum(-1))
                .mean())
    _rf = float(np.log(np.maximum((_g > 0.0).sum(-1), 1)).mean())
    _rm = float(np.sqrt((_g ** 2).mean()))
    print(f"  b{_kb}:  KL={_kl:>9.5f}  log n_tok={_rf:>7.4f}  "
          f"aim={_kl / _rf:>7.5f}  rms={_rm:.5g}")
print("  [grammar] the gate is a DISTRIBUTION over the prompt's real "
      "tokens (exactly 0 on pads), not an injection: it has no band "
      "baseline, no gain and no pre-form. Read `aim`, not `rms`.")

if CHECK:
    print("[tokgate census] --check: ast OK on the would-be result; "
          "NOTHING written")
else:
    open(fn, 'w').write(s)
    print(f"[tokgate census] APPLIED ({fn}); ast OK")
