"""apply_nl_tap.py — THE NL TAP (2026-09-05, the paired atlas). STAGED
patch to scripts/phase1_algebra_head.py (a live chain imports it —
running-invocation law: applied only by the staged paired-atlas chain
after round-2b exits).

THE FIND (why a patch at all): breath_step already computes the
per-breath token attention — `h_tok, fat_cur = bank(p["fq"], ...)` —
and THROWS IT AWAY (only door #10's CLOCK gate reads it). out["fat"] is
breath-0 only. ALG_MINE_BREATHS exposes breaths_all (slot states, the
COMMITMENT trajectory) but nothing of the READING. This tap banks the
reading:

  per breath k: the factor bank's head-averaged token attention
  (B, L_FAC, T), slot-averaged into ONE token distribution (B, T) —
  each row is already a distribution over live tokens, so the
  slot-mean IS the attention-weighted read — then pooled over the
  WAISTED token states: (B, T) @ waist (B, T, H_W) -> (B, H_W).

POOLING CHOICE (documented per the registration): waist is the ckpt's
own 512-d waisted token bank (gelu(trunk @ waist_w + b) + sent_emb),
already in-graph — NO re-projection through any W_enc is needed, and
slot-mean-then-pool equals pool-per-slot-then-mean exactly (linearity),
so the page is slot-permutation-stable like the math chart's.

EXPOSED (read-only, lazy tensors — realized only by miners/readers;
the _CENSUS discipline: zero graph change, inert unless armed; training
never sets these envs):
  ALG_MINE_BREATHS=1 -> out["nl_all"]  = K_STEPS x (B, H_W) NL states
                        out["nlat_all"] = K_STEPS x (B, T) token reads
                        (breath 0 = the same bank pass fst came from)
  ALG_MINE_BREATHS=1 or ALG_MH_XPRIOR=1 -> out["nl0"] (B, H_W): the
      breath-0 NL state for the cross-atlas prior. IDENTICAL in pass-1
      and pass-2 by construction: the fq bank pass has no cur/fact_buf/
      slot_mask dependence, so feed-time consumers may read it off the
      unmasked pass-1 forward they already run.

ENVS UNSET = byte-identical: nothing is computed, nothing is appended
(the chain's eq gate A/B/C verifies bit-identity pre/post).

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


fn = 'scripts/phase1_algebra_head.py'
s = open(fn).read()
n_lines0 = s.count('\n')

assert 'nl_all' not in s and 'ALG_MH_XPRIOR' not in s and '"nl0"' not in s, \
    "NL tap already present — refuse (idempotence guard)"

# --- 1. breath_step: bank the per-breath reading ---------------------------
s = sub(s,
        '    h_tok, fat_cur = bank(p["fq"], L_FAC, extra=q_extra,\n'
        '                          pbias=(_sync[0](kb) if _sync is not None\n'
        '                                 else None),\n'
        '                          rbias=_rb7)\n',
        '    h_tok, fat_cur = bank(p["fq"], L_FAC, extra=q_extra,\n'
        '                          pbias=(_sync[0](kb) if _sync is not None\n'
        '                                 else None),\n'
        '                          rbias=_rb7)\n'
        '    if int(os.environ.get("ALG_MINE_BREATHS", "0")):\n'
        '        # NL TAP (apply_nl_tap.py, 2026-09-05, the paired atlas):\n'
        '        # read-only capture of this breath\'s READING — head-avg\n'
        '        # token attention, slot-averaged to one distribution,\n'
        '        # pooling the SAME waist the bank read (attention-weighted\n'
        '        # mean over tokens -> (B, H_W)). Lazy tensors on state;\n'
        '        # realized only by miners/readers (the _CENSUS discipline:\n'
        '        # inert unless armed; training never sets this env).\n'
        '        _nlw = fat_cur.mean(1)                       # (B, T) read\n'
        '        state.setdefault("nl_all", []).append(\n'
        '            (_nlw.unsqueeze(1) @ waist).squeeze(1))  # (B, H_W)\n'
        '        state.setdefault("nlat_all", []).append(_nlw)\n',
        1, "breath_step: per-breath NL read banked beside the walk")

# --- 2. forward: assemble the seven-page reading + the prior's nl0 ---------
s = sub(s,
        '    if int(os.environ.get("ALG_MINE_BREATHS", "0")) and K_B > 1 and slot_mask is not None:\n'
        '        out["breaths_all"] = out_breaths\n',
        '    if int(os.environ.get("ALG_MINE_BREATHS", "0")) and K_B > 1 and slot_mask is not None:\n'
        '        out["breaths_all"] = out_breaths\n'
        '        # NL TAP (apply_nl_tap.py): the seven-page reading — breath\n'
        '        # 0 is the same fq bank pass fst came from (fat); breaths\n'
        '        # 1..K-1 were appended by breath_step under the same env.\n'
        '        _nl0w = fat.mean(1)\n'
        '        out["nl_all"] = ([(_nl0w.unsqueeze(1) @ waist).squeeze(1)]\n'
        '                         + ((_bs_state or {}).get("nl_all") or []))\n'
        '        out["nlat_all"] = ([_nl0w]\n'
        '                           + ((_bs_state or {}).get("nlat_all") or []))\n'
        '    if (int(os.environ.get("ALG_MINE_BREATHS", "0"))\n'
        '            or int(os.environ.get("ALG_MH_XPRIOR", "0"))):\n'
        '        # breath-0 NL state for the CROSS-ATLAS PRIOR — identical\n'
        '        # in pass-1 and pass-2 (fq bank: no cur/fact/mask reach)\n'
        '        out["nl0"] = (fat.mean(1).unsqueeze(1) @ waist).squeeze(1)\n',
        1, "forward: out nl_all/nlat_all (mining) + out nl0 (prior)")

ast.parse(s)
assert s.count('nl_all') >= 3 and s.count('"nl0"') == 1

print(f"[nl tap] {len(ANCHORS)} anchors OK "
      f"(phase1 +{s.count(chr(10)) - n_lines0} lines):")
for i, desc in enumerate(ANCHORS, 1):
    print(f"  {i:2d}. {desc}")
if CHECK:
    print("[nl tap] --check: ast OK on the would-be result; NOTHING written")
else:
    open(fn, 'w').write(s)
    print("[nl tap] APPLIED (scripts/phase1_algebra_head.py); envs unset = "
          "byte-identical — run the eq gate (A/B/C) before trusting")
