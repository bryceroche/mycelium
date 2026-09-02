"""apply_alternator_v21.py — the staged ALTERNATOR v2.1 patch (2026-09-02).
The four-layer breath_step: stations 3-4 (INTEGRATE) — a SECOND
bank-attention (slots<-tokens) + a SECOND slot-mixer per breath, landing
between the existing GATHER+RELATE pair and the gate/commit. The room
where per-step-injected solver facts get absorbed before commitment (the
June engine's 4-per-breath precedent; ledger 2026-09-02, word given).

RUN ONLY AFTER the running chain exits (the running-chain law — a live
systemd unit has this module imported). --check mode loads the file,
asserts every anchor, ast-parses the WOULD-BE result and writes NOTHING.

Equivalence contract (rung 1 of the bring-up ladder, BY CONSTRUCTION):
  ALG_ALT21 unset -> zero behavior change (every new line env-guarded;
                     the chain's eq_check pre/post A/B/C dumps verify
                     bit-identity on banked ckpts)
  ALG_ALT21=1     -> at init both new blocks' OUTPUT projections are
                     ZEROS (the ResNet/V11 zero-init law), so their
                     additive deltas are exact zeros and forward equals
                     baseline exactly at birth; the identity path is
                     untouched (out = out + block(out) shape).
JIT discipline: no dtypes.float32 literals; both new score tensors carry
clip(-1e4, 1e4) (memory/reference_tinygrad_am_quirks.md). New params live
INSIDE the K_B > 1 block (like W_bq/W_bo) and inside the jitted pass-2
graph in every training mode — no None-grad at the optimizer.
"""
import ast
import sys

fn = 'scripts/phase1_algebra_head.py'
CHECK = '--check' in sys.argv
s = open(fn).read()
n_lines0 = s.count('\n')

assert 'alt21_' not in s, "alt21 already present — patch was applied; refuse"

PATCHES = []


def patch(num, desc, old, new):
    PATCHES.append((num, desc, old, new))


# 1. build_params: the INTEGRATE station pair, inside the K_B > 1 block
#    (anchored on the RINGS commit-head tail — the block's last organ).
#    Shapes copy the originals exactly: bank attention = the attn_* quartet
#    (lin(H_W, H_W) x4 + biases); mixer = W_bq/W_bk/W_bv (lin, or the
#    settle-transceiver phase idiom when ALG_SEPHASE_SETTLE, mirroring the
#    original branch) + W_bo/W_bo_b. INIT LAW: each block's OUTPUT
#    projection (alt21_attn_wo, alt21_W_bo) is ZEROS -> silent birth.
patch(1, "build_params: alt21 station params (env ALG_ALT21)",
      '''                                                     # door #54-R: bias-open
        pass''',
      '''                                                     # door #54-R: bias-open
        if int(os.environ.get("ALG_ALT21", "0")):
            # ALTERNATOR v2.1 (2026-09-02, word given): the INTEGRATE
            # station pair — a SECOND bank-attention (slots<-tokens) +
            # a SECOND slot-mixer per breath_step (stations 3-4 of the
            # four-layer step; the June engine's 4-per-breath precedent).
            # INIT LAW (ResNet/V11 zero-init): each block's OUTPUT
            # projection starts at ZERO so the ALG_ALT21=1 forward is
            # identical to baseline at birth (rung 1 of the bring-up
            # ladder); every other tensor copies its original's idiom.
            p["alt21_attn_wq"], p["alt21_attn_wq_b"] = lin(H_W, H_W)
            p["alt21_attn_wk"], p["alt21_attn_wk_b"] = lin(H_W, H_W)
            p["alt21_attn_wv"], p["alt21_attn_wv_b"] = lin(H_W, H_W)
            p["alt21_attn_wo"] = t(np.zeros((H_W, H_W)))   # ZERO: silent birth
            p["alt21_attn_wo_b"] = t(np.zeros(H_W))
            if ALG_SEPHASE_SETTLE:   # mirror the settle-transceiver idiom
                _stp21 = phase_alphabet(H_W, H_W, 1.0 / math.sqrt(H_W), rng)
                p["alt21_W_bq"] = t(_stp21 + rng.randn(H_W, H_W).astype(np.float32) / math.sqrt(H_W) * 0.2)
                p["alt21_W_bq_b"] = t(np.zeros((H_W,)))
                p["alt21_W_bk"] = t(_stp21 + rng.randn(H_W, H_W).astype(np.float32) / math.sqrt(H_W) * 0.2)
                p["alt21_W_bk_b"] = t(np.zeros((H_W,)))
            else:
                p["alt21_W_bq"], p["alt21_W_bq_b"] = lin(H_W, H_W)
                p["alt21_W_bk"], p["alt21_W_bk_b"] = lin(H_W, H_W)
            p["alt21_W_bv"], p["alt21_W_bv_b"] = lin(H_W, H_W)
            p["alt21_W_bo"] = t(np.zeros((H_W, H_W)))      # ZERO: silent birth
            p["alt21_W_bo_b"] = t(np.zeros(H_W))
        pass''')

# 2. forward breath loop: stations 3-4 between the RELATE mixer (h_slot,
#    ablation arms included) and the gate/commit (g = breath_gate). The
#    new blocks consume THIS breath's live context — q_extra's fresh
#    conditioning terms (q_extra - cur: breath_emb + notebook + garage +
#    detwave + sync oscillator, exactly what station 1 ate this kb), the
#    sync rotation _sync[0](kb), the router bias _rb7, and the breathed
#    mask _sm_kb (mask-re-formation included) — no stale copies.
patch(2, "forward: stations 3-4 before the gate (env ALG_ALT21)",
      '''            g = p["breath_gate"][kb].sigmoid()''',
      '''            if int(os.environ.get("ALG_ALT21", "0")) and "alt21_W_bo" in p:
                # ALTERNATOR v2.1 STATIONS 3-4 (2026-09-02): the INTEGRATE
                # pair, between GATHER+RELATE above and the gate/commit
                # below. Each block writes ADDITIVELY through its ZERO-INIT
                # output projection, identity path untouched
                # (out = out + block(out)). EQUIVALENCE BY CONSTRUCTION:
                # at init alt21_attn_wo and alt21_W_bo are zeros, so
                # _d21a = _d21b = exact zeros and h_slot (hence forward)
                # equals baseline exactly at birth; env unset skips the
                # whole block (the chain verifies via eq_check pre/post).
                _s21 = h_tok + h_slot            # the stream after 1-2
                # STATION 3: second bank-attention (slots<-tokens), live
                # query = slot codes + stream + this breath's conditioning
                _qx21 = p["fq"].unsqueeze(0) + _s21 + (q_extra - cur)
                _q21 = _qx21 @ p["alt21_attn_wq"] + p["alt21_attn_wq_b"]
                _k21 = waist @ p["alt21_attn_wk"] + p["alt21_attn_wk_b"]
                _v21 = waist @ p["alt21_attn_wv"] + p["alt21_attn_wv_b"]
                _hd21 = H_W // N_HEADS
                _qh21 = _q21.reshape(B, L_FAC, N_HEADS, _hd21).permute(0, 2, 1, 3)
                _kh21 = _k21.reshape(B, -1, N_HEADS, _hd21).permute(0, 2, 1, 3)
                _vh21 = _v21.reshape(B, -1, N_HEADS, _hd21).permute(0, 2, 1, 3)
                _sa21 = (_qh21 @ _kh21.transpose(-2, -1)) / math.sqrt(_hd21)
                if _sync is not None:            # the same breath rotation
                    _sa21 = _sa21 + _sync[0](kb)
                if _rb7 is not None:             # the same router bias
                    _sa21 = _sa21 + _rb7.unsqueeze(1) * p["r_gain"].reshape(1, 1, 1, 1)
                _sa21 = _sa21.clip(-1e4, 1e4) + (1.0 - tokmask.reshape(B, 1, 1, -1)) * -1e4
                _st21 = (_sa21.softmax(-1) @ _vh21).permute(0, 2, 1, 3).reshape(B, L_FAC, H_W)
                _d21a = _st21 @ p["alt21_attn_wo"] + p["alt21_attn_wo_b"]
                _s21 = _s21 + _d21a              # exact zero at birth
                # STATION 4: second slot-mixer over the SAME breathed mask
                _bq21 = _s21 @ p["alt21_W_bq"] + p["alt21_W_bq_b"]
                _bk21 = _s21 @ p["alt21_W_bk"] + p["alt21_W_bk_b"]
                _bv21 = _s21 @ p["alt21_W_bv"] + p["alt21_W_bv_b"]
                _sm21 = (_bq21 @ _bk21.transpose(-2, -1)) / math.sqrt(H_W)
                _sm21 = _sm21.clip(-1e4, 1e4) + (1.0 - _sm_kb) * -1e4
                if _A5 is not None and "alt_g" in p:   # v0 bias, as station 2
                    _sm21 = _sm21 + (_A5 + _A5.transpose(-2, -1)) \\
                        * p["alt_g"].reshape(1, 1, 1)
                if RINGS and int(os.environ.get("ALG_BEXIT", "0")):
                    _sm21 = _sm21 + m_c.reshape(B, 1, L_FAC) * -8.0
                _d21b = (_sm21.softmax(-1) @ _bv21) @ p["alt21_W_bo"] \\
                    + p["alt21_W_bo_b"]
                h_slot = h_slot + _d21a + _d21b  # additive; zeros at birth
            g = p["breath_gate"][kb].sigmoid()''')


for num, desc, old, new in PATCHES:
    assert old in s, f"anchor {num} MISSING ({desc}) — read the file, adjust"
    assert s.count(old) == 1, f"anchor {num} NOT UNIQUE ({desc})"
    s = s.replace(old, new, 1)

tree = ast.parse(s)                       # the would-be result must parse

# structural asserts on the would-be module (cheap, no import, no GPU)
for key in ("alt21_attn_wq", "alt21_attn_wo", "alt21_W_bq", "alt21_W_bo"):
    assert f'p["{key}"]' in s, f"patched tree lost {key}"
assert s.count('os.environ.get("ALG_ALT21", "0")') == 2, \
    "expected exactly 2 ALG_ALT21 guards (params + stations)"
assert s.index('_s21 = h_tok + h_slot') < s.index(
    'g = p["breath_gate"][kb].sigmoid()'), "stations must precede the gate"

print(f"[alternator-v2.1] {len(PATCHES)} anchors OK "
      f"(+{s.count(chr(10)) - n_lines0} lines):")
for num, desc, _o, _n in PATCHES:
    print(f"  {num:2d}. {desc}")
if CHECK:
    print("[alternator-v2.1] --check: ast OK on the would-be result; "
          "NOTHING written")
else:
    open(fn, 'w').write(s)
    print("[alternator-v2.1] APPLIED; ast OK — run the eq_check pre/post "
          "gate before trusting (equivalence contract, rung 1)")
