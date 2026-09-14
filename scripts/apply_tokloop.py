"""THE NL LOOP v1 (2026-09-14, the word): a token step per breath, interleaved
BEFORE the slot step; the slot loop's grounding roads and station 3 read the
token loop's current states every breath. The step: multi-head token<-token
attention under the SENTENCE mask (the NL loop's own structural mask; claims
and the certificates are v2), a zero-init output projection (tok <- tok + attn
@ W_o, W_o = 0: bit-identical at birth; a gain the mandatory-road law says to
census — _CENSUS "tokloop" entries carry ||attn @ W_o|| / ||tok|| per breath).
The shared clock on the token states and the expand-collapse are v2 (each
needs its own birth read). ALG_TOKLOOP=1; unset = bit-identical. Idempotent."""
import os, sys
p = os.environ.get("HEAD_PATH", "scripts/phase1_algebra_head.py"); s = open(p).read()
if "ALG_TOKLOOP" in s:
    print("[apply] already applied"); sys.exit(0)
def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:80]); s = s.replace(old, new)
rep('''ALG_SHELF = int(os.environ.get("ALG_SHELF", "0"))
''', '''ALG_SHELF = int(os.environ.get("ALG_SHELF", "0"))
# THE NL LOOP v1 (2026-09-14): ALG_TOKLOOP=1 — a sentence-masked token<-token
# attention step per breath, interleaved before the slot step; the slot loop
# reads the token loop's states every breath. Zero-init output: exact birth.
ALG_TOKLOOP = int(os.environ.get("ALG_TOKLOOP", "0"))
''')
rep('''    if ALG_SHELF:
        _rS = np.random.RandomState(seed + 7331)
''', '''    if ALG_TOKLOOP:
        _rT = np.random.RandomState(seed + 4242)
        for _nm in ("tok_wq", "tok_wk", "tok_wv"):
            p[_nm] = t((_rT.randn(H_W, H_W) / math.sqrt(H_W)).astype(np.float32))
            p[_nm + "_b"] = t(np.zeros(H_W))
        p["tok_wo"] = t(np.zeros((H_W, H_W), np.float32))     # the zero door: exact birth (censused)
        p["tok_wo_b"] = t(np.zeros(H_W))
    if ALG_SHELF:
        _rS = np.random.RandomState(seed + 7331)
''')
rep('''def _make_bank(p, waist, tokmask, B):
''', '''def _token_step(p, tok, tokmask, sent, B, kb):
    """THE NL LOOP's breath (v1): tok (B, T, HW) <- tok + SentenceAttn(tok) @ W_o.
    Tokens attend within their own sentence (sent ids equal) and only over
    real tokens; W_o is zero at birth."""
    T = int(tok.shape[1]); hd = H_W // N_HEADS
    q = (tok @ p["tok_wq"] + p["tok_wq_b"]).reshape(B, T, N_HEADS, hd).permute(0, 2, 1, 3)
    k = (tok @ p["tok_wk"] + p["tok_wk_b"]).reshape(B, T, N_HEADS, hd).permute(0, 2, 1, 3)
    v = (tok @ p["tok_wv"] + p["tok_wv_b"]).reshape(B, T, N_HEADS, hd).permute(0, 2, 1, 3)
    sc = (q @ k.transpose(-2, -1)) / math.sqrt(hd)                       # (B, H, T, T)
    same = (sent.reshape(B, 1, T, 1) == sent.reshape(B, 1, 1, T)).float()  # the sentence mask
    ok = same * tokmask.reshape(B, 1, 1, T)
    sc = sc.clip(-1e4, 1e4) + (1.0 - ok) * -1e4
    a = sc.softmax(-1)
    o = (a @ v).permute(0, 2, 1, 3).reshape(B, T, H_W) @ p["tok_wo"] + p["tok_wo_b"]
    if _CENSUS is not None:      # the pre/post knob law: the injection vs the state, per breath
        _CENSUS.append((kb, "tokloop", (o.pow(2).sum(-1).sqrt() / (tok.pow(2).sum(-1).sqrt() + 1e-6)).mean().realize().numpy()))
    return tok + o


def _make_bank(p, waist, tokmask, B):
''')
rep('''        if not (_STEP_TAP is not None and _STEP_TAP.get("hold")):
            for kb in range(1, K_B):
                breath_step(p, _bs_state, kb, _bs_ctx)
''', '''        if not (_STEP_TAP is not None and _STEP_TAP.get("hold")):
            _tok = waist
            for kb in range(1, K_B):
                if ALG_TOKLOOP and "tok_wq" in p:      # THE NL LOOP: the token step, then the slot step reads it
                    _tok = _token_step(p, _tok, tokmask, sent, B, kb)
                    _bs_ctx["waist"] = _tok
                    _bs_ctx["bank"] = _make_bank(p, _tok, tokmask, B)
                breath_step(p, _bs_state, kb, _bs_ctx)
''')
open(p, "w").write(s); print("[apply] the NL loop v1 applied to", p)
