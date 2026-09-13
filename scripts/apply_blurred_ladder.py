"""THE BLURRED LADDER (2026-09-12, the word; "Two roads to the middle"): the
denoising schedule in its target-side form. The v98 ladder scores every breath's
raw state against the SHARP gold; here rung k's gold is blurred toward uniform by
beta_k = ALG_BLUR_MAX * cos^2(k*pi/(2(K-1))) (the stellarator's own clock: rung 0
most blurred, the last rung sharp). CE and BCE are linear in the target, so the
blurred-target loss is exactly (1-beta)*L(gold) + beta*L(uniform) — implemented
inside the two helpers, so every categorical/binary term blurs at once; the
stage-0 attention terms (fat/vat/ref: soft targets, shared across rungs) do not.
ALG_BLUR=0 (default) is bit-identical. Idempotent; HEAD_PATH env for a copy."""
import os, sys
p = os.environ.get("HEAD_PATH", "scripts/phase1_algebra_head.py"); s = open(p).read()
if "ALG_BLUR" in s:
    print("[apply] already applied"); sys.exit(0)
def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:80]); s = s.replace(old, new)
rep('''ALG_SHELF = int(os.environ.get("ALG_SHELF", "0"))
''', '''ALG_SHELF = int(os.environ.get("ALG_SHELF", "0"))
# THE BLURRED LADDER (2026-09-12, the word): ALG_BLUR=1 blurs rung k's gold
# toward uniform by ALG_BLUR_MAX * cos^2(k*pi/(2(K-1))) — the denoising
# schedule's target-side form (each breath graded on a job its level can do;
# the last rung sharp). CE/BCE are linear in the target: exact.
ALG_BLUR = int(os.environ.get("ALG_BLUR", "0"))
ALG_BLUR_MAX = float(os.environ.get("ALG_BLUR_MAX", "0.7"))
''')
rep('''def _loss_single(o, g):
''', '''def _loss_single(o, g, blur=0.0):
''')
rep('''    def bce(lg, tg):
        return (lg.maximum(0) - lg * tg + (1 + (-lg.abs()).exp()).log()).contiguous()   # perf: own kernel

    def ce(lg, tg):
        return (lg.log_softmax(-1) * -1).gather(-1, tg.unsqueeze(-1)).squeeze(-1).contiguous()   # perf: own kernel
''', '''    def bce(lg, tg):
        b = lg.maximum(0) - lg * tg + (1 + (-lg.abs()).exp()).log()
        if blur:   # THE BLURRED LADDER: target (1-blur)*t + blur*0.5, exactly (BCE is linear in t)
            b = (1.0 - blur) * b + blur * (lg.maximum(0) - 0.5 * lg + (1 + (-lg.abs()).exp()).log())
        return b.contiguous()   # perf: own kernel

    def ce(lg, tg):
        ls = lg.log_softmax(-1)
        c = (ls * -1).gather(-1, tg.unsqueeze(-1)).squeeze(-1)
        if blur:   # THE BLURRED LADDER: target (1-blur)*onehot + blur*uniform, exactly (CE is linear in t)
            c = (1.0 - blur) * c + blur * (ls * -1).mean(-1)
        return c.contiguous()   # perf: own kernel
''')
rep('''        for kb, ob in enumerate(o["breaths"]):
            full = dict(o, **ob)
            w = 1.0 + kb / max(K_B - 1, 1)
            term = _loss_single(full, g) * w
''', '''        for kb, ob in enumerate(o["breaths"]):
            full = dict(o, **ob)
            w = 1.0 + kb / max(K_B - 1, 1)
            beta = (ALG_BLUR_MAX * math.cos(kb * math.pi / (2 * max(K_B - 1, 1))) ** 2) if ALG_BLUR else 0.0
            term = _loss_single(full, g, blur=beta) * w
''')
open(p, "w").write(s); print("[apply] the blurred ladder applied to", p)
