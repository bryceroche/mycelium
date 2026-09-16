"""apply_late_wheel.py — THE LATE WHEEL (2026-09-15, the perf win for the
silent-wheel mode): with ST_WHEEL_BETA=0 and no melt the forward never reads
the wheel's banks — only the loss does (the certificate weights). So the
wheel runs AFTER the forward: no per-breath GPU stop, one host pull of all
breaths' decodes, one pool call with every breath's memo misses, the banks
filled before the backward. ST_WHEEL_LATE=1 (asserts beta == 0 and no
melt). Bit-identical by construction; the eq gates + a bank-identity
smoke (per-breath vs late) prove it. Also ST_WHEEL_TIME now times the whole
per-breath wheel block (dec + pull + wheel_bias + puts), not only the parse
and cores. usage: apply_late_wheel.py [path]  (default scripts/step_trainer.py)"""
import sys
p = sys.argv[1] if len(sys.argv) > 1 else "scripts/step_trainer.py"; s = open(p).read()
assert "ST_WHEEL_LATE" not in s, "already applied"
def sub(old, new, what):
    global s
    assert s.count(old) == 1, (what, s.count(old)); s = s.replace(old, new)
sub('''        self.cert_lambda = float(os.environ.get("ST_CERT_LAMBDA", "0") or 0)
''', '''        self.cert_lambda = float(os.environ.get("ST_CERT_LAMBDA", "0") or 0)
        # THE LATE WHEEL (2026-09-15): the silent wheel (beta 0, no melt) steers nothing in the
        # forward, so it runs once after it — one pull, one pool call, the banks before the backward
        self.wheel_late = bool(_envi("ST_WHEEL_LATE"))
        assert not (self.wheel_late and (self.wheel_beta != 0.0 or H._WHEEL_MELT is not None)), "ST_WHEEL_LATE needs a silent wheel (ST_WHEEL_BETA=0, no ALG_WHEEL_MELT)"
''', "init")
sub('''            if self.wheel and k <= self.K_B - 2:
                # THE WHEEL (2026-09-11): commit this breath's parse, solve,
                # core, spotlight for breath k+1 — a detached constant
                bias, turned, melt = wheel_bias(H, onp, self.fat_np, se_np, nv, ma,
                                                self.wheel_beta, self.wheel_mode,
                                                self.workers, self.LT)
                self.put(self.wheel_bank[k - 1], bias)
                self.put(self.melt_bank[k - 1], melt)
                self.turned.append(turned)
''', '''            if self.wheel and k <= self.K_B - 2 and not self.wheel_late:
                # THE WHEEL (2026-09-11): commit this breath's parse, solve,
                # core, spotlight for breath k+1 — a detached constant
                _tw0 = time.time()
                bias, turned, melt = wheel_bias(H, onp, self.fat_np, se_np, nv, ma,
                                                self.wheel_beta, self.wheel_mode,
                                                self.workers, self.LT)
                self.put(self.wheel_bank[k - 1], bias)
                self.put(self.melt_bank[k - 1], melt)
                self.turned.append(turned)
                if _WHEEL_TIME: print(f"[wheel-block] breath {k}: {time.time() - _tw0:.2f}s (wheel_bias + puts)", flush=True)
''', "per-breath block")
sub('''            self.put(self.b_facts[k], fact_cur)
        return rates
''', '''            self.put(self.b_facts[k], fact_cur)
        if self.wheel and self.wheel_late:
            # THE LATE WHEEL: every breath's parse pulled once, solved in ONE pool call
            _tl0 = time.time(); rows_all = []; onps = []
            for k in range(1, self.K_B - 1):
                dec = self.dec_fns[k](); onp = {kk: t.numpy() for kk, t in zip(self.dec_keys, dec)}; onps.append(onp)
            for k, onp in zip(range(1, self.K_B - 1), onps):
                bias, turned, melt = wheel_bias(H, onp, self.fat_np, se_np, nv, ma, self.wheel_beta, self.wheel_mode, self.workers, self.LT)
                self.put(self.wheel_bank[k - 1], bias); self.put(self.melt_bank[k - 1], melt); self.turned.append(turned)
            if _WHEEL_TIME: print(f"[wheel-late] {self.K_B - 2} breaths: {time.time() - _tl0:.2f}s", flush=True)
        return rates
''', "late block")
open(p, "w").write(s); print(f"[apply] the late wheel + the per-breath timer -> {p}")
