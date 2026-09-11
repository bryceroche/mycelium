"""apply_wheel_train.py — THE WHEEL, STAGE 2 (2026-09-11, the word): the step
trainer extended with the solver's core -> spotlight as a second DETACHED
CONSTANT per seam (beside the facts), read by breath_step as
state["wheel_bias"]; the cores in a spawn pool. Also: (a) the ALG_STELLAR
fence lifted (v2 lives inside breath_step; proven by --eqfwd/--eqbwd
under the family env before any training); (b) the state banks sized by
L_TOT (the loop carries 24 factor + 8 scratch rows); (c) the prep pass
cached by warm-ckpt sha and run through the JIT reader at batch 32.
Dials: ST_WHEEL=1, ST_WHEEL_BETA (3), ST_WHEEL_MODE (own|union),
ST_WORKERS. Unset = the walker as before."""
import sys
P = "scripts/step_trainer.py"; s = open(P).read()
if "ST_WHEEL" in s:
    print("apply_wheel_train: already applied — refusing"); sys.exit(2)
def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (s.count(old), old[:90]); s = s.replace(old, new)

# (a) the fence
rep('REFUSED = ("ALG_RINGS", "ALG_XOUT", "ALG_CLOCK", "ALG_STELLAR",\n',
    'REFUSED = ("ALG_RINGS", "ALG_XOUT", "ALG_CLOCK",   # ALG_STELLAR lifted 2026-09-11: v2 lives inside breath_step (rung 1 re-proven)\n')
# (b) L_TOT banks + wheel buffers + dials
rep('        L, K, T = H.L_FAC, H.K_VARS, H.T_ALG\n'
    '        HW, HT = H.H_W, H.H_TRUNK\n'
    '        self.L = L\n',
    '        L, K, T = H.L_FAC, H.K_VARS, H.T_ALG\n'
    '        LT = H.L_TOT                       # the loop state\'s rows (factor + scratch)\n'
    '        HW, HT = H.H_W, H.H_TRUNK\n'
    '        self.L = L; self.LT = LT; self.T = T\n'
    '        self.wheel = bool(_envi("ST_WHEEL"))\n'
    '        self.wheel_beta = float(os.environ.get("ST_WHEEL_BETA", "3"))\n'
    '        self.wheel_mode = os.environ.get("ST_WHEEL_MODE", "own")\n'
    '        self.workers = int(os.environ.get("ST_WORKERS", "0")) or None\n')
rep('        self.cur_bank = [fix((B, L, HW), rg=True) for _ in range(self.K_B)]\n'
    '        nb_shape = (B, L, HW) if self.perslot else (B, HW)\n',
    '        self.cur_bank = [fix((B, LT, HW), rg=True) for _ in range(self.K_B)]\n'
    '        nb_shape = (B, LT, HW) if self.perslot else (B, HW)\n')
rep('        self.gar_bank = ([fix((B, L, gd)) for _ in range(self.K_B - 1)]\n',
    '        self.gar_bank = ([fix((B, LT, gd)) for _ in range(self.K_B - 1)]\n')
rep('        self.G_cur = fix((B, L, HW))\n', '        self.G_cur = fix((B, LT, HW))\n')
rep('        self.cur_dec = fix((B, L, HW))\n'
    '        self.fact_dec = fix((B, K, 4))\n',
    '        self.cur_dec = fix((B, LT, HW))\n'
    '        self.fact_dec = fix((B, K, 4))\n'
    '        self.fat_bank = fix((B, LT, T))    # stage-0 attention: the wheel\'s source sentences\n'
    '        self.fat_np = None\n'
    '        self.wheel_bank = ([fix((B, 1, LT, T)) for _ in range(self.K_B - 2)]\n'
    '                           if self.wheel else [])\n')
# dec keys for decode() when the wheel is on (set in prime, after emit_keys)
rep('        self.rot2 = tap["ctx"]["rot2"]\n'
    '        self.emit_keys = sorted(tap["heads_of"](tap["fst"]).keys())\n',
    '        self.rot2 = tap["ctx"]["rot2"]\n'
    '        self.emit_keys = sorted(tap["heads_of"](tap["fst"]).keys())\n'
    '        if self.wheel:\n'
    '            self.dec_keys = [k for k in ("pres", "ftype", "op", "dig", "args", "res", "dup",\n'
    '                                         "sgn", "dargs", "dig2", "sel", "islit", "y")\n'
    '                             if k in self.emit_keys]\n')
# the state: the wheel's bias for breath k (computed after breath k-1)
rep('        return {"cur": cur, "breaths": [],\n'
    '                "nb": ([self.nb_bank[j] for j in range(n_nb)]\n',
    '        return {"cur": cur, "breaths": [],\n'
    '                "wheel_bias": (self.wheel_bank[k - 2]\n'
    '                               if (self.wheel and k >= 2) else None),\n'
    '                "nb": ([self.nb_bank[j] for j in range(n_nb)]\n')
# stage 0 banks fat
rep('            return [tap["waist"].detach(), tap["vst_base"].detach(),\n'
    '                    tap["fst"].detach()]\n',
    '            return [tap["waist"].detach(), tap["vst_base"].detach(),\n'
    '                    tap["fst"].detach(), tap["fat"].detach()]\n')
rep('        self.cur_bank[0].assign(r[2])\n'
    '        self.Tensor.realize(self.waist_bank, self.vst_base_bank,\n'
    '                            self.cur_bank[0])\n',
    '        self.cur_bank[0].assign(r[2])\n'
    '        self.fat_bank.assign(r[3])\n'
    '        self.Tensor.realize(self.waist_bank, self.vst_base_bank,\n'
    '                            self.cur_bank[0], self.fat_bank)\n'
    '        self.fat_np = self.fat_bank.numpy() if self.wheel else None\n')
# the seam: decode when ping OR wheel; facts if ping; the wheel's bias if wheel
rep('            if self.ping:\n'
    '                # the seam stub: decode confident slots on vst(fact_{k-1}),\n'
    '                # ping the organ, pack the fixed (B,24,4) buffer\n'
    '                self.put(self.fact_dec, fact_cur)\n'
    '                self.cur_dec.assign(self.cur_bank[k]).realize()\n'
    '                dec = self.dec_fn()\n'
    '                onp = {kk: t.numpy() for kk, t in zip(self.dec_keys, dec)}\n'
    '                fact_cur = H.alt2_fact_buf(onp, se_np, nv, ma,\n'
    '                                           theta=self.theta)\n'
    '                rates.append(int((fact_cur[:, :, 0] > 0).sum()))\n',
    '            if self.ping or self.wheel:\n'
    '                # the seam stub: decode confident slots on vst(fact_{k-1}),\n'
    '                # ping the organ, pack the fixed (B,24,4) buffer\n'
    '                self.put(self.fact_dec, fact_cur)\n'
    '                self.cur_dec.assign(self.cur_bank[k]).realize()\n'
    '                dec = self.dec_fn()\n'
    '                onp = {kk: t.numpy() for kk, t in zip(self.dec_keys, dec)}\n'
    '            if self.ping:\n'
    '                fact_cur = H.alt2_fact_buf(onp, se_np, nv, ma,\n'
    '                                           theta=self.theta)\n'
    '                rates.append(int((fact_cur[:, :, 0] > 0).sum()))\n'
    '            if self.wheel and k <= self.K_B - 2:\n'
    '                # THE WHEEL (2026-09-11): commit this breath\'s parse, solve,\n'
    '                # core, spotlight for breath k+1 — a detached constant\n'
    '                bias, turned = wheel_bias(H, onp, self.fat_np, se_np, nv, ma,\n'
    '                                          self.wheel_beta, self.wheel_mode,\n'
    '                                          self.workers, self.LT)\n'
    '                self.put(self.wheel_bank[k - 1], bias)\n'
    '                self.turned.append(turned)\n')
rep('        fact_cur = fact0\n        rates = []\n',
    '        fact_cur = fact0\n        rates = []\n        self.turned = []\n')
# the wheel's bias builder (module level) + the prep cache
rep('def _row_meta(H, samples, idx):\n',
    'def wheel_bias(H, onp, fat_np, se_np, nv, ma, beta, mode, workers, LT):\n'
    '    """The spotlight: per row, decode the parse slot by slot, solve, and\n'
    '    on a certified refusal +beta on the token scores of the core slots\'\n'
    '    source sentences (own | union). Returns ((B,1,LT,T) bias, rows turned)."""\n'
    '    sys.path.insert(0, "scripts")\n'
    '    from alternator_bridge import core_rows\n'
    '    B, T = se_np.shape\n'
    '    parses = []; rows = []\n'
    '    for b in range(B):\n'
    '        row = {kk: onp[kk][b] for kk in onp}\n'
    '        row["query"] = np.zeros(H.K_VARS, np.float32)\n'
    '        parse = []\n'
    '        for j in range(H.L_FAC):\n'
    '            if row["pres"][j] <= 0:\n'
    '                continue\n'
    '            rj = dict(row); pr = np.full_like(row["pres"], -1.0); pr[j] = row["pres"][j]; rj["pres"] = pr\n'
    '            try:\n'
    '                facs, _ = H.decode(rj)\n'
    '            except Exception:\n'
    '                facs = []\n'
    '            for f in facs:\n'
    '                f["_slot"] = j; parse.append(f)\n'
    '        parses.append(parse); rows.append((int(nv[b]), parse, int(ma[b])))\n'
    '    res = core_rows(rows, workers)\n'
    '    bias = np.zeros((B, 1, LT, T), np.float32); turned = 0\n'
    '    for b, (status, core) in enumerate(res):\n'
    '        if status != "unsat" or not core:\n'
    '            continue\n'
    '        turned += 1\n'
    '        parse = parses[b]\n'
    '        slots = [parse[k]["_slot"] for k in core]\n'
    '        sents = {j: int(se_np[b, min(int(fat_np[b, j].argmax()), T - 1)]) for j in slots}\n'
    '        for j in slots:\n'
    '            want = set(sents.values()) if mode == "union" else {sents[j]}\n'
    '            bias[b, 0, j, :] = np.where(np.isin(se_np[b], list(want)), beta, 0.0)\n'
    '    return bias, turned\n'
    '\n'
    '\n'
    'def _row_meta(H, samples, idx):\n')
# prep: cache by warm-ckpt sha + JIT reader at batch 32
rep('def prep_masks_facts(H, p, samples, states, tokmask, sent):\n',
    'def prep_masks_facts(H, p, samples, states, tokmask, sent):\n'
    '    """Cached by the warm checkpoint\'s sha + the mask door (2026-09-11);\n'
    '    the pass itself runs through the JIT reader at batch 32 (bit-identical\n'
    '    to eager: ledger 2026-09-10)."""\n'
    '    import hashlib\n'
    '    from mycelium.jit_read import read_forward as _rf, reset as _jr_reset\n'
    '    _ck = os.environ.get("ST_CKPT", "")\n'
    '    _key = hashlib.sha256((open(_ck, "rb").read() if _ck and os.path.isfile(_ck) else b"") + os.environ.get("ALG_SLOT_ALL", "0").encode() + str(states.shape[0]).encode()).hexdigest()[:16]\n'
    '    _cp = f".cache/st_prep_{_key}.npz"\n'
    '    if os.path.isfile(_cp):\n'
    '        z = np.load(_cp); print(f"[prep] cache HIT {_cp}", flush=True)\n'
    '        return z["MASKS"], (z["FACTS"] if "FACTS" in z.files else None)\n'
    '    print(f"[prep] cache MISS {_cp} -> the JIT pass at batch 32", flush=True)\n'
    '    _prev = os.environ.get("ALG_JIT_READ"); os.environ["ALG_JIT_READ"] = "1"\n'
    '    try:\n'
    '        MASKS, FACTS = _prep_pass(H, p, samples, states, tokmask, sent, _rf)\n'
    '    finally:\n'
    '        _jr_reset()\n'
    '        if _prev is None: os.environ.pop("ALG_JIT_READ", None)\n'
    '        else: os.environ["ALG_JIT_READ"] = _prev\n'
    '    np.savez_compressed(_cp, MASKS=MASKS, **({"FACTS": FACTS} if FACTS is not None else {}))\n'
    '    print(f"[prep] cached -> {_cp}", flush=True)\n'
    '    return MASKS, FACTS\n'
    '\n'
    '\n'
    'def _prep_pass(H, p, samples, states, tokmask, sent, _rf):\n')
rep('    for s0 in range(0, n, 8):\n'
    '        sl = np.arange(s0, min(s0 + 8, n))\n'
    '        pad = 8 - len(sl)\n'
    '        sl_p = np.concatenate([sl, sl[:1].repeat(pad)]) if pad else sl\n'
    '        out0 = H.forward(\n'
    '            p, Tensor(states[sl_p].astype(np.float32), dtype=dtypes.float),\n'
    '            Tensor(tokmask[sl_p].astype(np.float32), dtype=dtypes.float),\n'
    '            Tensor(sent[sl_p].astype(np.int32), dtype=dtypes.int))\n',
    '    _PB = 32\n'
    '    _keys = ("fat", "args", "res", "pres", "ftype", "op", "dig", "dup")\n'
    '    for s0 in range(0, n, _PB):\n'
    '        sl = np.arange(s0, min(s0 + _PB, n))\n'
    '        pad = _PB - len(sl)\n'
    '        sl_p = np.concatenate([sl, sl[:1].repeat(pad)]) if pad else sl\n'
    '        out0 = _rf(H.forward,\n'
    '            p, Tensor(states[sl_p].astype(np.float32), dtype=dtypes.float),\n'
    '            Tensor(tokmask[sl_p].astype(np.float32), dtype=dtypes.float),\n'
    '            Tensor(sent[sl_p].astype(np.int32), dtype=dtypes.int), keys=_keys)\n')
# log the wheel's turns
rep('                print(f"  step {s:5d} loss={loss:.4f} lr={cur_lr:.1e} "\n'
    '                      f"({(time.time() - t0) / (s + 1):.2f}s/step) "\n'
    '                      f"facts/item/breath={rr}", flush=True)\n',
    '                print(f"  step {s:5d} loss={loss:.4f} lr={cur_lr:.1e} "\n'
    '                      f"({(time.time() - t0) / (s + 1):.2f}s/step) "\n'
    '                      f"facts/item/breath={rr}"\n'
    '                      + (f" wheel-turned/breath={getattr(w, \'turned\', [])}" if w.wheel else ""), flush=True)\n')
open(P, "w").write(s); print("apply_wheel_train: applied (fence, L_TOT banks, wheel seam, pool, prep cache)")
