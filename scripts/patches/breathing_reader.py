"""Patch (ledger 2026-11-02, Bryce: "lets do this"): THE BREATHING READER (MBR). DR_BREATHS=K (0 = off, bit-identical): the DR_LAYERS
distinct attention layers are applied K times, weight-tied (breaths never specialize), on the T9-NS clock (no RoPE, no MLP).
MATRYOSHKA / coarse-to-fine: breath b sees only the offset gears whose period is >= UNLOCK[b] (the outer doll = coarse position;
each breath unlocks the next faster gear, the last breath sees all six); the slow token gears 7-9 are always on.
BREATH STAMP: queries (only) turn by the T6 gears x the breath tick on 6 of the quiet planes [15:21) -- 'rotate the QUERIES'.
DEEP SUPERVISION (diffusion-style successive estimates): every breath is read out and distilled, weight (b+1)/sum.
Applied after lfq5's last launch and CY's export (both re-read distill_reader.py)."""
p = "scripts/distill_reader.py"; s = open(p).read()
assert "DR_BREATHS" not in s, "already applied"
R = [
('WARM = os.environ.get("DR_WARM", "")',
 'BR = int(os.environ.get("DR_BREATHS", "0"))      # (ledger 11-02) THE BREATHING READER: DR_LAYERS weight-tied layers x BR breaths\n'
 'WARM = os.environ.get("DR_WARM", "")'),
('def init_params(mean_out):',
 '''if BR:   # (ledger 11-02 MBR) per-breath plane masks for the coarse-to-fine clock + the query-only breath stamp, constants outside the JIT
    assert T9 and T9NS, "the breathing reader runs on the T9-NS clock (no RoPE)"
    _UNLOCK = {4: (3, 2, 1, 0), 3: (3, 1, 0), 2: (3, 0), 1: (0,)}[BR]       # first unlocked offset gear (index into T6_SPEEDS, fastest = 0)
    _GQ = np.radians(np.asarray(CK.T6_SPEEDS, np.float32))
    BRM, BRQC, BRQS = [], [], []
    for _b in range(BR):
        _m = np.ones(HD // 2, np.float32); _m[:_UNLOCK[_b]] = 0.0               # offset planes [0:6): the faster gears quiet this breath
        _qa = np.zeros(HD // 2, np.float32); _qa[15:21] = (_b + 1) * _GQ        # the breath stamp, queries only, quiet planes [15:21)
        _m, _qa = np.concatenate([_m, _m]), np.concatenate([_qa, _qa])
        BRM.append(Tensor(_m, dtype=dtypes.float).reshape(1, 1, 1, HD).contiguous().realize())
        BRQC.append(Tensor(np.cos(_qa), dtype=dtypes.float).reshape(1, 1, 1, HD).contiguous().realize())
        BRQS.append(Tensor(np.sin(_qa), dtype=dtypes.float).reshape(1, 1, 1, HD).contiguous().realize())


def init_params(mean_out):'''),
('def forward(p, x0, am, db=None, rc=None):\n    return _rms_norm(body(p, x0, am, db, rc), p["fnorm"], EPS) @ p["w_out"] + p["b_out"]\n\n\ndef body(p, x0, am, db=None, rc=None):',
 'def forward(p, x0, am, db=None, rc=None):\n    return head(p, body(p, x0, am, db, rc))\n\n\ndef head(p, x):\n    return _rms_norm(x, p["fnorm"], EPS) @ p["w_out"] + p["b_out"]\n\n\n'
 'def body(p, x0, am, db=None, rc=None, every=False):'),
('''    for i in range(NL):
        h = _rms_norm(x, p[f"an{i}"], EPS)''',
 '''    xs = []
    for br in range(max(BR, 1)):
      if BR:   # the coarse-to-fine clock: quiet planes read cos 1 / sin 0
        kc, ks = rc[0] * BRM[br] + (1 - BRM[br]), rc[1] * BRM[br]
      for i in range(NL):
        h = _rms_norm(x, p[f"an{i}"], EPS)'''),
('''        if rc is None: q, k = _apply_rope(q, k, COS, SIN, T_ALG)
        else: q, k = q * rc[0] + _rotate_half(q) * rc[1], k * rc[0] + _rotate_half(k) * rc[1]   # THE T6 ADDRESS (no RoPE)
        sc = (q @ k.transpose(-2, -1)) * (1 / math.sqrt(HD)) + am
        a = sc.clip(-1e4, 1e4).softmax(-1)
        x = x + (a @ v).transpose(1, 2).reshape(B, T_ALG, D) @ p[f"wo{i}"]
        if FF:
            h2 = _rms_norm(x, p[f"fn{i}"], EPS)
            x = x + ((h2 @ p[f"wg{i}"]).silu() * (h2 @ p[f"wu{i}"])) @ p[f"wd{i}"]
    return x''',
 '''        if BR:
            q, k = q * kc + _rotate_half(q) * ks, k * kc + _rotate_half(k) * ks
            q = q * BRQC[br] + _rotate_half(q) * BRQS[br]                                            # the breath stamp (queries only)
        elif rc is None: q, k = _apply_rope(q, k, COS, SIN, T_ALG)
        else: q, k = q * rc[0] + _rotate_half(q) * rc[1], k * rc[0] + _rotate_half(k) * rc[1]   # THE T6 ADDRESS (no RoPE)
        sc = (q @ k.transpose(-2, -1)) * (1 / math.sqrt(HD)) + am
        a = sc.clip(-1e4, 1e4).softmax(-1)
        x = x + (a @ v).transpose(1, 2).reshape(B, T_ALG, D) @ p[f"wo{i}"]
        if FF:
            h2 = _rms_norm(x, p[f"fn{i}"], EPS)
            x = x + ((h2 @ p[f"wg{i}"]).silu() * (h2 @ p[f"wu{i}"])) @ p[f"wd{i}"]
      xs.append(x)
    return xs if every else x'''),
('''def _train(x0, am, tm, tgt, db, cs=None, sn=None):
    with Tensor.train():
        out = forward(p, x0, am, db, None if cs is None else (cs, sn)); loss, mse = loss_fn(out, tgt, tm)''',
 '''BRW = [(b + 1) / sum(range(1, BR + 1)) for b in range(BR)]   # (MBR) deep supervision: every breath distilled, later breaths weigh more


def _train(x0, am, tm, tgt, db, cs=None, sn=None):
    with Tensor.train():
        if BR:
            ls = [loss_fn(head(p, xb), tgt, tm) for xb in body(p, x0, am, db, (cs, sn), every=True)]
            loss, mse = sum(w * l for w, (l, _) in zip(BRW, ls)), ls[-1][1]
        else:
            out = forward(p, x0, am, db, None if cs is None else (cs, sn)); loss, mse = loss_fn(out, tgt, tm)'''),
('safe_save({k: v.realize() for k, v in p.items()}, CKPT)\njson.dump(dict(tag=TAG, D=D, layers=NL, ff=FF,',
 '''brv = []
if BR:   # (MBR, bar MB2) the per-breath val mse: does each breath sharpen the estimate?
    for b in VB:
        x0, am, tm, tg, db, cs, sn = batch_of(b)
        brv.append([float(loss_fn(head(p, xb), tg, tm)[1].numpy()) for xb in body(p, x0, am, db, (cs, sn), every=True)])
    brv = np.mean(np.array(brv), 0).tolist(); print(f"[distill] BREATH val mse {' > '.join(f'{v:.4f}' for v in brv)}", flush=True)
safe_save({k: v.realize() for k, v in p.items()}, CKPT)
json.dump(dict(tag=TAG, D=D, layers=NL, breaths=BR, breath_val_mse=brv, ff=FF,'''),
]
for a, b in R:
    assert s.count(a) == 1, a[:70]; s = s.replace(a, b)
open(p, "w").write(s); print("applied breathing_reader")
