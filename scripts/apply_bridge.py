"""apply_bridge.py — THE BRIDGE applied (2026-09-14, word given): the head's
certificate roads (_claim_bias, _melt, _nogood_apply, _wheel_turn's
spotlight) and the step trainer's wheel_bias spotlight delegate to
mycelium/loop_bridge (one construction, shown identical to both copies by
scripts/bridge_identity_check.py BEFORE this apply); the miner accumulates
through step_atlas.AtlasBanks (one map, two charts). _token_step's own
sentence mask is NOT re-pointed here: the tokloop arms execute it and
nothing they run is edited under them (deferred; same_sentence is ready).
Idempotent: refuses a second apply."""
import sys

def sub(s, old, new, n=1, what=""):
    assert s.count(old) == n, (what, s.count(old))
    return s.replace(old, new)

# ---- the head -------------------------------------------------------------
p = "scripts/phase1_algebra_head.py"; s = open(p).read()
assert "loop_bridge" not in s, "apply_bridge: already applied to the head"
s = sub(s, '''    C = (prev_at > _T2_TAU).float()
    if melted is not None:
        C = C * (1.0 - melted.reshape(B, LT, 1))
    anyc = C.sum(1, keepdim=True)                     # (B, 1, T)
    other = ((anyc - C) > 0).float()                  # claimed by someone else
    return (other * -_T2_BETA).reshape(B, 1, LT, -1)
''', '''    from mycelium.loop_bridge import claims_bias    # THE BRIDGE (2026-09-14): the claims road, one definition
    return claims_bias(prev_at, melted, B, LT, _T2_BETA, _T2_TAU)
''', what="_claim_bias")
s = sub(s, '''    """Mask the previous choices out of the head logits, in place. nogoods: list of (b, j, field, idx)."""
    for b, j, f, idx in nogoods:
        onp[f][b, j, idx] = -1e9
    return onp
''', '''    """Mask the previous choices out of the head logits, in place. nogoods: list of (b, j, field, idx)."""
    from mycelium.loop_bridge import nogood_apply   # THE BRIDGE: the nogood road
    return nogood_apply(onp, nogoods)
''', what="_nogood_apply")
s = sub(s, '''    B, LT, HW = (int(x) for x in cur.shape)
    n = _whip_noise(B, LT, HW)
    g = _polar_sink()[2] if ALG_POLAR else 1.0
    n = n * g
    nn = (n * n).sum(-1, keepdim=True).sqrt() + 1e-6
    nc = (cur * cur).sum(-1, keepdim=True).sqrt()
    fresh = n / nn * nc * _WHEEL_MELT                 # the melted content (amp 0 -> zeros)
    mm = m.reshape(B, LT, 1)
    return cur + mm * (fresh - cur * g)               # content replaced on melted slots; clock (g=0) kept
''', '''    from mycelium.loop_bridge import melt as _bridge_melt   # THE BRIDGE: the melt road
    B, LT, HW = (int(x) for x in cur.shape)
    return _bridge_melt(cur, m, _WHEEL_MELT, _whip_noise(B, LT, HW),
                        _polar_sink()[2] if ALG_POLAR else 1.0)
''', what="_melt")
s = sub(s, '''    bias = _np.zeros((B, 1, L_TOT, T), _np.float32)
    melt = _np.zeros((B, L_TOT), _np.float32)          # THE MELT's mask for the next breath
''', '''    melt = _np.zeros((B, L_TOT), _np.float32)          # THE MELT's mask for the next breath = the MUC certificate on slots
''', what="_wheel_turn bias alloc")
s = sub(s, '''        for j in core_slots:
            melt[b, j] = 1.0
        sents = {}
        for j in core_slots:
            tok = int(fat_np[b, j].argmax()); sents[j] = int(sent_np[b, min(tok, T - 1)])
        for j in core_slots:
            want = set(sents.values()) if mode == "union" else {sents[j]}
            m = _np.isin(sent_np[b], list(want))
            bias[b, 0, j, :] = _np.where(m, beta, 0.0).astype(_np.float32)
    _WHEEL.setdefault("turned", []).append((kb, turned, B))
''', '''        for j in core_slots:
            melt[b, j] = 1.0
    # THE BRIDGE (2026-09-14): the MUC certificate, born on slots, projected to the
    # tokens through this breath's cross attention — the spotlight, one construction
    from mycelium.loop_bridge import Bridge
    bias = Bridge(fat_np, sent_np).spotlight(melt, beta, mode)
    _WHEEL.setdefault("turned", []).append((kb, turned, B))
''', what="_wheel_turn spotlight")
open(p, "w").write(s); print("[apply] head: _claim_bias/_nogood_apply/_melt/_wheel_turn -> the bridge")

# ---- the step trainer -----------------------------------------------------
p = "scripts/step_trainer.py"; s = open(p).read()
assert "loop_bridge" not in s, "apply_bridge: already applied to the trainer"
s = sub(s, '''    bias = np.zeros((B, 1, LT, T), np.float32); turned = 0
    melt = np.zeros((B, LT), np.float32)                 # THE MELT's mask
    for b, (status, core) in enumerate(res):
        if status != "unsat" or not core:
            continue
        turned += 1
        parse = parses[b]
        slots = [parse[k]["_slot"] for k in core]
        for j in slots:
            melt[b, j] = 1.0
        sents = {j: int(se_np[b, min(int(fat_np[b, j].argmax()), T - 1)]) for j in slots}
        for j in slots:
            want = set(sents.values()) if mode == "union" else {sents[j]}
            bias[b, 0, j, :] = np.where(np.isin(se_np[b], list(want)), beta, 0.0)
    return bias, turned, melt
''', '''    melt = np.zeros((B, LT), np.float32); turned = 0     # THE MELT's mask = the MUC certificate on slots
    for b, (status, core) in enumerate(res):
        if status != "unsat" or not core:
            continue
        turned += 1
        for k in core:
            melt[b, parses[b][k]["_slot"]] = 1.0
    # THE BRIDGE (2026-09-14): the certificate projected to the tokens — the
    # spotlight, ONE construction shared with the head (bridge_identity_check.py)
    from mycelium.loop_bridge import Bridge
    bias = Bridge(fat_np, se_np).spotlight(melt, beta, mode)
    return bias, turned, melt
''', what="wheel_bias spotlight")
open(p, "w").write(s); print("[apply] trainer: wheel_bias -> the bridge")

# ---- the miner ------------------------------------------------------------
p = "scripts/mine_step_atlas.py"; s = open(p).read()
assert "AtlasBanks" not in s, "apply_bridge: already applied to the miner"
s = sub(s, '''    from mycelium.step_atlas import (StepWelford, save_atlas, atlas_class,
                                     K_STEPS)
''', '''    from mycelium.step_atlas import AtlasBanks, atlas_class, K_STEPS
''', what="miner import")
s = sub(s, '''    cells = {}
    cells_nl = {}      # the second chart (the reading)
''', '''    banks = AtlasBanks(H_W)   # ONE MAP, TWO CHARTS: one accumulator, one save (2026-09-14)
''', what="miner banks")
s = sub(s, '''            for s_id in range(K_STEPS):
                key = (s_id, cls)
                if key not in cells:
                    cells[key] = StepWelford(H_W)
                # POOLING (see module docstring): mean over the 24 slots
                cells[key].add(br[s_id][bi].mean(0).astype(np.float64))
                if key not in cells_nl:
                    cells_nl[key] = StepWelford(H_W)
                # NL: attention-pooled waisted token state (the tap
                # already pooled in-graph; accumulate as-is)
                cells_nl[key].add(nl[s_id][bi].astype(np.float64))
''', '''            for s_id in range(K_STEPS):
                # THE SLOT CHART (the commitment): mean over the 24 slots
                banks.add("slot", s_id, cls, br[s_id][bi].mean(0).astype(np.float64))
                # THE TOKEN CHART (the reading): the attention-pooled waisted
                # token state (the tap pooled in-graph; accumulate as-is)
                banks.add("token", s_id, cls, nl[s_id][bi].astype(np.float64))
''', what="miner add")
s = sub(s, '''    path = save_atlas(cells, H_W, path=ATLAS_OUT,
                      manifest_path=RESEARCH_MANIFEST,
                      nl_cells=cells_nl)
    classes = sorted({c for (_, c) in cells})
''', '''    path = banks.save(ATLAS_OUT, RESEARCH_MANIFEST)   # the paired-count law lives in save
    classes = banks.classes()
''', what="miner save")
s = sub(s, '''        counts = [cells[(s, cls)].n if (s, cls) in cells else 0
                  for s in range(K_STEPS)]
        print(f"[mine-atlas]   {cls:12s} n/step={counts}")
        ncounts = [cells_nl[(s, cls)].n if (s, cls) in cells_nl
                   else 0 for s in range(K_STEPS)]
        assert ncounts == counts, \\
            f"paired charts disagree on {cls}: {ncounts} vs {counts}"
        print(f"[mine-atlas]   {cls:12s} nl n/step={ncounts}")
''', '''        counts = [banks.count("slot", s, cls) for s in range(K_STEPS)]
        print(f"[mine-atlas]   {cls:12s} n/step={counts}")
        print(f"[mine-atlas]   {cls:12s} token n/step={[banks.count('token', s, cls) for s in range(K_STEPS)]}")
''', what="miner report")
open(p, "w").write(s); print("[apply] miner: AtlasBanks (one map, two charts)")
