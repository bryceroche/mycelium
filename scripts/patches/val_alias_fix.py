"""Patch (ledger 2026-11-02): THE VAL ALIAS. eval_step / eval_g are TinyJits that reuse their output buffers, so a list of their
returns read AFTER the loop holds N references to the LAST batch's result: every banked distill 'VAL mse' was the last val batch
(32 rows), not all 256 (found by the MBR smoke: JIT == eager per batch; the logged mean == batch 8 exactly). Fix: read each
batch's floats inside the loop. 'VAL32' keeps the old last-batch read for continuity with banked numbers. Applied with MBR."""
p = "scripts/distill_reader.py"; s = open(p).read()
assert "VAL32" not in s, "already applied"
R = [('''            vl = [eval_step(*batch_of(b)) for b in VB]
            vloss = float(np.mean([float(a.numpy()) for a, _ in vl])); vmse = float(np.mean([float(b.numpy()) for _, b in vl]))
            msg += f" | VAL loss {vloss:.4f} mse {vmse:.4f}"; hist.append(dict(step=step, val_loss=vloss, val_mse=vmse))''',
      '''            vl = [tuple(float(t.numpy()) for t in eval_step(*batch_of(b))) for b in VB]     # floats per batch: the JIT reuses its outputs
            vloss = float(np.mean([a for a, _ in vl])); vmse = float(np.mean([b for _, b in vl])); v32 = vl[-1][1]
            msg += f" | VAL loss {vloss:.4f} mse {vmse:.4f} (VAL32 {v32:.4f})"
            hist.append(dict(step=step, val_loss=vloss, val_mse=vmse, val32_mse=v32))'''),
     ('''                gv = [eval_g(*[Tensor''', '''                gv = [tuple(float(t.numpy()) for t in eval_g(*[Tensor'''),
     ('''for k, a in enumerate(gsm_rows(g_va[i:i + BATCH]))]) for i in range(0, len(g_va) - BATCH + 1, BATCH)]''',
      '''for k, a in enumerate(gsm_rows(g_va[i:i + BATCH]))])) for i in range(0, len(g_va) - BATCH + 1, BATCH)]'''),
     ('''gvm = float(np.mean([float(b.numpy()) for _, b in gv]))''', '''gvm = float(np.mean([b for _, b in gv]))''')]
for a, b in R:
    assert s.count(a) == 1, a[:70]; s = s.replace(a, b)
open(p, "w").write(s); print("applied val_alias_fix")
