"""t9_clock.py -- the T9 form (ledger 2026-11-01, REGISTERED): adds DR_T6=3 to scripts/distill_reader.py. Per head (32 planes):
[0:6) the six T6 gears x the OFFSET in the sentence | [6:12) the six T6 gears x the SENTENCE index | [12:15) gears 7-9 x the absolute
TOKEN index, periods 192 / 768 / 3072 tokens (the slow distance band RoPE's planes ~8-14 cover) | [15:32) the QUIET planes, speed 0
(position-free content channels, RoPE's planes ~15-31 rotate < 0.6 rad over the window). No duplicate gear anywhere; no RoPE.
APPLY ONLY after unit lfhy's last distill_reader launch (it re-launches the script for its export). Refuses if already applied."""
import sys
p = sys.argv[1] if len(sys.argv) > 1 else "scripts/distill_reader.py"; s = open(p).read()
assert "T9 =" not in s, "already applied"
r = [('HY = os.environ.get("DR_T6") == "2"\n',
      'HY = os.environ.get("DR_T6") == "2"\nT9 = os.environ.get("DR_T6") == "3"   # (ledger 11-01) T6 offset | T6 sentence | gears 7-9 x token | quiet planes\n'),
     ('T6 = os.environ.get("DR_T6") in ("1", "2")', 'T6 = os.environ.get("DR_T6") in ("1", "2", "3")'),
     ('    if HY:                                                                                            # RoPE micro | T6 macro, disjoint planes\n',
      '    if T9:                                                                                            # the clock owns every plane\n'
      '        G = np.radians(np.asarray(CK.T6_SPEEDS, np.float32)); S9 = 360.0 / np.asarray([192.0, 768.0, 3072.0], np.float32)\n'
      '        a = np.concatenate([off[..., None] * G, sent[..., None] * G, i[..., None].repeat(sent.shape[0], 0) * np.radians(S9),\n'
      '                            np.zeros(sent.shape + (HD // 2 - 15,), np.float32)], -1) * (tm[..., None] > 0)\n'
      '        a = np.concatenate([a, a], -1)[:, None].astype(np.float32)\n'
      '        return np.cos(a), np.sin(a)\n'
      '    if HY:                                                                                            # RoPE micro | T6 macro, disjoint planes\n')]
for a, b in r:
    assert s.count(a) == 1, a[:60]; s = s.replace(a, b)
open(p, "w").write(s); print("T9 applied to", p)
