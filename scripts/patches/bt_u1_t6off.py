"""U1 (ledger 2026-10-31): gsm_breath_reader.py gains BT_T6OFF (default-off): the learned in-sentence OFFSET embedding is removed and the
offset turns its own T6 planes instead: per head, planes [0:DH/4) = the T6 gears x the sentence index (as before, on half the planes),
[DH/4:DH/2) = the T6 gears x the offset (slots: their slot index, as encode() writes). Applied by the lfu1 chain only after BT3's last
launch of gsm_breath_reader.py (never edit a script a running chain re-reads). Idempotent check: refuses if BT_T6OFF exists."""
import sys; p = sys.argv[1] if len(sys.argv) > 1 else "scripts/gsm_breath_reader.py"; s = open(p).read()
assert "BT_T6OFF" not in s, "already applied"
def rep(old, new):
    global s
    assert s.count(old) == 1, old[:70]; s = s.replace(old, new)
rep('''ZOOM_TAU = 0.5''', '''ZOOM_TAU = 0.5
T6OFF = os.environ.get("BT_T6OFF") == "1"            # (ledger 10-31, U1) no learned offset: the offset turns its own T6 planes''')
rep('''    w = torch.as_tensor(np.radians(CK.t6_plane_speeds(DH // 2)), dtype=torch.float32)      # attention space: the T6 clock, per head''',
'''    w = torch.as_tensor(np.radians(CK.t6_plane_speeds(DH // 2)), dtype=torch.float32)      # attention space: the T6 clock, per head
    wh = torch.as_tensor(np.radians(CK.t6_plane_speeds(DH // 4)), dtype=torch.float32)

    def ang(sid, off):                                                                     # (B, L) x2 -> (B, L, DH/2) attention angles
        if not T6OFF: return sid[..., None].float() * w
        return torch.cat([sid[..., None].float() * wh, off[..., None].float() * wh], -1)    # sentence planes | offset planes''')
rep('''            x = x + s.isn(t["isn"].long()) + s.isnum(t["isnum"].long()) + s.gp(t["g"]) + s.off(t["off"])''',
'''            x = x + s.isn(t["isn"].long()) + s.isnum(t["isnum"].long()) + s.gp(t["g"]) + (0 if T6OFF else s.off(t["off"]))''')
rep('''            Lq = x.shape[1] - K; xq = x[:, :Lq]; aq = t["sid"][:, :Lq, None].float() * w''',
'''            Lq = x.shape[1] - K; xq = x[:, :Lq]; aq = ang(t["sid"][:, :Lq], t["off"][:, :Lq])''')
rep('''            sa = t["sid"][:, Lq:, None].float() * w; ka = t["sid"][..., None].float() * w''',
'''            sa = ang(t["sid"][:, Lq:], t["off"][:, Lq:]); ka = ang(t["sid"], t["off"])''')
if True:   # the learned offset table is not built under T6OFF
    rep('''            s.off = nn.Embedding(OMAX, D); s.idp''', '''            s.off = None if T6OFF else nn.Embedding(OMAX, D); s.idp''')
open(p, "w").write(s); print("U1 applied")
