"""(ledger 11-14) GK-ST: RD_GKNN_ST=1 (with RD_GKNN=k, tag K{k}S) -- the KNN picks enter the routed heads' scores as a pure mask; the
learned similarity rides only as a straight-through term (sim - sim.detach(): value 0, gradient as a bias), self excluded -- the
forward has no road to turn message passing off (the 14:28 autopsy: the bias's self term let G8K3 shut it)."""
import sys
p = sys.argv[1]; s = open(p).read()
def rep(a, b):
    global s
    assert s.count(a) == 1, a[:70]; s = s.replace(a, b)
rep('''KD = 32''', '''KD = 32
GKNN_ST = int(os.environ.get("RD_GKNN_ST", "0"))   # (GK-ST, ledger 11-14) the similarity in the scores as straight-through only, self excluded''')
rep('''{f'K{GKNN}' if GKNN else ''}_depth2c''', '''{f'K{GKNN}' if GKNN else ''}{'S' if GKNN_ST else ''}_depth2c''')
rep('''assert (GPLACEBO > 0) + (GNOISE > 0) + (GKNN > 0) <= 1, "one routing form at a time"''', '''assert (GPLACEBO > 0) + (GNOISE > 0) + (GKNN > 0) <= 1, "one routing form at a time"
assert not GKNN_ST or GKNN, "GKNN_ST is a form of the KNN heads"''')
rep('''            km = (sel.where(0.0, -1e4) + sim).reshape(B, 1, L, L)''', '''            km = (sel.where(0.0, -1e4) + ((sim - sim.detach()) * (1 - eyeL) if GKNN_ST else sim)).reshape(B, 1, L, L)''')
open(p, "w").write(s); print("gk-st patch applied")
