"""(ledger 11-14) GK-ST FIX: the straight-through similarity must reach ONLY knn_w -- e is built from the detached state in the ST form.
Without this the ST term's gradient flowed back through y = LN(x) into the residual stream: a phantom gradient the forward never acts
on (CPU test: forward and loss identical, ln1.weight grad off by 6.0e-4 at a scale of 5.9e-4), and G8K3S diverged (loss 0.21 -> 2.35)."""
import sys
p = sys.argv[1]; s = open(p).read()
def rep(a, b):
    global s
    assert s.count(a) == 1, a[:70]; s = s.replace(a, b)
rep('''y = lnorm(P, x, "blocks.0.ln1"); e = y @ P["knn_w"]''', '''y = lnorm(P, x, "blocks.0.ln1"); e = (y.detach() if GKNN_ST else y) @ P["knn_w"]''')
open(p, "w").write(s); print("gk-st fix applied")
