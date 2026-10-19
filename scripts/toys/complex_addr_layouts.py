"""complex_addr_layouts.py -- four ways to address a 3-level hierarchy with complex phasors (2026-10-19; zero GPU): A one register per level, B pure product + resonator, C weighted prefix sum, D many addresses superposed in one vector; under random phase noise. Results in docs/complex_planes.md."""
import numpy as np
rng=np.random.default_rng(0); D=256; N=64; L=3; T=500
C=[np.exp(1j*rng.uniform(0,2*np.pi,(N,D))) for _ in range(L)]   # one codebook per level
def clean(v,cb): return int(np.argmax((cb.conj()@v).real))
def noisy(v,s): return v*np.exp(1j*rng.normal(0,s,v.shape))       # phase noise from an imperfect encoder
for s in (0.0,0.5,1.0,1.5):
    a=b=c=r=0
    for _ in range(T):
        p=rng.integers(0,N,L)
        # A: SEPARATE REGISTERS (each level in its own plane group; width L*D)
        a+=all(clean(noisy(C[l][p[l]],s),C[l])==p[l] for l in range(L))
        # B: PURE PRODUCT in one D vector, factorized by a RESONATOR (alternating cleanup)
        x=noisy(C[0][p[0]]*C[1][p[1]]*C[2][p[2]],s)
        est=[cb.sum(0) for cb in C]
        for _ in range(50):
            for l in range(L):
                o=np.prod([est[k] for k in range(L) if k!=l],axis=0)
                w=(C[l].conj()@(x*o.conj())).real; est[l]=C[l].T@w; est[l]=est[l]/np.abs(est[l]).clip(1e-9)
        b+=all(clean(est[l],C[l])==p[l] for l in range(L))
        # C: GEMINI'S WEIGHTED PREFIX SUM in one D vector, read top-down
        P=[C[0][p[0]],C[0][p[0]]*C[1][p[1]],C[0][p[0]]*C[1][p[1]]*C[2][p[2]]]
        y=noisy(P[0]+0.5*P[1]+0.25*P[2],s); g=[]; pre=np.ones(D)
        for l in range(L):
            k=clean(y*pre.conj(),C[l]*pre if l else C[l]) if l==0 else int(np.argmax((( C[l]*pre).conj()@y).real)); g.append(k); pre=pre*C[l][k]
        c+=g==list(p)
        # D: STORAGE -- 25 such full addresses superposed in ONE register, query one by its known path
        S=sum(C[0][q[0]]*C[1][q[1]]*C[2][q[2]] for q in rng.integers(0,N,(25,L)))
        q=rng.integers(0,N,L); tgt=C[0][q[0]]*C[1][q[1]]
        r+=clean(noisy(S+tgt*C[2][q[2]],s)*tgt.conj(),C[2])==q[2]
    print(f"phase noise {s:.1f} rad | A separate registers {a/T:.3f} | B product+resonator {b/T:.3f} | C prefix-sum {c/T:.3f} | D 26 addresses in one vector {r/T:.3f}")
print(f"(D={D} planes per register, {N} codewords per level, {N**L:,} addresses)")
