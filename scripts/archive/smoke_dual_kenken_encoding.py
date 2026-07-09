"""Smoke + VISUALIZE the dual-view KenKen encoding (CPU, no model).

Verifies the crux — the dual gold (D[v,r] = column of value v in row r) and the
channeling factor membership — on one real puzzle, and that the PRIMAL half is
byte-identical to mycelium.kenken_data.encode_puzzle (so primal solve stays
comparable to the 0.796 baseline).

  .venv/bin/python3 scripts/smoke_dual_kenken_encoding.py
"""
import sys
import numpy as np

sys.path.insert(0, ".")
from mycelium.kenken_data import encode_puzzle, load_jsonl, N_CELLS, N_MAX  # noqa: E402
from mycelium.kenken_dual_data import encode_dual_puzzle, _dual_pos, S_DUAL  # noqa: E402

PATH = ".cache/kenken_test_curriculum.jsonl"


def main():
    recs = load_jsonl(PATH)
    n_cages_max = max(len(r["cages"]) for r in recs)
    # pick a full N=7 puzzle (no padding) for the clearest visualization
    rec = next(r for r in recs if int(r["N"]) == 7)
    N = int(rec["N"])
    sol = np.array(rec["solution"], dtype=np.int32)  # (N,N)
    print(f"=== puzzle N={N}  band={rec.get('band')}  n_givens={rec.get('n_givens')} ===")
    print("solution:")
    for r in range(N):
        print("  " + " ".join(str(int(x)) for x in sol[r]))

    p = encode_puzzle(rec, n_cages_max)
    d = encode_dual_puzzle(rec, n_cages_max)

    # ---- (1) PRIMAL half byte-identical to encode_puzzle ----
    ok_primal = (
        np.array_equal(d["input_cells"][:N_CELLS], p["input_cells"]) and
        np.array_equal(d["gold"][:N_CELLS], p["gold"]) and
        np.array_equal(d["cell_valid"][:N_CELLS], p["cell_valid"]) and
        np.array_equal(d["value_domain_mask"][:N_CELLS], p["value_domain_mask"])
    )
    print(f"\n[1] primal half == encode_puzzle: {'PASS' if ok_primal else 'FAIL'}")

    # ---- (2) dual gold: D[v,r] == column (1..N) where sol[r][c]==v ----
    gold = d["gold"]
    dual_ok = True
    dual_grid = np.zeros((N_MAX, N_MAX), dtype=np.int32)  # rows=v, cols=r
    for v in range(1, N + 1):
        for r in range(N):
            pos = _dual_pos(v, r)
            got = int(gold[pos])
            want = int(np.where(sol[r] == v)[0][0]) + 1  # 1-indexed column
            dual_grid[v - 1, r] = got
            if got != want:
                dual_ok = False
                print(f"   MISMATCH D[v={v},r={r}]: got {got} want {want}")
    print(f"[2] dual gold D[v,r]=col(v in row r): {'PASS' if dual_ok else 'FAIL'}")
    print("    dual gold grid (row=value v 1..N, col=row r 0..N-1):")
    for v in range(N):
        print("     v=%d: %s" % (v + 1, " ".join(str(int(x)) for x in dual_grid[v, :N])))

    # cross-check: each dual row (fixed v) is a permutation of 1..N (value once per col)
    perm_ok = all(sorted(dual_grid[v, :N]) == list(range(1, N + 1)) for v in range(N))
    print(f"[2b] each D[v,.] is a permutation (value once per column): "
          f"{'PASS' if perm_ok else 'FAIL'}")

    # ---- (3) channeling factor membership: per row r, 7 cells + 7 duals ----
    mem = d["membership"]            # (L, 98)
    lt = d["latent_type"]            # (L,)
    chan_rows = np.where(lt == 5)[0]
    print(f"\n[3] channeling factors (type 5): count={len(chan_rows)} (expect {N_MAX})")
    r0 = chan_rows[0]
    members = np.where(mem[r0] > 0)[0]
    cells = [m for m in members if m < N_CELLS]
    duals = [m for m in members if m >= N_CELLS]
    print(f"    channeling[row 0] members: {len(cells)} cells {cells}")
    print(f"                              {len(duals)} duals {duals}")
    exp_cells = [0 * N_MAX + c for c in range(N_MAX)]
    exp_duals = [_dual_pos(v, 0) for v in range(1, N_MAX + 1)]
    chan_ok = cells == exp_cells and duals == exp_duals
    print(f"    channeling[row 0] == row-0 cells + row-0 duals: {'PASS' if chan_ok else 'FAIL'}")

    # ---- (4) factor-type counts + shapes ----
    counts = {t: int((lt == t).sum()) for t in range(6)}
    print(f"\n[4] factor-type counts {counts} "
          f"(expect 0:7 1:7 2:{n_cages_max} 3:7 4:7 5:7)")
    L = mem.shape[0]
    shapes_ok = (mem.shape == (35 + n_cages_max, S_DUAL)
                 and gold.shape == (S_DUAL,)
                 and d["value_domain_mask"].shape == (S_DUAL, N_MAX)
                 and d["cell_cage_id"].shape == (N_CELLS,))
    print(f"[4b] shapes: membership {mem.shape} L={L} s_max={S_DUAL}  "
          f"{'PASS' if shapes_ok else 'FAIL'}")

    # ---- (5) cell_valid: 49 primal (N*N) + 49 dual (N*N) for N=7 = all 98 ----
    nvp = int(d["cell_valid"][:N_CELLS].sum())
    nvd = int(d["cell_valid"][N_CELLS:].sum())
    print(f"\n[5] valid: primal={nvp} (expect {N*N})  dual={nvd} (expect {N*N})")
    valid_ok = nvp == N * N and nvd == N * N

    # ---- (6) value_domain_mask on duals: columns 1..N legal ----
    vdm_dual_ok = True
    for v in range(1, N + 1):
        for r in range(N):
            row = d["value_domain_mask"][_dual_pos(v, r)]
            if not (row[:N].sum() == N and row[N:].sum() == 0):
                vdm_dual_ok = False
    print(f"[6] dual value_domain_mask = columns 1..N legal: "
          f"{'PASS' if vdm_dual_ok else 'FAIL'}")

    allp = ok_primal and dual_ok and perm_ok and chan_ok and shapes_ok and valid_ok and vdm_dual_ok
    print(f"\n=== ENCODING SMOKE: {'ALL PASS' if allp else 'FAIL'} ===")
    sys.exit(0 if allp else 1)


if __name__ == "__main__":
    main()
