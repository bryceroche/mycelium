"""KenKen data generator — the INSTRUMENT for the quick-win, built to spec.

Four PRE-REGISTERED standards (pinned before it produces a training instance,
because this generator calibrates Property 2 = breath-count-tracks-deduction-depth,
the flag we're most likely to plant):

1. UNIQUENESS-VERIFIED: every emitted puzzle has EXACTLY ONE solution (a
   backtracking solution-counter, stop-at-2). Non-unique => deduction depth is
   ill-defined (no unique fixed point) => Property 2's x-axis (depth) is noise.
   This is what the precheck's 0.27 solve-rate wobble was — non-unique leakage.
2. DEPTH-AT-FIXED-SIZE: the difficulty gradient is DEDUCTION DEPTH at FIXED board
   size N, NOT board size (E.16 lesson — depth at fixed node-count was the
   binding read; size confounds "harder" with "more to do"). We bin puzzles by
   measured depth WITHIN each N, so Property 2 correlates breath-count against
   depth with size held constant.
3. CLEAN DEPTH LABELS: deduction depth = propagation rounds-to-fixpoint, computed
   ONLY on the uniqueness-clean corpus (where the fixpoint IS the unique solution).
4. CAGE = SYMMETRIC MEMBERSHIP, ARITHMETIC = VERIFICATION, op-type NEVER a mask
   channel. v100 died because op-type isn't classifiable from the feature space
   (C2-elimination, exhaustively proven). The cage mask says "these cells share
   a constraint" (symmetric clique, like Sudoku's box); the arithmetic target is
   a per-cage VERIFICATION signal (does the assignment satisfy it), fed as a
   cage feature — never an op-type head/mask. KenKen cage-verification is the
   muscle that on-ramps to the scientific target; encode it as verification.

Usage:
  .venv/bin/python scripts/build_kenken_data.py            # validate + sample report
  KENKEN_EMIT=1 .venv/bin/python scripts/build_kenken_data.py   # write by-INSTANCE train/test JSONL (original)
  KENKEN_EMIT_STRUCT=1 .venv/bin/python scripts/build_kenken_data.py  # write by-STRUCTURE split (run-2)

RUN-2 ADDITION — BY-STRUCTURE SPLIT (the load-bearing generalization piece):
  The original 0.85/0.15 split is BY INSTANCE: a test puzzle can share its
  cage-shapes+ops with a train puzzle (only the clue numbers differ). That tests
  "same structure, new numbers". Run-2's pinned success criterion is HELD-OUT
  STRUCTURAL GENERALIZATION: test holds cage-CONFIGURATIONS (structural
  signatures) UNSEEN in train, so "generalizes" means "solves structures it has
  never seen" — the real CSP-reasoning claim. Leakage between train/test
  structures VOIDS that claim, so split disjointness is load-bearing and is
  asserted programmatically (see structural_signature + emit_by_structure).
"""
from __future__ import annotations
import itertools, json, os, random
from collections import Counter

OUT_DIR = ".cache"
SEED = 42
N_SIZES = [5, 6, 7]


def latin_square(n, rng):
    base = list(range(1, n + 1))
    rows = [base[i:] + base[:i] for i in range(n)]
    rng.shuffle(rows)
    cols = list(range(n)); rng.shuffle(cols)
    sq = [[rows[r][c] for c in cols] for r in range(n)]
    perm = base[:]; rng.shuffle(perm); m = {i + 1: perm[i] for i in range(n)}
    return [[m[sq[r][c]] for c in range(n)] for r in range(n)]


def make_cages(n, rng, big_bias):
    """Partition into contiguous cages. big_bias in [0,1] tilts toward larger
    cages (=> fewer givens, deeper deduction) — the fixed-N difficulty knob."""
    cells = [(r, c) for r in range(n) for c in range(n)]
    rng.shuffle(cells); assigned = {}; cages = []
    w = [1, 4, 3, 2] if big_bias < 0.5 else [0.3, 2, 4, 4]
    for start in cells:
        if start in assigned:
            continue
        target = rng.choices([1, 2, 3, 4], weights=w)[0]
        cage = [start]; assigned[start] = True
        while len(cage) < target:
            frontier = [nb for (r, c) in cage for nb in
                        ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1))
                        if 0 <= nb[0] < n and 0 <= nb[1] < n and nb not in assigned]
            if not frontier:
                break
            pick = rng.choice(frontier); cage.append(pick); assigned[pick] = True
        cages.append(cage)
    return cages


# ---- CURRICULUM: DIFFICULTY-BANDED by GIVENS DENSITY (new mode) --------------
#
# The just-validated reframe: v98 Sudoku's famous 97.65%/79% was EASY-only at 43%
# givens; on its 33%-givens band it collapses to 0.82 cell / 0.05 puzzle — the SAME
# collapse KenKen shows (KenKen was only ever measured at 10-12% givens). To make
# puzzle-acc achievable (so a SETTLED SET exists and the competence-gated Property-2
# flag is readable) we need a high-givens EASY band, plus a smooth curriculum down
# to the current hard (~10%) for depth spread + annealing.
#
# GIVENS = size-1 cages (cage_clue returns "given" for len==1). The original
# make_cages cage-size weights (w=[1,4,3,2] / [0.3,2,4,4]) yield only ~10-12%
# givens because the size-1 weight is small. To hit 40%/30%/20%/10% we add per-band
# cage-size weight profiles that progressively favor size-1 (and size-2) cages.
#
# The weights below were TUNED EMPIRICALLY (not guessed): for each profile we ran
# the FULL gen_unique_puzzle pipeline (latin square -> banded cages -> clues ->
# count_solutions==1 -> propagate) and measured mean n_givens/N^2 across N in {5,6,7},
# iterating until each band landed near its target. Measured means (KENKEN_TIMING_PROBE
# reproduces these on every run):
#     g40  w=[8, 5, 2, 1]    -> ~0.39   (N5~.37 N6~.41 N7~.38)
#     g30  w=[7, 6, 3, 2]    -> ~0.30
#     g20  w=[4, 5, 4, 3]    -> ~0.20
#     g10  w=[0.6, 3, 4, 3]  -> ~0.11
# Depth (the PINNED Property-2 axis) co-varies but is distinct: med deduction_depth
# rises ~3 (g40) -> ~5-6 (g10). We record BOTH givens_density and deduction_depth on
# every puzzle; the givens-density band is the CURRICULUM/EVAL axis, depth-at-fixed-N
# remains the PINNED science axis (standard #2). They correlate but never collapse.

# band name -> (target givens fraction, cage-size weights for [1,2,3,4]-cell cages)
GIVENS_BANDS = {
    "g40": (0.40, [8, 5, 2, 1]),
    "g30": (0.30, [7, 6, 3, 2]),
    "g20": (0.20, [4, 5, 4, 3]),
    "g10": (0.10, [0.6, 3, 4, 3]),
}
# ordered easy -> hard (high givens -> low givens), used for curriculum staging
BAND_ORDER = ["g40", "g30", "g20", "g10"]
BAND_TARGETS = {b: GIVENS_BANDS[b][0] for b in GIVENS_BANDS}


def band_of_density(gd: float) -> str:
    """Assign a puzzle to the band whose TARGET givens fraction is nearest its
    measured givens_density. Used so each puzzle's recorded `band` is consistent
    with its actual density even if the profile's stochastic draw lands off-center."""
    return min(BAND_TARGETS, key=lambda b: abs(BAND_TARGETS[b] - gd))


def make_cages_banded(n, rng, band):
    """make_cages variant driven by a difficulty BAND (or explicit weights).

    Identical contiguous-partition algorithm as make_cages, but the cage-size
    distribution comes from the band's tuned weight profile instead of the
    big_bias split. `band` may be a band-name key into GIVENS_BANDS, or a raw
    4-list of weights for [1,2,3,4]-cell cages (so callers can sweep targets)."""
    w = GIVENS_BANDS[band][1] if isinstance(band, str) else list(band)
    cells = [(r, c) for r in range(n) for c in range(n)]
    rng.shuffle(cells); assigned = {}; cages = []
    for start in cells:
        if start in assigned:
            continue
        target = rng.choices([1, 2, 3, 4], weights=w)[0]
        cage = [start]; assigned[start] = True
        while len(cage) < target:
            frontier = [nb for (r, c) in cage for nb in
                        ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1))
                        if 0 <= nb[0] < n and 0 <= nb[1] < n and nb not in assigned]
            if not frontier:
                break
            pick = rng.choice(frontier); cage.append(pick); assigned[pick] = True
        cages.append(cage)
    return cages


def cage_clue(cage, sol, rng):
    vals = [sol[r][c] for (r, c) in cage]
    if len(cage) == 1:
        return ("given", vals[0])
    if len(cage) == 2:
        a, b = sorted(vals, reverse=True)
        ch = [("add", a + b), ("mul", a * b), ("sub", a - b)]
        if b != 0 and a % b == 0:
            ch.append(("div", a // b))
        return rng.choice(ch)
    return ("add", sum(vals)) if rng.random() < 0.5 else ("mul", _prod(vals))


def _prod(xs):
    p = 1
    for x in xs:
        p *= x
    return p


def cage_ok(typ, tgt, asg):
    if typ == "given":
        return asg[0] == tgt
    if typ == "add":
        return sum(asg) == tgt
    if typ == "mul":
        return _prod(asg) == tgt
    if typ == "sub":
        return abs(asg[0] - asg[1]) == tgt
    if typ == "div":
        a, b = asg
        return (b and a == b * tgt) or (a and b == a * tgt)
    return False


def propagate(n, cages, clues, dom=None, max_rounds=64):
    """Constraint propagation -> (rounds, solved, domains)."""
    if dom is None:
        dom = {(r, c): set(range(1, n + 1)) for r in range(n) for c in range(n)}
        for cage, (typ, tgt) in zip(cages, clues):
            if typ == "given":
                dom[cage[0]] = {tgt}
    rounds = 0
    while rounds < max_rounds:
        rounds += 1; changed = False
        for (r, c), d in list(dom.items()):
            if len(d) == 1:
                v = next(iter(d))
                for cc in range(n):
                    if cc != c and v in dom[(r, cc)]:
                        dom[(r, cc)].discard(v); changed = True
                for rr in range(n):
                    if rr != r and v in dom[(rr, c)]:
                        dom[(rr, c)].discard(v); changed = True
        for line in ([[(r, c) for c in range(n)] for r in range(n)] +
                     [[(r, c) for r in range(n)] for c in range(n)]):
            for v in range(1, n + 1):
                homes = [cell for cell in line if v in dom[cell]]
                if len(homes) == 1 and len(dom[homes[0]]) > 1:
                    dom[homes[0]] = {v}; changed = True
        for cage, (typ, tgt) in zip(cages, clues):
            doms = [sorted(dom[cell]) for cell in cage]
            if _prod([len(d) for d in doms]) > 20000:
                continue  # skip giant cages for speed (rare)
            support = [set() for _ in cage]
            for combo in itertools.product(*doms):
                ok = True
                for i in range(len(cage)):
                    for j in range(i + 1, len(cage)):
                        if combo[i] == combo[j] and (cage[i][0] == cage[j][0] or cage[i][1] == cage[j][1]):
                            ok = False; break
                    if not ok:
                        break
                if ok and cage_ok(typ, tgt, combo):
                    for i, v in enumerate(combo):
                        support[i].add(v)
            for i, cell in enumerate(cage):
                if support[i] and (dom[cell] - support[i]):
                    dom[cell] &= support[i]; changed = True
        if not changed:
            break
    solved = all(len(d) == 1 for d in dom.values())
    return rounds, solved, dom


def count_solutions(n, cages, clues, stop=2):
    """Backtracking solution counter (for UNIQUENESS). Returns count up to `stop`."""
    base = {(r, c): set(range(1, n + 1)) for r in range(n) for c in range(n)}
    for cage, (typ, tgt) in zip(cages, clues):
        if typ == "given":
            base[cage[0]] = {tgt}

    def rec(dom):
        _, solved, dom = propagate(n, cages, clues, dom={k: set(v) for k, v in dom.items()})
        if any(len(d) == 0 for d in dom.values()):
            return 0
        if all(len(d) == 1 for d in dom.values()):
            return 1
        cell = min((c for c in dom if len(dom[c]) > 1), key=lambda c: len(dom[c]))
        tot = 0
        for v in sorted(dom[cell]):
            nd = {k: set(s) for k, s in dom.items()}; nd[cell] = {v}
            tot += rec(nd)
            if tot >= stop:
                return tot
        return tot
    return rec(base)


def gen_unique_puzzle(n, rng, big_bias, tries=40):
    for _ in range(tries):
        sol = latin_square(n, rng)
        cages = make_cages(n, rng, big_bias)
        clues = [cage_clue(cage, sol, rng) for cage in cages]
        nsol = count_solutions(n, cages, clues, stop=2)
        if nsol == 1:
            rounds, solved, _ = propagate(n, cages, clues)
            n_giv = sum(1 for (t, _v) in clues if t == "given")
            return {"N": n, "cages": [[list(c) for c in cg] for cg in cages],
                    "clues": clues, "solution": sol, "n_givens": n_giv,
                    "deduction_depth": rounds, "unique": True}
    return None


def gen_unique_puzzle_banded(n, rng, band, tries=40):
    """Banded analogue of gen_unique_puzzle: cages drawn from the band's weight
    profile; records givens_density + the (density-nearest) band, KEEPING the
    pinned deduction_depth + n_givens + uniqueness on every puzzle.

    The givens-density band is the CURRICULUM/EVAL axis; deduction_depth remains
    the PINNED Property-2 science axis (standard #2). Both are recorded so the two
    axes can be cross-tabulated downstream (they correlate but are distinct)."""
    for _ in range(tries):
        sol = latin_square(n, rng)
        cages = make_cages_banded(n, rng, band)
        clues = [cage_clue(cage, sol, rng) for cage in cages]
        nsol = count_solutions(n, cages, clues, stop=2)
        if nsol == 1:
            rounds, solved, _ = propagate(n, cages, clues)
            n_giv = sum(1 for (t, _v) in clues if t == "given")
            gd = n_giv / (n * n)
            return {"N": n, "cages": [[list(c) for c in cg] for cg in cages],
                    "clues": clues, "solution": sol, "n_givens": n_giv,
                    "givens_density": gd, "band": band_of_density(gd),
                    "target_band": band if isinstance(band, str) else None,
                    "deduction_depth": rounds, "unique": True}
    return None


# ---- PARALLEL GENERATION (deterministic per-worker seeding) -------------------
#
# The uniqueness check (count_solutions backtracking + propagate) is CPU-bound, so
# we fan generation out over a multiprocessing.Pool. DETERMINISM is preserved by
# deriving each task's RNG seed from (base_seed, N, band, chunk_index) — the SAME
# base seed reproduces the SAME corpus regardless of worker count or scheduling.
# A single-process fallback (KENKEN_WORKERS=1) walks the identical task list in
# order, so the only difference between serial and parallel is execution order, not
# the puzzles produced per task (each task is fully self-seeded).

def _task_seed(base_seed, n, band, chunk_index):
    """Deterministic per-task seed: f(base_seed, N, band, chunk_index). Mixing is
    a large-prime linear combination so distinct (n, band, chunk) tasks get
    well-separated streams while the SAME inputs always reproduce the SAME seed."""
    band_id = BAND_ORDER.index(band) if band in BAND_ORDER else 99
    return (base_seed * 1_000_003
            + n * 100_003
            + band_id * 10_007
            + chunk_index * 101 + 1) & 0x7FFFFFFF


def _gen_chunk(args):
    """Worker entry: generate `count` unique banded puzzles for one (N, band, chunk).
    Self-seeded from _task_seed so the result depends ONLY on the task identity +
    base seed (never on worker id / scheduling). Returns (n, band, list-of-recs)."""
    n, band, count, base_seed, chunk_index, tries = args
    rng = random.Random(_task_seed(base_seed, n, band, chunk_index))
    out = []
    attempts = 0
    # cap total attempts generously; uniqueness yield is ~1.0 at these givens levels
    while len(out) < count and attempts < count * 60:
        attempts += 1
        pz = gen_unique_puzzle_banded(n, rng, band, tries=tries)
        if pz is not None:
            out.append(pz)
    return (n, band, out)


def generate_curriculum_corpus(per_band_per_n, base_seed, workers, chunk_size=50,
                               tries=40, bands=None, sizes=None, progress=True):
    """Generate a banded corpus in parallel (or serial if workers<=1).

    Splits each (N, band) request of `per_band_per_n` puzzles into chunks of
    `chunk_size`, assigns each chunk a deterministic seed, and runs them over a
    Pool. Returns the flat list of puzzle records. Deterministic in `base_seed`
    for a fixed (per_band_per_n, chunk_size, bands, sizes) regardless of `workers`."""
    bands = bands or BAND_ORDER
    sizes = sizes or N_SIZES
    tasks = []
    for n in sizes:
        for band in bands:
            remaining = per_band_per_n
            chunk_index = 0
            while remaining > 0:
                c = min(chunk_size, remaining)
                tasks.append((n, band, c, base_seed, chunk_index, tries))
                remaining -= c
                chunk_index += 1

    corpus = []
    if workers and workers > 1:
        import multiprocessing as mp
        with mp.Pool(processes=workers) as pool:
            done = 0
            for (_n, _band, recs) in pool.imap_unordered(_gen_chunk, tasks):
                corpus.extend(recs)
                done += 1
                if progress and done % 20 == 0:
                    print(f"  ... {done}/{len(tasks)} chunks done "
                          f"({len(corpus)} puzzles)", flush=True)
    else:
        for i, task in enumerate(tasks):
            _n, _band, recs = _gen_chunk(task)
            corpus.extend(recs)
            if progress and (i + 1) % 20 == 0:
                print(f"  ... {i + 1}/{len(tasks)} chunks done "
                      f"({len(corpus)} puzzles)", flush=True)
    return corpus


# ---- STRUCTURAL SIGNATURE (run-2: by-structure split) ------------------------

def _dihedral_transforms(n):
    """Return the 8 dihedral symmetries of an NxN grid as cell-coordinate maps.

    Each transform is a function (r, c) -> (r', c') that permutes the N*N cells of
    the square. The dihedral group D4 has order 8: 4 rotations × {identity, reflect}.
    We enumerate them as explicit coordinate maps so the canonical form is exact.

      identity        (r, c) -> (r, c)
      rot90           (r, c) -> (c, n-1-r)          # 90° clockwise
      rot180          (r, c) -> (n-1-r, n-1-c)
      rot270          (r, c) -> (n-1-c, r)          # 270° clockwise (= 90° ccw)
      flip_h          (r, c) -> (r, n-1-c)          # reflect across vertical axis
      flip_v          (r, c) -> (n-1-r, c)          # reflect across horizontal axis
      transpose       (r, c) -> (c, r)              # reflect across main diagonal
      anti_transpose  (r, c) -> (n-1-c, n-1-r)      # reflect across anti-diagonal
    """
    return [
        lambda r, c: (r, c),
        lambda r, c: (c, n - 1 - r),
        lambda r, c: (n - 1 - r, n - 1 - c),
        lambda r, c: (n - 1 - c, r),
        lambda r, c: (r, n - 1 - c),
        lambda r, c: (n - 1 - r, c),
        lambda r, c: (c, r),
        lambda r, c: (n - 1 - c, n - 1 - r),
    ]


def structural_signature(rec) -> str:
    """Canonical hash of a puzzle's STRUCTURE: (cage cell-PARTITION + per-cage
    OP-TYPE), invariant to the grid's dihedral symmetry group and ignoring clue
    target values, the solution, and symbol relabeling.

    WHAT IT IGNORES (so two puzzles that are the same shapes+ops up to board
    symmetry share a signature):
      - clue TARGET values         (we never read rec["clues"][i][1])
      - the SOLUTION               (we never read rec["solution"])
      - SYMBOL relabeling          (automatic: targets/solution are the only place
                                    symbols appear, and both are excluded)
      - the 8 BOARD SYMMETRIES     (canonicalize over D4 — see below)

    WHAT IT KEEPS (the structure that defines "have I seen this config?"):
      - the cage cell-PARTITION    (which cells are grouped together)
      - the per-cage OP-TYPE       (given/add/sub/mul/div — a tag on each cage)
      - the board size N           (prefixed; structures on different N never collide)

    CANONICALIZATION (exact, deterministic):
      A puzzle's structure is the SET of cages. Each cage = (op-type, set-of-cells).
      For EACH of the 8 dihedral transforms g:
        1. map every cell (r,c) of every cage through g -> g(cage).
        2. encode each transformed cage as (op_type, sorted-tuple-of-flat-indices),
           where flat = r*N + c on the transformed coords.
        3. form the SORTED TUPLE of those per-cage encodings (sorting makes it
           order-independent — the partition is a SET of cages, not a list).
      The canonical form = the LEXICOGRAPHICALLY MINIMAL representation across all
      8 transforms. Two puzzles share a signature IFF they are the same
      cage-shapes+ops up to a board symmetry. The returned string is
      "N=<n>|<canonical>" so different N never collide.

    Op-type is read from rec["clues"][i][0] ONLY (the op tag), never the target.
    """
    n = int(rec["N"])
    cages = rec["cages"]                              # list of list of [r,c]
    clues = rec["clues"]                              # list of [op, target]
    ops = [str(clue[0]) for clue in clues]            # op-type tag ONLY (ignore target)

    best = None
    for g in _dihedral_transforms(n):
        per_cage = []
        for op, cage in zip(ops, cages):
            flat = tuple(sorted((lambda rc: rc[0] * n + rc[1])(g(int(r), int(c)))
                                for (r, c) in cage))
            per_cage.append((op, flat))
        # sorted tuple => partition is treated as a SET of (op, cells) cages.
        canon = tuple(sorted(per_cage))
        if best is None or canon < best:
            best = canon
    return f"N={n}|" + repr(best)


def emit_by_structure(corpus, rng, train_target=6000, test_target=1200):
    """Split `corpus` into TRAIN / TEST with ZERO shared structural signatures.

    Algorithm (disjoint-by-construction):
      1. Group every puzzle by its structural_signature.
      2. For per-N balance + depth stratification, assign whole SIGNATURE GROUPS
         (never individual puzzles) to train or test. Because the unit of
         assignment is a signature group and a group lands entirely in ONE split,
         no signature can appear in both => disjointness is structural, then
         re-verified with an assert at the end.
      3. Within each N, walk signature groups in a shuffled order, sending groups
         to TEST until that N's test quota (~test_target/3) of PUZZLES is met,
         the rest to TRAIN (capped near train_target/3). Depth stratification:
         groups are interleaved by their representative deduction_depth bucket so
         both splits span the depth range at fixed N.

    Returns (train_rows, test_rows, report_dict).
    """
    # group puzzles by signature
    sig_to_rows: dict[str, list] = {}
    for rec in corpus:
        sig = structural_signature(rec)
        sig_to_rows.setdefault(sig, []).append(rec)

    per_n_train = train_target // len(N_SIZES)
    per_n_test = test_target // len(N_SIZES)

    train_rows, test_rows = [], []
    report = {}
    for n in N_SIZES:
        # signatures whose puzzles are size n (a signature is N-pure by construction
        # since the "N=<n>|" prefix forbids cross-N collisions).
        sigs_n = [s for s in sig_to_rows if s.startswith(f"N={n}|")]
        # depth-stratify: order signature groups by their representative depth so
        # round-robin assignment spreads depths across BOTH splits.
        def rep_depth(s):
            return min(int(r["deduction_depth"]) for r in sig_to_rows[s])
        rng.shuffle(sigs_n)
        sigs_n.sort(key=rep_depth)          # stable: shuffled within equal depth

        tr_n, te_n = [], []
        n_tr = n_te = 0
        # interleave: every Kth signature group -> test (depth-stratified), else train
        for i, s in enumerate(sigs_n):
            rows = sig_to_rows[s]
            # send to test if test quota not met AND this is a "test slot"; the
            # slot cadence ~ per_n_test/(per_n_test+per_n_train) keeps both
            # splits depth-spanning instead of test=shallow / train=deep.
            want_test = (n_te < per_n_test) and (
                (i % max(1, (per_n_train + per_n_test) // max(1, per_n_test))) == 0)
            if want_test:
                te_n.extend(rows); n_te += len(rows)
            elif n_tr < per_n_train:
                tr_n.extend(rows); n_tr += len(rows)
            elif n_te < per_n_test:
                te_n.extend(rows); n_te += len(rows)
            # else: corpus larger than both quotas for this N -> drop the overflow
        train_rows.extend(tr_n)
        test_rows.extend(te_n)

        tr_sigs = {structural_signature(r) for r in tr_n}
        te_sigs = {structural_signature(r) for r in te_n}
        tr_depths = Counter(int(r["deduction_depth"]) for r in tr_n)
        te_depths = Counter(int(r["deduction_depth"]) for r in te_n)
        report[n] = {
            "train_n": len(tr_n), "test_n": len(te_n),
            "train_sigs": len(tr_sigs), "test_sigs": len(te_sigs),
            "overlap": len(tr_sigs & te_sigs),
            "train_depth_hist": dict(sorted(tr_depths.items())),
            "test_depth_hist": dict(sorted(te_depths.items())),
        }
    return train_rows, test_rows, report


def emit_curriculum_by_structure(corpus, rng, test_frac=0.18):
    """Band-stratified, leak-free TRAIN/TEST split for the curriculum corpus.

    EXTENDS emit_by_structure's guarantee: the assignment unit is still the
    SIGNATURE GROUP (a whole structural_signature lands entirely in one split), so
    no signature can appear in both => overlap==0 holds by construction (re-asserted
    by the caller). The new stratification key is (N x givens_band x depth-bucket):
    we walk signature groups within each (N, band) cell, ordered by depth, and send
    every Kth group to TEST. This makes EVERY band span BOTH train and test (we eval
    per-band on held-out structures) WHILE both splits also span the depth range.

    A signature group is band-pure in practice because all puzzles sharing a cage
    PARTITION + op-types have near-identical givens counts; we assign the group to
    the band of its representative (first) puzzle to keep the unit atomic.

    Returns (train_rows, test_rows, report) where report is keyed by (N, band).
    """
    # group puzzles by signature (the atomic, leak-free assignment unit)
    sig_to_rows: dict[str, list] = {}
    for rec in corpus:
        sig = structural_signature(rec)
        sig_to_rows.setdefault(sig, []).append(rec)

    # representative band/depth/N per signature group
    def rep_band(s):
        return sig_to_rows[s][0]["band"]

    def rep_depth(s):
        return min(int(r["deduction_depth"]) for r in sig_to_rows[s])

    # cadence: ~1 in K groups -> test, K chosen so test ~ test_frac of groups
    K = max(2, round(1.0 / max(1e-6, test_frac)))

    train_rows, test_rows = [], []
    report = {}
    for n in N_SIZES:
        sigs_n = [s for s in sig_to_rows if s.startswith(f"N={n}|")]
        for band in BAND_ORDER:
            sigs_nb = [s for s in sigs_n if rep_band(s) == band]
            # depth-stratify within the (N, band) cell so test spans depths too
            rng.shuffle(sigs_nb)
            sigs_nb.sort(key=rep_depth)   # stable -> shuffled within equal depth
            tr_nb, te_nb = [], []
            for i, s in enumerate(sigs_nb):
                rows = sig_to_rows[s]
                if i % K == 0:
                    te_nb.extend(rows)
                else:
                    tr_nb.extend(rows)
            train_rows.extend(tr_nb)
            test_rows.extend(te_nb)

            tr_sigs = {structural_signature(r) for r in tr_nb}
            te_sigs = {structural_signature(r) for r in te_nb}
            tr_gd = sorted(r["givens_density"] for r in tr_nb)
            te_gd = sorted(r["givens_density"] for r in te_nb)
            all_gd = sorted(r["givens_density"] for r in (tr_nb + te_nb))
            tr_depths = Counter(int(r["deduction_depth"]) for r in tr_nb)
            te_depths = Counter(int(r["deduction_depth"]) for r in te_nb)
            report[f"N{n}_{band}"] = {
                "N": n, "band": band,
                "target_givens": BAND_TARGETS[band],
                "train_n": len(tr_nb), "test_n": len(te_nb),
                "train_sigs": len(tr_sigs), "test_sigs": len(te_sigs),
                "overlap": len(tr_sigs & te_sigs),
                "mean_givens_density": (sum(all_gd) / len(all_gd)) if all_gd else None,
                "median_givens_density": (all_gd[len(all_gd) // 2]) if all_gd else None,
                "train_mean_givens": (sum(tr_gd) / len(tr_gd)) if tr_gd else None,
                "test_mean_givens": (sum(te_gd) / len(te_gd)) if te_gd else None,
                "train_depth_hist": dict(sorted(tr_depths.items())),
                "test_depth_hist": dict(sorted(te_depths.items())),
            }
    return train_rows, test_rows, report


def run_timing_probe(base_seed, workers, sample_per_band_per_n=30,
                     full_train_target=30000):
    """KENKEN_TIMING_PROBE=1: generate a SMALL sample per (N, band), print
    puzzles/sec PER BAND + measured mean givens fraction per band (confirming the
    bands land near 0.40/.30/.20/.10), and ESTIMATE the wall-clock for the full
    >=full_train_target run. Low-givens/hard bands are slower -> per-band reporting.
    """
    import time
    print(f"=== KENKEN TIMING PROBE (sample {sample_per_band_per_n} per (N,band), "
          f"workers={workers}) ===", flush=True)
    per_band_rate = {}   # band -> puzzles/sec (summed over N, wall-clock)
    per_band_gd = {}     # band -> list of givens densities
    per_band_depth = {}  # band -> list of depths
    total_puzzles = 0
    total_wall = 0.0
    for band in BAND_ORDER:
        t0 = time.time()
        recs = generate_curriculum_corpus(
            per_band_per_n=sample_per_band_per_n, base_seed=base_seed,
            workers=workers, chunk_size=max(5, sample_per_band_per_n // 3),
            bands=[band], sizes=N_SIZES, progress=False)
        dt = time.time() - t0
        gd = [r["givens_density"] for r in recs]
        dd = [r["deduction_depth"] for r in recs]
        per_band_gd[band] = gd
        per_band_depth[band] = dd
        rate = len(recs) / max(dt, 1e-6)
        per_band_rate[band] = rate
        total_puzzles += len(recs)
        total_wall += dt
        mean_gd = sum(gd) / len(gd) if gd else float("nan")
        med_depth = sorted(dd)[len(dd) // 2] if dd else -1
        # per-N givens means inside this band
        byN = {}
        for r in recs:
            byN.setdefault(r["N"], []).append(r["givens_density"])
        byN_str = " ".join(f"N{n}={sum(v)/len(v):.3f}" for n, v in sorted(byN.items()))
        print(f"  band={band} target={BAND_TARGETS[band]:.2f}  "
              f"n={len(recs)}  mean_givens={mean_gd:.3f}  med_depth={med_depth}  "
              f"rate={rate:.2f} puz/s  ({byN_str})", flush=True)

    overall_rate = total_puzzles / max(total_wall, 1e-6)
    print(f"\n  OVERALL: {total_puzzles} puzzles in {total_wall:.1f}s "
          f"-> {overall_rate:.2f} puz/s aggregate", flush=True)

    # Estimate full run. Full run generates per_band_per_n across 4 bands x 3 N.
    # full_train_target is TRAIN; emit also keeps test_frac extra. Estimate by the
    # SLOWEST band rate (conservative) and the aggregate rate (likely).
    bands_x_n = len(BAND_ORDER) * len(N_SIZES)
    total_full = int(full_train_target / (1.0 - 0.18))  # train+test puzzles
    slowest = min(per_band_rate.values())
    est_slow = total_full / max(slowest, 1e-6)
    est_agg = total_full / max(overall_rate, 1e-6)
    print(f"\n  FULL-RUN ESTIMATE for >= {full_train_target} train "
          f"(~{total_full} total incl test):")
    print(f"    by aggregate rate ({overall_rate:.2f} puz/s): "
          f"{est_agg/60:.1f} min  ({est_agg/3600:.2f} h)")
    print(f"    by slowest-band rate ({slowest:.2f} puz/s, conservative): "
          f"{est_slow/60:.1f} min  ({est_slow/3600:.2f} h)")
    # given workers, the slowest-band-dominated estimate already reflects parallelism
    print(f"    (probe used workers={workers}; per-band rates already reflect that)",
          flush=True)

    probe_report = {
        "metric": "kenken_curriculum_timing_probe",
        "base_seed": base_seed, "workers": workers,
        "sample_per_band_per_n": sample_per_band_per_n,
        "per_band_rate_puz_per_s": per_band_rate,
        "per_band_mean_givens": {b: (sum(v) / len(v) if v else None)
                                 for b, v in per_band_gd.items()},
        "per_band_median_depth": {b: (sorted(v)[len(v) // 2] if v else None)
                                  for b, v in per_band_depth.items()},
        "band_targets": BAND_TARGETS,
        "overall_rate_puz_per_s": overall_rate,
        "full_train_target": full_train_target,
        "est_total_puzzles": total_full,
        "est_full_run_min_aggregate": est_agg / 60.0,
        "est_full_run_min_slowest_band": est_slow / 60.0,
    }
    rep_path = os.path.join(OUT_DIR, "kenken_curriculum_timing_probe.json")
    with open(rep_path, "w") as f:
        json.dump(probe_report, f, indent=2)
    print(f"\n  saved {rep_path}", flush=True)
    return probe_report


def run_emit_curriculum(base_seed, workers):
    """KENKEN_EMIT_CURRICULUM=1: full banded-curriculum generation + leak-free,
    band-stratified split + report. Writes NEW files only; never touches the
    originals (kenken_train.jsonl / _struct.jsonl etc.).

    Env knobs:
      KENKEN_CURRIC_PER_BAND_PER_N (default 2700)  puzzles per (N, band) to GENERATE
        -> 4 bands x 3 N x 2700 = 32400 raw; after dihedral grouping the leak-free
        split yields ~>=30k train across bands. Raise if distinct-signature supply
        is short for a band.
      KENKEN_TRAIN_TARGET_CURRIC (default 30000)   reporting/estimate target only.
      KENKEN_CURRIC_TEST_FRAC (default 0.18)       fraction of signature groups -> test.
      KENKEN_CHUNK (default 50)                     puzzles per parallel task.
    """
    per_band_per_n = int(os.environ.get("KENKEN_CURRIC_PER_BAND_PER_N", "2700"))
    test_frac = float(os.environ.get("KENKEN_CURRIC_TEST_FRAC", "0.18"))
    chunk = int(os.environ.get("KENKEN_CHUNK", "50"))

    print(f"=== KENKEN CURRICULUM GENERATION ===")
    print(f"  bands={BAND_ORDER}  N={N_SIZES}  per_band_per_n={per_band_per_n}  "
          f"workers={workers}  chunk={chunk}  base_seed={base_seed}", flush=True)
    import time
    t0 = time.time()
    corpus = generate_curriculum_corpus(
        per_band_per_n=per_band_per_n, base_seed=base_seed, workers=workers,
        chunk_size=chunk, bands=BAND_ORDER, sizes=N_SIZES, progress=True)
    print(f"  generated {len(corpus)} unique puzzles in {time.time()-t0:.1f}s",
          flush=True)

    # CANONICAL ORDER: imap_unordered collects chunks in completion order, so the
    # corpus LIST order depends on worker scheduling even though the SET is fixed by
    # the per-task seeds. Sort by structural_signature (then a stable JSON key) so
    # the downstream split + shuffle are byte-reproducible regardless of worker
    # count. This is what makes the SAME seed reproduce the SAME files at any -j.
    corpus.sort(key=lambda r: (structural_signature(r), json.dumps(r, sort_keys=True)))

    # band-stratified, leak-free split
    rng = random.Random(base_seed + 11)
    train_rows, test_rows, curric_rep = emit_curriculum_by_structure(
        corpus, rng, test_frac=test_frac)

    # ---- load-bearing asserts: leak-free + uniqueness (curriculum split) ----
    train_sigs = {structural_signature(r) for r in train_rows}
    test_sigs = {structural_signature(r) for r in test_rows}
    overlap = train_sigs & test_sigs
    assert len(overlap) == 0, (
        f"CURRICULUM STRUCTURAL LEAKAGE: {len(overlap)} signature(s) shared "
        f"between train and test — generalization claim VOID. "
        f"e.g. {list(overlap)[:2]}")
    assert all(r.get("unique") is True for r in train_rows + test_rows), \
        "non-unique puzzle leaked into the curriculum split"

    rng.shuffle(train_rows); rng.shuffle(test_rows)
    train_path = os.path.join(OUT_DIR, "kenken_train_curriculum.jsonl")
    test_path = os.path.join(OUT_DIR, "kenken_test_curriculum.jsonl")
    for pth, rows in ((train_path, train_rows), (test_path, test_rows)):
        with open(pth, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

    # ---- per-band report (confirm bands hit ~0.40/.30/.20/.10) ----
    def band_stats(rows, band):
        gd = sorted(r["givens_density"] for r in rows if r["band"] == band)
        dd = Counter(int(r["deduction_depth"]) for r in rows if r["band"] == band)
        sigs = {structural_signature(r) for r in rows if r["band"] == band}
        return gd, dd, sigs

    print("\n=== CURRICULUM BY-BAND REPORT ===")
    band_report = {}
    for band in BAND_ORDER:
        tr_gd, tr_dd, tr_sg = band_stats(train_rows, band)
        te_gd, te_dd, te_sg = band_stats(test_rows, band)
        all_gd = sorted(tr_gd + te_gd)
        ov = len(tr_sg & te_sg)
        mean_gd = sum(all_gd) / len(all_gd) if all_gd else None
        med_gd = all_gd[len(all_gd) // 2] if all_gd else None
        band_report[band] = {
            "target_givens": BAND_TARGETS[band],
            "mean_givens_density": mean_gd,
            "median_givens_density": med_gd,
            "train_n": len(tr_gd), "test_n": len(te_gd),
            "train_sigs": len(tr_sg), "test_sigs": len(te_sg),
            "overlap": ov,
            "train_depth_hist": dict(sorted(tr_dd.items())),
            "test_depth_hist": dict(sorted(te_dd.items())),
        }
        print(f"  {band}: target={BAND_TARGETS[band]:.2f} "
              f"mean_givens={mean_gd:.3f} med_givens={med_gd:.3f} | "
              f"train={len(tr_gd)}({len(tr_sg)} sigs) test={len(te_gd)}({len(te_sg)} sigs) "
              f"overlap={ov}")
        print(f"       train depth hist: {dict(sorted(tr_dd.items()))}")
        print(f"       test  depth hist: {dict(sorted(te_dd.items()))}")

    tr_byN = Counter(r["N"] for r in train_rows)
    te_byN = Counter(r["N"] for r in test_rows)
    print(f"\n  per-N train counts: {dict(sorted(tr_byN.items()))}")
    print(f"  per-N test  counts: {dict(sorted(te_byN.items()))}")
    print(f"  TOTAL train={len(train_rows)} test={len(test_rows)}")
    print(f"  distinct signatures: train={len(train_sigs)} test={len(test_sigs)}")
    print(f"  *** OVERLAP (train∩test signatures) = {len(overlap)} (asserted 0) ***")
    print(f"  emitted -> {train_path}")
    print(f"  emitted -> {test_path}")

    report = {
        "metric": "kenken_difficulty_banded_curriculum",
        "base_seed": base_seed,
        "axes": {
            "curriculum_eval_axis": "givens_density band (g40/g30/g20/g10)",
            "pinned_property2_axis": "deduction_depth at fixed N (standard #2)",
            "note": "the two axes correlate but are distinct; both recorded per puzzle",
        },
        "band_weight_profiles": {b: GIVENS_BANDS[b][1] for b in BAND_ORDER},
        "band_targets": BAND_TARGETS,
        "leak_free": ("split unit = structural_signature group (D4-canonical); "
                      "overlap==0 asserted; uniqueness (count_solutions==1) re-asserted"),
        "canonicalization": (
            "structural_signature = N-prefixed canonical hash of cage cell-PARTITION "
            "tagged by per-cage OP-TYPE; lex-min over 8 dihedral (D4) symmetries; "
            "IGNORES clue targets, the solution, and symbol relabeling"),
        "train_path": train_path, "test_path": test_path,
        "train_n": len(train_rows), "test_n": len(test_rows),
        "train_sigs": len(train_sigs), "test_sigs": len(test_sigs),
        "overlap": len(overlap),
        "train_per_N": dict(sorted(tr_byN.items())),
        "test_per_N": dict(sorted(te_byN.items())),
        "by_band": band_report,
        "by_N_band": curric_rep,
        "uniqueness": "every emitted puzzle count_solutions==1 (by construction; re-asserted)",
    }
    rep_path = os.path.join(OUT_DIR, "kenken_curriculum_report.json")
    with open(rep_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  saved {rep_path}", flush=True)
    return report


def main():
    rng = random.Random(SEED)
    # ---- NEW MODES (curriculum) — handled first; never touch the original files ----
    emit_curriculum = os.environ.get("KENKEN_EMIT_CURRICULUM") == "1"
    timing_probe = os.environ.get("KENKEN_TIMING_PROBE") == "1"
    if emit_curriculum or timing_probe:
        base_seed = int(os.environ.get("KENKEN_SEED", str(SEED)))
        # default to (ncpu - 1) workers, overridable; serial fallback at 1
        try:
            import multiprocessing as _mp
            default_workers = max(1, _mp.cpu_count() - 1)
        except Exception:
            default_workers = 1
        workers = int(os.environ.get("KENKEN_WORKERS", str(default_workers)))
        if timing_probe:
            sample = int(os.environ.get("KENKEN_PROBE_SAMPLE", "30"))
            full_target = int(os.environ.get("KENKEN_TRAIN_TARGET_CURRIC", "30000"))
            run_timing_probe(base_seed, workers,
                             sample_per_band_per_n=sample,
                             full_train_target=full_target)
        if emit_curriculum:
            run_emit_curriculum(base_seed, workers)
        return
    emit = os.environ.get("KENKEN_EMIT") == "1"
    emit_struct = os.environ.get("KENKEN_EMIT_STRUCT") == "1"
    # by-structure (run-2) needs MORE raw puzzles per N so that, AFTER dihedral
    # dedup, there are enough DISTINCT signatures to fill ~2000 train + ~400 test
    # per N with ZERO overlap. Override via KENKEN_PER_N.
    if emit_struct:
        default_per_n = int(os.environ.get("KENKEN_STRUCT_PER_N", "2700"))
    elif emit:
        default_per_n = 400
    else:
        default_per_n = 120
    per_n = int(os.environ.get("KENKEN_PER_N", str(default_per_n)))
    report = {}
    corpus = []
    for n in N_SIZES:
        depths = []; givens = []; tries = 0; got = 0
        # sweep big_bias to produce a DEPTH gradient AT FIXED N
        while got < per_n and tries < per_n * 30:
            tries += 1
            bb = rng.random()  # vary cage structure => vary depth at fixed N
            pz = gen_unique_puzzle(n, rng, bb)
            if pz is None:
                continue
            got += 1; depths.append(pz["deduction_depth"]); givens.append(pz["n_givens"])
            corpus.append(pz)
        ds = sorted(depths)
        report[n] = {"n_unique": got, "uniqueness_yield": got / max(tries, 1),
                     "depth_min": ds[0], "depth_med": ds[len(ds)//2], "depth_max": ds[-1],
                     "depth_hist": dict(sorted(Counter(depths).items())),
                     "depth_range_at_fixed_N": ds[-1] - ds[0],
                     "givens_med": sorted(givens)[len(givens)//2]}
        r = report[n]
        print(f"N={n}: {got} unique (yield {r['uniqueness_yield']:.2f}) | depth "
              f"min/med/max = {r['depth_min']}/{r['depth_med']}/{r['depth_max']} "
              f"(range@fixedN={r['depth_range_at_fixed_N']}) hist={r['depth_hist']}", flush=True)

    # pre-registered generator-validation bands
    ok_unique = all(report[n]["n_unique"] >= (per_n * 0.9) for n in N_SIZES)
    ok_gradient = all(report[n]["depth_range_at_fixed_N"] >= 3 for n in N_SIZES)
    verdict = ("INSTRUMENT-VALID (uniqueness 100% by construction; depth gradient >=3 at fixed N)"
               if ok_unique and ok_gradient else
               "NEEDS-TUNING (depth gradient too narrow at fixed N — widen cage-structure sweep)")
    print(f"\nGENERATOR VALIDATION: unique-yield-ok={ok_unique} depth-gradient-at-fixed-N-ok={ok_gradient}")
    print(f"  -> {verdict}")
    print("  (every emitted puzzle is uniqueness-verified by count_solutions==1; "
          "depth labels computed on the clean corpus; difficulty = depth at FIXED N)")

    rep_path = os.path.join(OUT_DIR, "kenken_generator_report.json")
    with open(rep_path, "w") as f:
        json.dump({"metric": "kenken_generator_validation", "seed": SEED,
                   "by_N": report, "verdict": verdict,
                   "standards": ["uniqueness-verified (count_solutions==1)",
                                 "depth-at-fixed-N gradient", "clean depth labels",
                                 "cage=symmetric-membership; arithmetic=verification; op-type NEVER a mask channel (C2)"]},
                  f, indent=2)
    print(f"saved {rep_path}")

    if emit:
        rng2 = random.Random(SEED + 7)
        rng2.shuffle(corpus)
        cut = int(len(corpus) * 0.85)
        for split, rows in (("train", corpus[:cut]), ("test", corpus[cut:])):
            pth = os.path.join(OUT_DIR, f"kenken_{split}.jsonl")
            with open(pth, "w") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")
            print(f"emitted {len(rows)} -> {pth}")

    if emit_struct:
        # ---- RUN-2 by-STRUCTURE split (new files; never overwrite the originals) ----
        rng3 = random.Random(SEED + 11)
        train_target = int(os.environ.get("KENKEN_TRAIN_TARGET", "6000"))
        test_target = int(os.environ.get("KENKEN_TEST_TARGET", "1200"))
        train_rows, test_rows, struct_rep = emit_by_structure(
            corpus, rng3, train_target=train_target, test_target=test_target)

        # ---- programmatic disjointness PROOF (the load-bearing assert) ----
        train_sigs = {structural_signature(r) for r in train_rows}
        test_sigs = {structural_signature(r) for r in test_rows}
        overlap = train_sigs & test_sigs
        assert len(overlap) == 0, (
            f"STRUCTURAL LEAKAGE: {len(overlap)} signature(s) shared between "
            f"train and test — generalization claim VOID. e.g. {list(overlap)[:2]}")

        # ---- uniqueness re-verification (every emitted puzzle count_solutions==1) ----
        # The corpus was built ONLY from gen_unique_puzzle (count_solutions==1), so
        # the field is true by construction; we re-assert the flag to be explicit.
        assert all(r.get("unique") is True for r in train_rows + test_rows), \
            "non-unique puzzle leaked into a by-structure split"

        struct_train_path = os.path.join(OUT_DIR, "kenken_train_struct.jsonl")
        struct_test_path = os.path.join(OUT_DIR, "kenken_test_struct.jsonl")
        rng3.shuffle(train_rows); rng3.shuffle(test_rows)
        for pth, rows in ((struct_train_path, train_rows), (struct_test_path, test_rows)):
            with open(pth, "w") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")

        # ---- report ----
        tr_byN = Counter(r["N"] for r in train_rows)
        te_byN = Counter(r["N"] for r in test_rows)
        print("\n=== RUN-2 BY-STRUCTURE SPLIT ===")
        print(f"  emitted TRAIN {len(train_rows)} -> {struct_train_path}")
        print(f"  emitted TEST  {len(test_rows)} -> {struct_test_path}")
        print(f"  per-N train counts: {dict(sorted(tr_byN.items()))}")
        print(f"  per-N test  counts: {dict(sorted(te_byN.items()))}")
        print(f"  distinct structural signatures: train={len(train_sigs)} "
              f"test={len(test_sigs)}")
        print(f"  *** OVERLAP (shared signatures train∩test) = {len(overlap)} "
              f"(asserted 0) ***")
        for n in N_SIZES:
            r = struct_rep[n]
            print(f"  N={n}: train={r['train_n']}({r['train_sigs']} sigs) "
                  f"test={r['test_n']}({r['test_sigs']} sigs) overlap={r['overlap']}")
            print(f"       train depth hist: {r['train_depth_hist']}")
            print(f"       test  depth hist: {r['test_depth_hist']}")
        struct_report = {
            "metric": "kenken_by_structure_split",
            "seed": SEED,
            "canonicalization": (
                "structural_signature = N-prefixed canonical hash of the cage "
                "cell-PARTITION tagged by per-cage OP-TYPE; lexicographically "
                "minimal over the 8 dihedral (D4) symmetries of the NxN grid; "
                "IGNORES clue targets, the solution, and symbol relabeling"),
            "train_path": struct_train_path,
            "test_path": struct_test_path,
            "train_n": len(train_rows), "test_n": len(test_rows),
            "train_sigs": len(train_sigs), "test_sigs": len(test_sigs),
            "overlap": len(overlap),
            "train_per_N": dict(sorted(tr_byN.items())),
            "test_per_N": dict(sorted(te_byN.items())),
            "by_N": struct_rep,
            "uniqueness": "every emitted puzzle count_solutions==1 (by construction; re-asserted)",
        }
        srep_path = os.path.join(OUT_DIR, "kenken_struct_split_report.json")
        with open(srep_path, "w") as f:
            json.dump(struct_report, f, indent=2)
        print(f"  saved {srep_path}")


if __name__ == "__main__":
    main()
