"""doors.py — THE ONE-DOOR MODULE (2026-07-31, built on the law sweep's
ranking under the standing word). One authority per law; freedom in the
harness. New code imports from here; existing correct inline copies
migrate on next touch (no new inline copies — the migration policy).

The three doors + the m-bound lookup + the view-seed registry:

1. certify_unique — the uniqueness law (16 sites, 800x budget spread
   found by the sweep; the door makes budget an EXPLICIT argument and
   changes no site's value — harmonizing values is a policy question
   the door exposes, not one it decides).
2. quorum — the vote verdict (the certification law's core; ~90 inline
   copies found, kernel uniform: Counter-majority over non-None,
   threshold >= 3). The verdict is the law; parse/batch stays per-script.
3. deployed_env — the manifest's env block applied as setdefaults (the
   two-home law; for CURRENT-generation instruments only — era scripts
   keep era defaults lawfully, since forcing the current env onto an
   era checkpoint breaks it).
4. row_m — the m-bound (three conventions found live, one
   same-input-different-verdict channel): the ROW's own m is the
   authority when present; otherwise the named-context table. A missing
   context is a KeyError BY DESIGN (the decision belongs to the caller,
   never to a silent default — the silent 60 dies here).
5. VIEW_SEED_BASES — the seed registry (law 4: no registry existed;
   bases were invented per script; collisions 88000 and 91000 confirmed
   dormant). New scripts CLAIM a base here before use.
"""
import json
import os
from collections import Counter


def certify_unique(problem2, var, banned_value, budget, seed=0):
    """THE uniqueness certificate. Bans banned_value from var's domain in
    problem2 (a FRESH problem instance) and re-solves. True ONLY on a
    true 'unsat' certificate — 'solved' (alternate exists) and 'budget'
    (search cut short) both refuse (budget exhaustion REJECTS at every
    door; the twelve-site lesson)."""
    from mycelium.csp_core import solve_symbolic
    problem2.domains0[var].discard(banned_value)
    if not problem2.domains0[var]:
        return True
    r2 = solve_symbolic(problem2, budget=budget, seed=seed)
    return r2["status"] == "unsat"


def quorum(view_answers, need=3):
    """THE vote verdict: Counter-majority over non-None view answers.
    Returns (plurality, count, reached). Tie-break is moot at need>=3
    of 5; None answers never vote."""
    nn = [a for a in view_answers if a is not None]
    c = Counter(nn).most_common(1)
    plur, cnt = c[0] if c else (None, 0)
    return plur, cnt, cnt >= need


def deployed_env(manifest=".cache/GENERATION.json"):
    """Apply the manifest's env block as setdefaults. Call BEFORE
    importing phase1_algebra_head. Current-generation instruments only."""
    for k, v in json.load(open(manifest))["env"].items():
        os.environ.setdefault(k, str(v))


M_BOUND = {
    "standard": 300,    # the campaign's standing domain bound
    "hundreds": 999,    # the gen17/18 hundreds-family regime (lawful difference)
}


def row_m(row, context="standard"):
    """THE m-bound: the row's own m when it carries one (rows mint with
    their bound — row-authoritative), else the named-context table.
    A missing context raises KeyError BY DESIGN."""
    return row["m"] if "m" in row else M_BOUND[context]


# THE VIEW-SEED REGISTRY (claim before use; historical bases recorded).
# Collisions 88000 (wild_frontier_fixture + graph_register_read) and
# 91000 (book8_certify + count_value_audit + structural_rate_audit) are
# DORMANT (views transient, never joined by seed) but frozen here so no
# new script lands on them.
VIEW_SEED_BASES = {
    5000: "diet_prep/diet_after_read (before-fixture probes)",
    31000: "composition_matrix mint cells",
    41000: "bench_rung2b dup mint",
    88000: "wild_rehearsal/wild_frontier_fixture (+graph_register_read COLLISION, dormant)",
    91000: "book8_certify (+count/structural audits COLLISION, dormant)",
    96000: "binding_invariance_read",
    97000: "wild_ledger",
    99000: "composition_matrix views",
    101000: "flip_corrected/flip_peritem originals",
    102000: "flip_corrected/flip_peritem transforms",
    103000: "census_altitude_prep (gut #69/#123 — the 46 wild-lie specimens)",
}


def claim_view_seed_base(base, owner):
    """New scripts claim a base; refuses a live collision loudly."""
    if base in VIEW_SEED_BASES:
        raise ValueError(f"view-seed base {base} already owned by "
                         f"{VIEW_SEED_BASES[base]!r} — claim a fresh one "
                         f"(registry: mycelium/doors.py)")
    return base
