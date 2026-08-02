"""diagnostic_register.py — THE GOODHART FENCE (2026-08-02, registered
under the standing word; the commitment family's general form).

THE LAW: any signal whose informativeness depends on the model NOT
optimizing for it must be structurally excluded from the training path.
Supervising a diagnostic does not cure the pathology it reads — it
teaches CONCEALMENT of the diagnostic's referent, which is strictly
worse than losing the instrument: the campaign would believe the
pathology cured while it went underground, and the meter would keep
printing clean. (The self-loop specimen: a gate trained to emit fewer
self-loops produces cleaner-looking graphs at identical binding
accuracy, and the one gold-free difficulty gauge the campaign owns
goes dark — not a lost instrument, a LYING one.)

THE BOUNDARY (so the fence doesn't overreach): training on OUTCOMES a
diagnostic predicts is lawful; training on the diagnostic is not.
Curing misbinding is lawful; adding a settle term to a loss is not.
The distinction is whether the signal is the TARGET or the METER.

COUPLING TIERS:
  - "never": may not appear in any loss term, reward, or objective,
    and may not drive data selection. Hard fence.
  - "loss-never": may not appear in any loss/reward/objective term.
    Declared data-selection use is lawful where a standing law
    sanctions it (no gradient flows through the read; the coupling is
    curricular, not optimizational) — but every such use is DECLARED
    at the call site with a comment naming this register.

THE AUDIT QUESTION (greppable, joins the deep-clean checklist): does
any loss term, reward, or selection criterion read a quantity named in
DIAGNOSTICS? Training scripts that build a loss from named signals
call assert_not_supervised() with those names — the same assert-at-
the-door pattern as mycelium/doors.py.
"""

DIAGNOSTICS = {
    "self_loop_count": {
        "coupling": "never",
        "why": (
            "Unintended emission — a tell the gate doesn't know it's "
            "giving; the load signature and the only gold-free "
            "difficulty gauge. Value exists BECAUSE it is unasked-for; "
            "any pressure against it erases the tell, not the failure. "
            "Fenced 'instrument, never objective' the day it was found."
        ),
    },
    "determination_breath0": {
        "coupling": "never",
        "why": (
            "The determination signal at breath 0 — incidental "
            "structure read off the deducer's first step; fenced "
            "identically at its docket entry."
        ),
    },
    "settle": {
        "coupling": "never",
        "why": (
            "READINESS IS SETTLING — post-evidence-quantum stability "
            "reads basin RESIDENCY, not correctness. A settle loss "
            "term would buy fast confident residency in wrong basins "
            "and blind the campaign's best mechanism-grade signal."
        ),
    },
    "vote_entropy": {
        "coupling": "loss-never",
        "why": (
            "TEMPERATURE-PERPENDICULAR-TO-TRUTH: entropy reads basin "
            "depth, never correctness. Sanctioned data-selection use: "
            "correct-but-shallow items as rehearsal targets (the "
            "standing law; curricular coupling, no gradient through "
            "the read). Never a loss/reward term."
        ),
    },
    "mouth_distance": {
        "coupling": "loss-never",
        "why": (
            "The register read. Optimizing the read directly trains "
            "the system to LOOK native — band camouflage; admission "
            "would admit exactly what it should refuse. Books wetting "
            "the register through real training data is the lawful "
            "outcome-path (the distribution moves; the meter is never "
            "the target). Mouth-side mining/recalibration reads are "
            "instrument maintenance, not supervision."
        ),
    },
    # --- registered 2026-08-02, the completeness pass (Bryce named all
    # five; whys quote their ledger definition entries) ---
    "engagement_point": {
        "coupling": "never",
        "why": (
            "The parse-side engagement read (the pawl's click): the "
            "last prefix index at which the arg-pointer argmax "
            "changes. Its value is that early engagement is a TELL — "
            "the pointer settling on a distractor before the operand's "
            "token arrives. A loss pushing engagement later would "
            "produce deliberate-LOOKING pointers at identical binding "
            "accuracy; distractor-load's live pre-certification signal "
            "goes dark. (Rung 2's reversible-commitment work changes "
            "the ARCHITECTURE the meter reads — lawful; the meter as "
            "target — never.)"
        ),
    },
    "ramp_preservation": {
        "coupling": "never",
        "why": (
            "Layer-separation, live — a KILL criterion: polarization "
            "is a spendable pretrained asset, and a loop that eats "
            "its own ramp should be killed for it. As a loss term it "
            "would buy surface layer-separation with no guarantee the "
            "anatomy underneath still functions — a kill-switch that "
            "can no longer fire is worse than none."
        ),
    },
    "thermometer": {
        "coupling": "never",
        "why": (
            "Effective-alpha, per-breath state-delta magnitude — the "
            "telemetry law's temperature channel; failure arrives "
            "with its temperature attached. Supervising toward a "
            "target update magnitude masks instability instead of "
            "curing it, and the fire's black box goes silent exactly "
            "when it's needed."
        ),
    },
    "flip_rate": {
        "coupling": "loss-never",
        "why": (
            "The band-width meter (binding invariance across phrasing "
            "transforms). A paraphrase-consistency loss would optimize "
            "the read directly — views agreeing because trained-to-"
            "agree, which poisons BOTH the flip instrument and the "
            "5-view vote whose quorum assumes views are independent "
            "evidence. Lawful outcome-path: augmentation diets through "
            "real data (the AUG fire); lawful selection: choosing "
            "which families need diet from measured flips (declared)."
        ),
    },
    "distance_screen": {
        "coupling": "never",
        "why": (
            "First-use distance past the passing p90 — chartered as "
            "'flagged on the sheet for scrutiny, NEVER judged by the "
            "flag' (the dead mechanism's lawful ghost). Training or "
            "selecting against it makes first-use distances look "
            "small — camouflage — and quietly promotes the flag from "
            "scrutiny-router to judge, which its own charter forbids."
        ),
    },
}


def is_diagnostic(name):
    return name in DIAGNOSTICS


def assert_not_supervised(signal_names, allow_selection=False):
    """Call from any training script that assembles a loss/reward/
    selection from named signals. Raises on any register hit; a
    'loss-never' entry passes ONLY when allow_selection=True (the
    caller thereby declares a sanctioned data-selection use — the
    declaration is the point)."""
    for name in signal_names:
        entry = DIAGNOSTICS.get(name)
        if entry is None:
            continue
        if entry["coupling"] == "never" or not allow_selection:
            raise AssertionError(
                f"GOODHART FENCE: '{name}' is a diagnostic "
                f"({entry['coupling']}) and may not be supervised. "
                f"{entry['why']}"
            )
