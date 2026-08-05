"""detector_authority.py — THE MIS-ANCHOR DETECTOR AUTHORITY (deep-clean
sweep, 2026-08-05; gut #156's lead reading executed; the row_m pattern:
one authority per context, named explicitly, KeyError on unnamed).
Three instruments answered "is this slot mis-anchored"; three declared
jobs; no improvisation. Tonight's instrument-invalid (a re-derived
family-resemblance proxy, 1/338 flagged) is the invoice this file
retires: consumers LOOK UP, never re-derive."""

AUTHORITY = {
    # gold spans don't exist at inference — structural, not preference
    "measurement": {
        "detector": "loc anchor artifact (gold-span inspan)",
        "source": ".cache/loc_anchors_bigtest.jsonl via scripts/loc_anchor_read.py",
        "scope": "offline only; ground-truth-grade (0.911/0.063)"},
    "deployed": {
        "detector": "licensed trigger (relative proxy, thr 0.3648)",
        "source": "scripts/trigger_license_v2.py — READ IT, never re-derive",
        "scope": "in-register ONLY (wild: FP 68.9%, AUC inverted — unlicensed)"},
    "derived": {
        "detector": "inspan<0.5 proxy (the anchor's shadow)",
        "source": "derives FROM the measurement authority's artifact",
        "scope": "cheap re-reads where the anchor artifact is banked"},
}

def authority(context):
    if context not in AUTHORITY:
        raise KeyError(f"no mis-anchor authority for context '{context}' — "
                       f"declare it here first (two-home law); "
                       f"known: {sorted(AUTHORITY)}")
    return AUTHORITY[context]
