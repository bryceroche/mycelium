"""era.py — THE ERA MANIFEST (2026-08-09; the doors pattern's third
domain after custody gold and detector authority; three specimens
bought it: the opatt val-fixture crash, the rebuild's 3-vs-7 gold
width, the aim sliver's missing decisions field).

ONE authority for the deployed lineage's environment era. Consumers
DERIVE — no fire, rebuild, or eval re-decides these. The sentinels
verify coherence; THIS verifies era membership (the era-check
distinction, 2026-08-08).

Usage:
    from mycelium.era import DEPLOYED_ENVS, apply_era
    apply_era()          # setdefault all deployed envs
    # or consume DEPLOYED_ENVS directly in fire scripts
Adoption at next touch per organize-machinery — no big-bang rewrite."""
import os, json

def _manifest_envs():
    try:
        man = json.load(open(".cache/GENERATION.json"))
        if isinstance(man.get("envs"), dict):
            return {str(k): str(v) for k, v in man["envs"].items()}
    except Exception:
        pass
    return None

# the gen-23 era (the deployed lineage), used when the manifest
# carries no explicit envs block:
_FALLBACK = {"ALG2": "1", "ALG_FTYPES": "8", "ALG_HW": "512",
             "ALG_DUP": "1", "ALG_WIDE": "1"}

DEPLOYED_ENVS = _manifest_envs() or dict(_FALLBACK)

# the mint-row schema fields every hand-minted row must carry (the
# aim sliver's specimen):
MINT_ROW_REQUIRED = ("text", "factors", "query_var", "n_vars", "m",
                     "mentions", "solution", "decisions")

def apply_era():
    for k, v in DEPLOYED_ENVS.items():
        os.environ.setdefault(k, v)
    return dict(DEPLOYED_ENVS)

def assert_mint_row(row, site=""):
    missing = [k for k in MINT_ROW_REQUIRED if k not in row]
    if missing:
        raise KeyError(f"mint row missing {missing} at {site} — the era "
                       f"manifest's schema contract (aim-sliver specimen)")
