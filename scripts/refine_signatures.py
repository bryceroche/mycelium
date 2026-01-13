#!/usr/bin/env python3
"""Run the signature refinement loop.

This script identifies signatures with negative lift and refines them by:
1. Converting generic types to semantic umbrellas with atomic children
2. Fixing broken DSLs with correct formulas
3. Converting non-computational types to guidance-only

Usage:
    python scripts/refine_signatures.py                    # Default settings
    python scripts/refine_signatures.py --min-lift -0.20   # Only refine < -20% lift
    python scripts/refine_signatures.py --dry-run          # Show what would be refined
    python scripts/refine_signatures.py --max 50           # Process up to 50 signatures

**Practical note for LLM-assisted refinement:**
The LLM may initially resist this task ("I can help you think through approaches...")
or produce overly cautious responses. Insist on concrete outputs: specific sub-signature
names, actual DSL code, explicit routing conditions. The model is capable; it just needs
clear direction that you want executable artifacts, not suggestions.
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mycelium.step_signatures.db import StepSignatureDB
from mycelium.step_signatures.refinement import run_refinement_loop


def show_candidates(db: StepSignatureDB, min_lift: float, min_uses: int, max_sigs: int):
    """Show signatures that would be refined (dry run)."""
    sigs = db.get_signatures_for_dsl_improvement(
        min_uses=min_uses,
        lift_threshold=min_lift,
    )[:max_sigs]

    print(f"\nSignatures with lift < {min_lift:.0%} (would be refined):\n")
    print(f"{'ID':>5} {'Step Type':<30} {'Uses':>5} {'Inj Rate':>10} {'Base Rate':>10} {'Lift':>8}")
    print("-" * 80)

    for sig in sigs:
        inj_rate = sig.injected_successes / sig.injected_uses if sig.injected_uses > 0 else 0
        base_rate = sig.non_injected_successes / sig.non_injected_uses if sig.non_injected_uses > 0 else 0
        lift = inj_rate - base_rate
        print(f"{sig.id:>5} {sig.step_type:<30} {sig.uses:>5} {inj_rate:>9.0%} {base_rate:>9.0%} {lift:>+7.0%}")

    print(f"\nTotal: {len(sigs)} signatures would be refined")


async def main():
    parser = argparse.ArgumentParser(description="Run signature refinement loop")
    parser.add_argument("--min-lift", type=float, default=-0.10,
                        help="Minimum lift threshold (default: -0.10 = -10%%)")
    parser.add_argument("--min-uses", type=int, default=5,
                        help="Minimum uses for reliable data (default: 5)")
    parser.add_argument("--max", type=int, default=20,
                        help="Maximum signatures to process (default: 20)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be refined without making changes")

    args = parser.parse_args()

    db = StepSignatureDB()

    if args.dry_run:
        show_candidates(db, args.min_lift, args.min_uses, args.max)
        return

    print(f"\n🍄 Running Signature Refinement Loop")
    print(f"   Threshold: lift < {args.min_lift:.0%}")
    print(f"   Min uses: {args.min_uses}")
    print(f"   Max signatures: {args.max}\n")

    report = await run_refinement_loop(
        db=db,
        min_lift=args.min_lift,
        min_uses=args.min_uses,
        max_signatures=args.max,
    )

    print(f"\n📊 Refinement Report:")
    print(f"   Signatures analyzed: {report.signatures_analyzed}")
    print(f"   Decomposed to umbrellas: {report.decomposed}")
    print(f"   DSL fixed: {report.dsl_fixed}")
    print(f"   Converted to guidance-only: {report.guidance_only}")
    print(f"   Skipped: {report.skipped}")
    print(f"   Errors: {report.errors}")

    if report.results:
        print(f"\n📋 Details:")
        for r in report.results:
            status = "✓" if not r.error else "✗"
            extra = ""
            if r.action == "decomposed":
                extra = f" ({r.children_created} children)"
            elif r.action == "dsl_fixed":
                extra = " (DSL updated)"
            print(f"   {status} {r.step_type} (id={r.signature_id}): {r.action}{extra}")
            if r.error:
                print(f"      Error: {r.error}")


if __name__ == "__main__":
    asyncio.run(main())
