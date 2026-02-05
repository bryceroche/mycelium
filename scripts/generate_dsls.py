#!/usr/bin/env python3
"""
Process atomic templates for the pipeline.

Templates are embedding clusters from atomic_split.py. They represent
patterns like "person has n items" but do NOT have pre-assigned operations.

Per CLAUDE.md: "AVOID Verb Classification Like The Plague"
- Templates are NOT classified by operation type (SET, ADD, SUB, etc.)
- Computation is learned from execution outcomes, not classified upfront
- Route by what operations DO (embedding similarity), not what they SOUND LIKE

Pipeline: vocab_reduce → atomic_split → cluster → fine-tune MiniLM → **package templates**

USAGE:
    python scripts/generate_dsls.py
    python scripts/generate_dsls.py --templates path/to/templates.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

OUTPUT_DIR = Path(__file__).parent.parent


def print_stats(templates):
    """Print template statistics (NO operation type categorization)."""
    print(f"\n{'='*60}")
    print("TEMPLATE STATISTICS")
    print(f"{'='*60}")

    total = len(templates)
    with_examples = sum(1 for t in templates if t.get('span_examples'))
    with_centroid = sum(1 for t in templates if t.get('centroid'))

    # Member count distribution
    member_counts = [t.get('member_count', 0) for t in templates]
    if member_counts:
        avg_members = sum(member_counts) / len(member_counts)
        max_members = max(member_counts)
        min_members = min(member_counts)
    else:
        avg_members = max_members = min_members = 0

    print(f"  Total templates: {total}")
    print(f"  With span examples: {with_examples}")
    print(f"  With centroid: {with_centroid}")
    print(f"  Member counts: min={min_members}, avg={avg_members:.1f}, max={max_members}")

    # Sample patterns
    print(f"\n  Sample patterns (top 10 by member count):")
    sorted_tpls = sorted(templates, key=lambda t: t.get('member_count', 0), reverse=True)
    for t in sorted_tpls[:10]:
        print(f"    [{t.get('member_count', 0):5d}] {t['pattern']}")


def main():
    parser = argparse.ArgumentParser(description="Package atomic templates")
    parser.add_argument("--templates", type=str, default=None,
                        help="Path to atomic_templates.json")
    args = parser.parse_args()

    print("=" * 60)
    print("Atomic Template Packaging")
    print("=" * 60)
    print("\nNote: Templates do NOT have pre-assigned operations.")
    print("Computation is learned from execution, not classified upfront.")

    # Load templates
    templates_path = Path(args.templates) if args.templates else OUTPUT_DIR / "atomic_templates.json"
    print(f"\nLoading templates from: {templates_path}")
    with open(templates_path) as f:
        templates = json.load(f)
    print(f"  Templates: {len(templates)}")

    # Stats (no operation categorization)
    print_stats(templates)

    # Save templates (unchanged - no DSL assignment)
    output_path = OUTPUT_DIR / "atomic_templates_packaged.json"
    with open(output_path, "w") as f:
        json.dump(templates, f, indent=2)
    print(f"\n  Saved to {output_path}")


if __name__ == "__main__":
    main()
