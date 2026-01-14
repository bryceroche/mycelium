#!/usr/bin/env python3
"""Migrate existing signatures to use updated step type classification.

This script:
1. Re-runs _infer_step_type() on all existing signatures
2. Updates step_type where classification changed
3. Assigns default DSL scripts for newly-typed signatures

Usage:
    python scripts/migrate_step_types.py [--dry-run]
"""

import argparse
import sqlite3
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mycelium.step_signatures.db import StepSignatureDB


def migrate_step_types(db_path: str = "mycelium.db", dry_run: bool = False):
    """Migrate signatures to updated step types."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Create a fresh DB instance to use _infer_step_type
    step_db = StepSignatureDB(db_path=db_path)

    # Get all signatures with their example text
    cursor = conn.execute("""
        SELECT s.id, s.step_type, s.dsl_script, e.step_text
        FROM step_signatures s
        LEFT JOIN step_examples e ON s.id = e.signature_id
        WHERE e.step_text IS NOT NULL
        GROUP BY s.id
    """)

    changes = []
    for row in cursor.fetchall():
        old_type = row["step_type"]
        step_text = row["step_text"]

        # Re-infer step type
        new_type = step_db._infer_step_type(step_text)

        if new_type != old_type:
            # Get default DSL for new type
            new_dsl = step_db._get_default_dsl_script(new_type)

            changes.append({
                "id": row["id"],
                "old_type": old_type,
                "new_type": new_type,
                "old_dsl": row["dsl_script"],
                "new_dsl": new_dsl,
                "step_text": step_text[:60],
            })

    # Report changes
    print(f"Found {len(changes)} signatures to update")
    print()

    by_new_type = {}
    for c in changes:
        by_new_type[c["new_type"]] = by_new_type.get(c["new_type"], 0) + 1

    print("Changes by new type:")
    for t, count in sorted(by_new_type.items(), key=lambda x: -x[1]):
        print(f"  {t:25s}: {count}")
    print()

    if dry_run:
        print("DRY RUN - no changes made")
        print()
        print("Sample changes:")
        for c in changes[:10]:
            print(f"  {c['old_type']:20s} -> {c['new_type']:20s}: {c['step_text']}...")
        return

    # Apply changes
    print("Applying changes...")
    for c in changes:
        conn.execute(
            "UPDATE step_signatures SET step_type = ?, dsl_script = ? WHERE id = ?",
            (c["new_type"], c["new_dsl"], c["id"]),
        )

    conn.commit()
    print(f"Updated {len(changes)} signatures")

    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate step types")
    parser.add_argument("--dry-run", action="store_true", help="Show changes without applying")
    parser.add_argument("--db", default="mycelium.db", help="Database path")
    args = parser.parse_args()

    migrate_step_types(db_path=args.db, dry_run=args.dry_run)
