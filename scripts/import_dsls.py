#!/usr/bin/env python3
"""Import Claude-generated DSLs into the database."""

import json
import sqlite3
import sys
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "mycelium.db"

def import_dsls(json_file: str):
    """Import DSLs from JSON file into database."""
    with open(json_file) as f:
        dsls = json.load(f)

    conn = sqlite3.connect(DB_PATH)
    updated = 0
    skipped = 0

    for sig_id, dsl in dsls.items():
        dsl_json = json.dumps(dsl)
        try:
            cursor = conn.execute(
                "UPDATE step_signatures SET dsl_script = ? WHERE id = ?",
                (dsl_json, int(sig_id))
            )
            if cursor.rowcount > 0:
                updated += 1
            else:
                skipped += 1
        except Exception as e:
            print(f"Error updating {sig_id}: {e}")
            skipped += 1

    conn.commit()
    conn.close()

    print(f"Updated {updated} signatures, skipped {skipped}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: import_dsls.py <json_file>")
        sys.exit(1)
    import_dsls(sys.argv[1])
