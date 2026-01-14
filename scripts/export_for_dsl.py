#!/usr/bin/env python3
"""Export signatures needing DSLs for Claude to generate."""

import json
import sqlite3
import sys
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "mycelium.db"

def get_signatures_needing_dsl(limit: int = 50):
    """Get signatures that need DSLs, prioritized by usage."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    # Get signatures without DSL, ordered by uses (highest value first)
    cursor = conn.execute("""
        SELECT id, signature_id, step_type, description, uses, successes,
               method_name, method_template
        FROM step_signatures
        WHERE (dsl_script IS NULL OR dsl_script = '')
          AND uses >= 1
        ORDER BY uses DESC
        LIMIT ?
    """, (limit,))

    signatures = []
    for row in cursor:
        sig_id = row['id']

        # Get examples for this signature
        examples_cursor = conn.execute("""
            SELECT step_text, result, success
            FROM step_examples
            WHERE signature_id = ?
              AND success = 1
            LIMIT 5
        """, (sig_id,))

        examples = [dict(ex) for ex in examples_cursor]

        signatures.append({
            "id": sig_id,
            "signature_id": row['signature_id'],
            "step_type": row['step_type'],
            "description": row['description'],
            "uses": row['uses'],
            "successes": row['successes'],
            "method_name": row['method_name'],
            "examples": examples,
        })

    conn.close()
    return signatures

def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    sigs = get_signatures_needing_dsl(limit)

    print(f"# {len(sigs)} signatures needing DSLs\n")

    for sig in sigs:
        print(f"## Signature ID: {sig['id']}")
        print(f"Type: {sig['step_type']}")
        print(f"Description: {sig['description']}")
        print(f"Uses: {sig['uses']}, Successes: {sig['successes']}")
        print(f"Method: {sig['method_name']}")
        print("\nExamples:")
        for ex in sig['examples']:
            print(f"  - Step: {ex['step_text'][:100]}")
            print(f"    Result: {ex['result'][:100] if ex['result'] else 'None'}")
        print("\n---\n")

if __name__ == "__main__":
    main()
