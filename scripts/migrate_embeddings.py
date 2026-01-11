#!/usr/bin/env python3
"""Migrate signatures from 384-dim to 768-dim embeddings.

This script:
1. Reads all signatures from the source DB (384-dim)
2. Re-embeds step text with the new model (768-dim)
3. Writes to the destination DB with updated embeddings

Usage:
    python scripts/migrate_embeddings.py
"""

import json
import os
import sqlite3
import sys
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mycelium.embedder import Embedder, DEFAULT_MODEL

SOURCE_DB = "mycelium.db"  # Old 384-dim DB
DEST_DB = "mycelium_768.db"  # New 768-dim DB


def migrate():
    print(f"Migration: {SOURCE_DB} -> {DEST_DB}")
    print(f"New embedding model: {DEFAULT_MODEL}")

    # Initialize embedder with new model
    embedder = Embedder.get_instance()
    print(f"Embedding dimension: {embedder.embedding_dim}")

    # Connect to source DB
    if not os.path.exists(SOURCE_DB):
        print(f"ERROR: Source DB not found: {SOURCE_DB}")
        return

    src_conn = sqlite3.connect(SOURCE_DB)
    src_cursor = src_conn.cursor()

    # Copy DB structure to destination
    if os.path.exists(DEST_DB):
        os.remove(DEST_DB)
        print(f"Removed existing {DEST_DB}")

    # Read source schema and create destination
    src_cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='step_signatures'")
    schema = src_cursor.fetchone()
    if not schema:
        print("ERROR: step_signatures table not found in source DB")
        return

    dst_conn = sqlite3.connect(DEST_DB)
    dst_cursor = dst_conn.cursor()

    # Enable WAL mode
    dst_cursor.execute("PRAGMA journal_mode = WAL")
    dst_cursor.execute("PRAGMA busy_timeout = 30000")

    # Create table with same schema
    dst_cursor.execute(schema[0])

    # Copy indexes
    src_cursor.execute("SELECT sql FROM sqlite_master WHERE type='index' AND tbl_name='step_signatures'")
    for row in src_cursor.fetchall():
        if row[0]:
            try:
                dst_cursor.execute(row[0])
            except sqlite3.OperationalError:
                pass  # Index might already exist

    dst_conn.commit()

    # Get all signatures
    src_cursor.execute("SELECT * FROM step_signatures")
    columns = [desc[0] for desc in src_cursor.description]
    rows = src_cursor.fetchall()

    print(f"Found {len(rows)} signatures to migrate")

    # Find column indexes
    try:
        centroid_idx = columns.index('centroid')
        id_idx = columns.index('id')
    except ValueError as e:
        print(f"ERROR: Missing column: {e}")
        return

    # Also copy step_examples table
    src_cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='step_examples'")
    examples_schema = src_cursor.fetchone()
    if examples_schema:
        dst_cursor.execute(examples_schema[0])
        dst_conn.commit()

    migrated = 0
    failed = 0

    for i, row in enumerate(rows):
        sig_id = row[id_idx]

        # Get examples for this signature
        src_cursor.execute(
            "SELECT step_text FROM step_examples WHERE signature_id = ?",
            (sig_id,)
        )
        examples = [r[0] for r in src_cursor.fetchall()]

        # If no examples, use step_type
        if not examples:
            step_type_idx = columns.index('step_type')
            examples = [row[step_type_idx] or "unknown"]

        # Embed examples and compute centroid
        try:
            embeddings = embedder.embed_batch(examples)
            centroid = embeddings.mean(axis=0)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-9)  # Normalize
            new_centroid = json.dumps(centroid.tolist())
        except Exception as e:
            print(f"ERROR embedding sig {sig_id}: {e}")
            failed += 1
            continue

        # Insert signature with new centroid
        row_list = list(row)
        row_list[centroid_idx] = new_centroid

        placeholders = ','.join(['?' for _ in columns])
        try:
            dst_cursor.execute(
                f"INSERT INTO step_signatures ({','.join(columns)}) VALUES ({placeholders})",
                row_list
            )
            migrated += 1
        except Exception as e:
            print(f"ERROR inserting sig {sig_id}: {e}")
            failed += 1

        if (i + 1) % 100 == 0:
            dst_conn.commit()
            print(f"  Migrated {i+1}/{len(rows)}...")

    dst_conn.commit()

    # Copy other tables as-is
    print("\nCopying auxiliary tables...")
    for table in ['step_examples', 'step_usage_log']:
        try:
            # Get schema
            src_cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table}'")
            schema = src_cursor.fetchone()
            if schema:
                try:
                    dst_cursor.execute(schema[0])
                except sqlite3.OperationalError:
                    pass  # Table exists

                # Copy data
                src_cursor.execute(f"SELECT * FROM {table}")
                rows = src_cursor.fetchall()
                if rows:
                    placeholders = ','.join(['?' for _ in range(len(rows[0]))])
                    dst_cursor.executemany(f"INSERT INTO {table} VALUES ({placeholders})", rows)
                    print(f"  Copied {len(rows)} rows to {table}")
                dst_conn.commit()
        except Exception as e:
            print(f"  Warning: Could not copy {table}: {e}")

    # Final stats
    dst_cursor.execute("SELECT COUNT(*) FROM step_signatures")
    final_count = dst_cursor.fetchone()[0]

    print(f"\n{'='*50}")
    print(f"Migration complete!")
    print(f"  Source: {len(rows)} signatures")
    print(f"  Migrated: {migrated}")
    print(f"  Failed: {failed}")
    print(f"  Destination: {final_count} signatures")
    print(f"  New embedding dim: {embedder.embedding_dim}")
    print(f"{'='*50}")

    src_conn.close()
    dst_conn.close()


if __name__ == "__main__":
    migrate()
