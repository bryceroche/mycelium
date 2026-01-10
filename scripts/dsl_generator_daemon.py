#!/usr/bin/env python3
"""Background DSL generator for new signatures.

This script runs continuously, checking for signatures without DSL
and generating DSL scripts for them using an LLM.

Usage:
    python scripts/dsl_generator_daemon.py [--interval 30] [--batch-size 5]

Options:
    --interval: Seconds between checks (default 30)
    --batch-size: Max signatures to process per batch (default 5)
    --once: Run once and exit (don't loop)
    --min-uses: Minimum uses before generating DSL (default 1, set to 0 for immediate)
"""

import argparse
import asyncio
import logging
import sqlite3
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mycelium.step_signatures import StepSignatureDB
from mycelium.step_signatures.dsl_generator import generate_dsl_for_signature
from mycelium.client import GroqClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_signatures_needing_dsl(db_path: str = "mycelium.db", limit: int = 5, min_uses: int = 1) -> list[dict]:
    """Find signatures that could benefit from DSL but don't have it yet.

    Criteria:
    - No DSL script yet
    - Not general_step (those can't have useful DSL)
    - Has at least min_uses (default 1)
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    cursor = conn.execute("""
        SELECT id, signature_id, step_type, description, method_template, uses,
               CAST(successes AS FLOAT) / NULLIF(uses, 0) as success_rate
        FROM step_signatures
        WHERE (dsl_script IS NULL OR dsl_script = '')
          AND step_type != 'general_step'
          AND uses >= ?
        ORDER BY uses DESC
        LIMIT ?
    """, (min_uses, limit))

    results = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return results


def get_signature_examples(db_path: str, signature_id: int, limit: int = 5) -> list[dict]:
    """Get examples for a signature."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    cursor = conn.execute("""
        SELECT step_text, result, success
        FROM step_examples
        WHERE signature_id = ?
          AND success = 1
        ORDER BY created_at DESC
        LIMIT ?
    """, (signature_id, limit))

    results = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return results


def update_signature_dsl(db_path: str, signature_id: int, dsl_script: str):
    """Update a signature with generated DSL."""
    conn = sqlite3.connect(db_path)
    conn.execute(
        "UPDATE step_signatures SET dsl_script = ? WHERE id = ?",
        (dsl_script, signature_id)
    )
    conn.commit()
    conn.close()
    logger.info(f"Saved DSL to signature {signature_id}")


async def generate_dsl_batch(
    signatures: list[dict],
    client: GroqClient,
    db_path: str = "mycelium.db"
) -> int:
    """Generate DSL for a batch of signatures.

    Returns number of successful generations.
    """
    generated = 0

    for sig in signatures:
        try:
            step_type = sig['step_type']
            description = sig.get('description') or sig.get('method_template') or step_type

            logger.info(f"Generating DSL for {step_type} (id={sig['id']}, uses={sig['uses']})")

            # Get examples for context
            examples = get_signature_examples(db_path, sig['id'], limit=5)

            if not examples:
                # No examples yet - create a synthetic one from description
                logger.info(f"  No examples yet, using description as context")
                examples = [{"step_text": description, "result": "unknown", "success": True}]

            # Generate DSL
            dsl_script = await generate_dsl_for_signature(
                client=client,
                step_type=step_type,
                description=description,
                examples=examples,
            )

            if dsl_script:
                update_signature_dsl(db_path, sig['id'], dsl_script)
                logger.info(f"  Generated DSL: {dsl_script[:100]}...")
                generated += 1
            else:
                logger.info(f"  No DSL generated (LLM determined not suitable)")

        except Exception as e:
            logger.error(f"  Error generating DSL for {sig['id']}: {e}")

    return generated


async def run_daemon(
    interval: int = 30,
    batch_size: int = 5,
    once: bool = False,
    db_path: str = "mycelium.db",
    min_uses: int = 1
):
    """Main daemon loop."""
    logger.info(f"DSL Generator started (interval={interval}s, batch_size={batch_size}, min_uses={min_uses})")

    client = GroqClient()
    total_generated = 0
    iterations = 0

    while True:
        iterations += 1

        # Find signatures needing DSL
        signatures = get_signatures_needing_dsl(db_path, limit=batch_size, min_uses=min_uses)

        if signatures:
            logger.info(f"[Iter {iterations}] Found {len(signatures)} signatures needing DSL")
            generated = await generate_dsl_batch(signatures, client, db_path)
            total_generated += generated
            logger.info(f"[Iter {iterations}] Generated {generated} DSL scripts (total: {total_generated})")
        else:
            logger.info(f"[Iter {iterations}] No signatures need DSL generation")

        if once:
            logger.info(f"Single run complete. Generated {total_generated} DSL scripts.")
            break

        logger.info(f"Sleeping {interval}s until next check...")
        await asyncio.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description="Background DSL generator for signatures")
    parser.add_argument("--interval", type=int, default=30, help="Seconds between checks")
    parser.add_argument("--batch-size", type=int, default=5, help="Max signatures per batch")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    parser.add_argument("--db", default="mycelium.db", help="Database path")
    parser.add_argument("--min-uses", type=int, default=1, help="Min uses before generating DSL (0=immediate)")
    args = parser.parse_args()

    asyncio.run(run_daemon(
        interval=args.interval,
        batch_size=args.batch_size,
        once=args.once,
        db_path=args.db,
        min_uses=args.min_uses,
    ))


if __name__ == "__main__":
    main()
