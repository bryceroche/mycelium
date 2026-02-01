"""Database operations for templates and examples."""
import sqlite3
import json
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
from .models import Template, Example, ExampleProposal

DB_PATH = Path.home() / ".mycelium" / "templates.db"


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize the database schema."""
    conn = get_connection()
    conn.executescript('''
        CREATE TABLE IF NOT EXISTS templates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            description TEXT,
            guidance TEXT,
            prompt_template TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS examples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            problem_text TEXT NOT NULL,
            embedding BLOB,
            template_id INTEGER NOT NULL,
            slots_mapped TEXT,  -- JSON dict
            similarity_to_nearest REAL DEFAULT 0.0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (template_id) REFERENCES templates(id)
        );

        CREATE TABLE IF NOT EXISTS example_proposals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            problem_text TEXT NOT NULL,
            embedding BLOB,
            template_id INTEGER NOT NULL,
            similarity_to_nearest REAL DEFAULT 0.0,
            slots_mapped TEXT,
            computed_answer TEXT,
            expected_answer TEXT,
            status TEXT DEFAULT 'pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (template_id) REFERENCES templates(id)
        );

        CREATE INDEX IF NOT EXISTS idx_examples_template ON examples(template_id);
        CREATE INDEX IF NOT EXISTS idx_proposals_status ON example_proposals(status);
    ''')
    conn.commit()
    conn.close()


def save_template(template: Template) -> int:
    """Save a template, return its ID."""
    conn = get_connection()
    cursor = conn.cursor()

    if template.id:
        cursor.execute('''
            UPDATE templates SET name=?, description=?, guidance=?, prompt_template=?
            WHERE id=?
        ''', (template.name, template.description, template.guidance,
              template.prompt_template, template.id))
    else:
        cursor.execute('''
            INSERT OR REPLACE INTO templates (name, description, guidance, prompt_template)
            VALUES (?, ?, ?, ?)
        ''', (template.name, template.description, template.guidance,
              template.prompt_template))
        template.id = cursor.lastrowid

    conn.commit()
    conn.close()
    return template.id


def get_template(template_id: int) -> Optional[Template]:
    """Get a template by ID."""
    conn = get_connection()
    row = conn.execute('SELECT * FROM templates WHERE id=?', (template_id,)).fetchone()
    conn.close()

    if not row:
        return None

    return Template(
        id=row['id'],
        name=row['name'],
        description=row['description'],
        guidance=row['guidance'] or "",
        prompt_template=row['prompt_template'] or "",
        created_at=row['created_at']
    )


def get_template_by_name(name: str) -> Optional[Template]:
    """Get a template by name."""
    conn = get_connection()
    row = conn.execute('SELECT * FROM templates WHERE name=?', (name,)).fetchone()
    conn.close()

    if not row:
        return None

    return Template(
        id=row['id'],
        name=row['name'],
        description=row['description'],
        guidance=row['guidance'] or "",
        prompt_template=row['prompt_template'] or "",
        created_at=row['created_at']
    )


def get_all_templates() -> List[Template]:
    """Get all templates."""
    conn = get_connection()
    rows = conn.execute('SELECT * FROM templates').fetchall()
    conn.close()

    return [Template(
        id=row['id'],
        name=row['name'],
        description=row['description'],
        guidance=row['guidance'] or "",
        prompt_template=row['prompt_template'] or "",
        created_at=row['created_at']
    ) for row in rows]


def save_example(example: Example) -> int:
    """Save an example, return its ID."""
    conn = get_connection()
    cursor = conn.cursor()

    embedding_blob = example.embedding.tobytes() if example.embedding is not None else None

    cursor.execute('''
        INSERT INTO examples (problem_text, embedding, template_id, slots_mapped, similarity_to_nearest)
        VALUES (?, ?, ?, ?, ?)
    ''', (example.problem_text, embedding_blob, example.template_id,
          json.dumps(example.slots_mapped), example.similarity_to_nearest))

    example.id = cursor.lastrowid
    conn.commit()
    conn.close()
    return example.id


def get_all_examples() -> List[Example]:
    """Get all examples with embeddings."""
    conn = get_connection()
    rows = conn.execute('SELECT * FROM examples').fetchall()
    conn.close()

    examples = []
    for row in rows:
        embedding = np.frombuffer(row['embedding'], dtype=np.float32) if row['embedding'] else None
        examples.append(Example(
            id=row['id'],
            problem_text=row['problem_text'],
            embedding=embedding,
            template_id=row['template_id'],
            slots_mapped=json.loads(row['slots_mapped']) if row['slots_mapped'] else {},
            similarity_to_nearest=row['similarity_to_nearest'],
            created_at=row['created_at']
        ))
    return examples


def find_nearest_example(embedding: np.ndarray, top_k: int = 1) -> List[Tuple[Example, float]]:
    """Find nearest examples by cosine similarity."""
    examples = get_all_examples()

    if not examples:
        return []

    # Normalize query
    embedding = embedding / (np.linalg.norm(embedding) + 1e-9)

    results = []
    for ex in examples:
        if ex.embedding is not None:
            ex_norm = ex.embedding / (np.linalg.norm(ex.embedding) + 1e-9)
            similarity = float(np.dot(embedding, ex_norm))
            results.append((ex, similarity))

    results.sort(key=lambda x: -x[1])
    return results[:top_k]


def save_proposal(proposal: ExampleProposal) -> int:
    """Save an example proposal."""
    conn = get_connection()
    cursor = conn.cursor()

    embedding_blob = proposal.embedding.tobytes() if proposal.embedding is not None else None

    cursor.execute('''
        INSERT INTO example_proposals
        (problem_text, embedding, template_id, similarity_to_nearest, slots_mapped, computed_answer, expected_answer, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (proposal.problem_text, embedding_blob, proposal.template_id,
          proposal.similarity_to_nearest, json.dumps(proposal.slots_mapped),
          json.dumps(proposal.computed_answer), json.dumps(proposal.expected_answer),
          proposal.status))

    proposal.id = cursor.lastrowid
    conn.commit()
    conn.close()
    return proposal.id


def get_pending_proposals() -> List[ExampleProposal]:
    """Get all pending proposals."""
    conn = get_connection()
    rows = conn.execute("SELECT * FROM example_proposals WHERE status='pending'").fetchall()
    conn.close()

    return [ExampleProposal(
        id=row['id'],
        problem_text=row['problem_text'],
        template_id=row['template_id'],
        similarity_to_nearest=row['similarity_to_nearest'],
        slots_mapped=json.loads(row['slots_mapped']) if row['slots_mapped'] else {},
        computed_answer=json.loads(row['computed_answer']) if row['computed_answer'] else None,
        expected_answer=json.loads(row['expected_answer']) if row['expected_answer'] else None,
        status=row['status'],
        created_at=row['created_at']
    ) for row in rows]


def approve_proposal(proposal_id: int):
    """Approve a proposal and create an example from it."""
    conn = get_connection()
    row = conn.execute('SELECT * FROM example_proposals WHERE id=?', (proposal_id,)).fetchone()

    if row:
        # Create example
        conn.execute('''
            INSERT INTO examples (problem_text, embedding, template_id, slots_mapped, similarity_to_nearest)
            VALUES (?, ?, ?, ?, ?)
        ''', (row['problem_text'], row['embedding'], row['template_id'],
              row['slots_mapped'], row['similarity_to_nearest']))

        # Update proposal status
        conn.execute("UPDATE example_proposals SET status='approved' WHERE id=?", (proposal_id,))
        conn.commit()

    conn.close()


def reject_proposal(proposal_id: int):
    """Reject a proposal."""
    conn = get_connection()
    conn.execute("UPDATE example_proposals SET status='rejected' WHERE id=?", (proposal_id,))
    conn.commit()
    conn.close()


# Initialize on import
init_db()
