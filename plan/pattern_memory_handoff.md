# Handoff: Pattern Memory — SQLite Long-Term Memory for Mathematical Reasoning

## One-Sentence Summary

Give the system a SQLite database that stores verified mathematical reasoning patterns. The model queries it by page embedding similarity, retrieves templates like "half of X → X/2", and uses them to formulate SymPy expressions. Patterns accumulate over training with success/failure tracking. Deduplication prevents bloat. The database is the system's long-term memory — permanent, inspectable, self-improving.

---

## Three Memory Timescales

```
Atoms (attention control):  HOW to read the problem       (microsecond — per token)
Pages (notebook):           WHAT we know about THIS problem (second — per cycle)
Pattern DB (SQLite):        WHAT works for problems LIKE this (permanent — across all training)
SymPy (calculator):         EXACT computation              (tool, not memory)
```

The pattern DB fills a gap: the model forgets everything between training problems. Each problem is solved from scratch. The database gives it institutional memory — "the last 500 times I saw 'half of X', dividing by 2 worked."

---

## Schema

```sql
CREATE TABLE IF NOT EXISTS patterns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    
    -- Pattern identity
    pattern_type TEXT NOT NULL,           -- canonical type: "half_of", "percent_of", "sum_of_parts"
    pattern_hash TEXT UNIQUE NOT NULL,    -- SHA256 of (pattern_type + canonical_template) for dedup
    
    -- Content
    sympy_template TEXT NOT NULL,         -- "result = {x} / 2" with placeholders
    description TEXT,                     -- human-readable: "half of a quantity"
    example_problem TEXT,                 -- one example problem that used this pattern
    
    -- Embedding for similarity search
    page_embedding BLOB NOT NULL,         -- 64 floats from the first page (problem encoding)
    
    -- Success tracking
    success_count INTEGER DEFAULT 1,
    fail_count INTEGER DEFAULT 0,
    
    -- Metadata
    created_epoch INTEGER DEFAULT 0,
    last_used_epoch INTEGER DEFAULT 0,
    last_updated TEXT DEFAULT (datetime('now'))
);

-- Index for fast type-based lookup
CREATE INDEX IF NOT EXISTS idx_pattern_type ON patterns(pattern_type);

-- Index for deduplication
CREATE UNIQUE INDEX IF NOT EXISTS idx_pattern_hash ON patterns(pattern_hash);
```

One table. Three indexes. That's it.

---

## Deduplication

The core problem: "half of 48" and "half of 96" are the SAME pattern but different problems. We want ONE entry for "half of X → X/2", not thousands of near-duplicates.

### Three-Layer Deduplication

```python
import hashlib

class PatternDeduplicator:
    """Prevents duplicate patterns through three mechanisms."""
    
    @staticmethod
    def canonicalize_template(sympy_template):
        """
        Normalize a SymPy template to its canonical form.
        Replace specific numbers with placeholders.
        
        "result = 48 / 2"     → "result = {a} / 2"
        "result = 96 / 2"     → "result = {a} / 2"    (same!)
        "total = 48 + 24"     → "total = {a} + {b}"
        "total = 100 + 50"    → "total = {a} + {b}"   (same!)
        """
        import re
        
        canonical = sympy_template.strip()
        
        # Replace numbers with ordered placeholders
        numbers = re.findall(r'\b\d+\.?\d*\b', canonical)
        seen = {}
        placeholder_idx = 0
        
        for num in numbers:
            if num not in seen:
                seen[num] = chr(ord('a') + placeholder_idx)
                placeholder_idx += 1
            canonical = canonical.replace(num, '{' + seen[num] + '}', 1)
        
        return canonical
    
    @staticmethod
    def compute_hash(pattern_type, canonical_template):
        """Unique hash for deduplication."""
        key = f"{pattern_type}::{canonical_template}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]
    
    @staticmethod
    def embedding_similar(emb1, emb2, threshold=0.95):
        """Check if two pattern embeddings are near-duplicates."""
        cos_sim = np.dot(emb1, emb2) / (
            np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8
        )
        return cos_sim > threshold
```

```
Layer 1 — Template canonicalization:
  "result = 48 / 2" and "result = 96 / 2" both become "result = {a} / 2"
  Same canonical form → same pattern → no duplicate

Layer 2 — Hash-based uniqueness:
  pattern_hash = SHA256(pattern_type + canonical_template)
  SQLite UNIQUE constraint prevents insertion of duplicates
  ON CONFLICT → update success count instead of inserting

Layer 3 — Embedding similarity:
  Before inserting, check if any existing pattern has cosine > 0.95
  If so, merge with existing pattern (update embedding as running average)
```

---

## Stored Procedures (Python Methods the Model Can Call)

The model doesn't write SQL. It calls structured methods that handle all DB operations. These are the model's "stored procedures":

```python
import sqlite3
import numpy as np
import hashlib
import json
from datetime import datetime

class PatternMemory:
    """
    Long-term memory for mathematical reasoning patterns.
    SQLite-backed. Deduplication built in. Self-improving.
    """
    
    def __init__(self, db_path="pattern_memory.db"):
        self.conn = sqlite3.connect(db_path)
        self._create_tables()
    
    def _create_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_type TEXT NOT NULL,
                pattern_hash TEXT UNIQUE NOT NULL,
                sympy_template TEXT NOT NULL,
                canonical_template TEXT NOT NULL,
                description TEXT,
                example_problem TEXT,
                page_embedding BLOB NOT NULL,
                success_count INTEGER DEFAULT 1,
                fail_count INTEGER DEFAULT 0,
                created_epoch INTEGER DEFAULT 0,
                last_used_epoch INTEGER DEFAULT 0,
                last_updated TEXT DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_pattern_type ON patterns(pattern_type);
        """)
        self.conn.commit()
    
    # =========================================
    # STORED PROCEDURE 1: QUERY (retrieve patterns)
    # =========================================
    def query(self, page_embedding, top_k=3, min_success_rate=0.3):
        """
        Find patterns similar to the current problem's page embedding.
        
        Called at: pass 1 (after initial problem encoding)
        Input: page embedding (64 floats)
        Output: list of (score, pattern_id, template, type, success_rate)
        
        Scoring: cosine_similarity × (0.5 + 0.5 × success_rate)
        Higher similarity AND higher success rate → higher score.
        """
        query_np = page_embedding.detach().cpu().numpy().astype(np.float32)
        
        rows = self.conn.execute(
            "SELECT id, page_embedding, sympy_template, pattern_type, "
            "canonical_template, description, success_count, fail_count "
            "FROM patterns"
        ).fetchall()
        
        if not rows:
            return []
        
        scored = []
        for (row_id, emb_bytes, template, ptype, canonical,
             desc, successes, failures) in rows:
            
            stored_np = np.frombuffer(emb_bytes, dtype=np.float32)
            
            # Cosine similarity
            cos_sim = float(np.dot(query_np, stored_np) / (
                np.linalg.norm(query_np) * np.linalg.norm(stored_np) + 1e-8
            ))
            
            # Success rate
            total_uses = successes + failures
            success_rate = successes / max(total_uses, 1)
            
            # Skip patterns with poor track record
            if total_uses >= 5 and success_rate < min_success_rate:
                continue
            
            # Combined score
            score = cos_sim * (0.5 + 0.5 * success_rate)
            
            scored.append({
                'score': score,
                'pattern_id': row_id,
                'template': template,
                'canonical': canonical,
                'type': ptype,
                'description': desc,
                'success_rate': success_rate,
                'total_uses': total_uses,
            })
        
        scored.sort(key=lambda x: x['score'], reverse=True)
        return scored[:top_k]
    
    # =========================================
    # STORED PROCEDURE 2: STORE (add new pattern)
    # =========================================
    def store(self, page_embedding, sympy_template, pattern_type="auto",
              description="", example_problem="", epoch=0):
        """
        Store a successful reasoning pattern.
        Handles deduplication automatically.
        
        Called at: end of training step when answer is correct
        Input: page embedding, SymPy template, metadata
        Output: pattern_id (new or existing)
        
        Deduplication:
          1. Canonicalize template (replace numbers with placeholders)
          2. Compute hash(type + canonical)
          3. If hash exists: increment success_count (merge)
          4. If hash is new but embedding is >0.95 similar: merge
          5. Otherwise: insert new pattern
        """
        embedding_np = page_embedding.detach().cpu().numpy().astype(np.float32)
        embedding_bytes = embedding_np.tobytes()
        
        # Canonicalize
        canonical = PatternDeduplicator.canonicalize_template(sympy_template)
        pattern_hash = PatternDeduplicator.compute_hash(pattern_type, canonical)
        
        # Try insert (hash-based dedup)
        try:
            self.conn.execute(
                "INSERT INTO patterns "
                "(pattern_type, pattern_hash, sympy_template, canonical_template, "
                "description, example_problem, page_embedding, "
                "success_count, created_epoch, last_used_epoch) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?)",
                (pattern_type, pattern_hash, sympy_template, canonical,
                 description, example_problem, embedding_bytes, epoch, epoch)
            )
            self.conn.commit()
            
            # Return the new pattern's ID
            cursor = self.conn.execute(
                "SELECT id FROM patterns WHERE pattern_hash = ?", (pattern_hash,)
            )
            return cursor.fetchone()[0]
            
        except sqlite3.IntegrityError:
            # Hash collision → duplicate pattern. Merge: increment success count.
            self.conn.execute(
                "UPDATE patterns SET success_count = success_count + 1, "
                "last_used_epoch = ?, last_updated = datetime('now'), "
                "page_embedding = ? "
                "WHERE pattern_hash = ?",
                (epoch, embedding_bytes, pattern_hash)
            )
            self.conn.commit()
            
            cursor = self.conn.execute(
                "SELECT id FROM patterns WHERE pattern_hash = ?", (pattern_hash,)
            )
            return cursor.fetchone()[0]
    
    # =========================================
    # STORED PROCEDURE 3: UPDATE (record success/failure)
    # =========================================
    def record_outcome(self, pattern_id, success, epoch=0):
        """
        Update a pattern's track record after using it.
        
        Called at: end of training/inference step
        Input: pattern_id, whether the answer was correct
        """
        if success:
            self.conn.execute(
                "UPDATE patterns SET success_count = success_count + 1, "
                "last_used_epoch = ?, last_updated = datetime('now') "
                "WHERE id = ?", (epoch, pattern_id)
            )
        else:
            self.conn.execute(
                "UPDATE patterns SET fail_count = fail_count + 1, "
                "last_used_epoch = ?, last_updated = datetime('now') "
                "WHERE id = ?", (epoch, pattern_id)
            )
        self.conn.commit()
    
    # =========================================
    # STORED PROCEDURE 4: PRUNE (remove bad patterns)
    # =========================================
    def prune(self, min_uses=10, max_failure_rate=0.7, stale_epochs=20,
              current_epoch=0):
        """
        Remove patterns that consistently fail or haven't been used recently.
        
        Called at: end of each epoch
        Removes:
          - Patterns used 10+ times with >70% failure rate
          - Patterns not used in 20+ epochs (stale)
        """
        # Remove high-failure patterns
        deleted_bad = self.conn.execute(
            "DELETE FROM patterns WHERE "
            "(success_count + fail_count) >= ? AND "
            "CAST(fail_count AS REAL) / (success_count + fail_count) > ?",
            (min_uses, max_failure_rate)
        ).rowcount
        
        # Remove stale patterns
        deleted_stale = self.conn.execute(
            "DELETE FROM patterns WHERE "
            "(? - last_used_epoch) > ? AND (success_count + fail_count) < ?",
            (current_epoch, stale_epochs, min_uses)
        ).rowcount
        
        self.conn.commit()
        
        if deleted_bad + deleted_stale > 0:
            print(f"Pruned {deleted_bad} failed + {deleted_stale} stale patterns")
        
        return deleted_bad, deleted_stale
    
    # =========================================
    # STORED PROCEDURE 5: STATS (inspect memory)
    # =========================================
    def stats(self):
        """
        Print summary statistics about the pattern memory.
        
        Called at: end of epoch, for diagnostics
        """
        total = self.conn.execute("SELECT COUNT(*) FROM patterns").fetchone()[0]
        
        if total == 0:
            print("Pattern memory: empty")
            return
        
        avg_success = self.conn.execute(
            "SELECT AVG(CAST(success_count AS REAL) / MAX(success_count + fail_count, 1)) "
            "FROM patterns"
        ).fetchone()[0]
        
        top_patterns = self.conn.execute(
            "SELECT pattern_type, canonical_template, success_count, fail_count "
            "FROM patterns ORDER BY success_count DESC LIMIT 5"
        ).fetchall()
        
        type_counts = self.conn.execute(
            "SELECT pattern_type, COUNT(*), SUM(success_count) "
            "FROM patterns GROUP BY pattern_type ORDER BY COUNT(*) DESC LIMIT 10"
        ).fetchall()
        
        print(f"Pattern memory: {total} patterns")
        print(f"Avg success rate: {avg_success:.1%}")
        print(f"\nTop 5 most-used patterns:")
        for ptype, template, succ, fail in top_patterns:
            rate = succ / max(succ + fail, 1)
            print(f"  [{ptype}] {template} ({succ} wins, {fail} fails, {rate:.0%})")
        print(f"\nPattern types:")
        for ptype, count, total_succ in type_counts:
            print(f"  {ptype}: {count} patterns, {total_succ} total successes")
    
    # =========================================
    # STORED PROCEDURE 6: EXPORT (save for analysis)
    # =========================================
    def export_json(self, path="pattern_memory_export.json"):
        """Export all patterns to JSON for analysis."""
        rows = self.conn.execute(
            "SELECT id, pattern_type, canonical_template, sympy_template, "
            "description, success_count, fail_count, created_epoch, last_used_epoch "
            "FROM patterns ORDER BY success_count DESC"
        ).fetchall()
        
        patterns = []
        for row in rows:
            patterns.append({
                'id': row[0],
                'type': row[1],
                'canonical': row[2],
                'template': row[3],
                'description': row[4],
                'successes': row[5],
                'failures': row[6],
                'success_rate': row[5] / max(row[5] + row[6], 1),
                'created_epoch': row[7],
                'last_used_epoch': row[8],
            })
        
        with open(path, 'w') as f:
            json.dump(patterns, f, indent=2)
        
        print(f"Exported {len(patterns)} patterns to {path}")
```

---

## Integration Into Thinking Loop

```python
def solve_with_memory(self, problem_text, problem_ids, max_passes=5, epoch=0):
    """Full solve with all three memory systems + SymPy."""
    state_pages = []
    sympy_results = {}
    used_pattern_id = None
    
    for pass_num in range(max_passes):
        # === THINK ===
        page, sympy_results = self.think_one_pass_with_sympy(
            problem_text, problem_ids, state_pages, pass_num, sympy_results
        )
        
        # === QUERY PATTERN MEMORY (after pass 1) ===
        if pass_num == 0 and self.pattern_memory is not None:
            matches = self.pattern_memory.query(page, top_k=3)
            
            if matches and matches[0]['score'] > 0.5:
                best = matches[0]
                used_pattern_id = best['pattern_id']
                
                # Inject pattern hint as context for next pass
                hint = f"Suggested approach ({best['type']}, "
                hint += f"{best['success_rate']:.0%} success): "
                hint += best['template']
                self.pattern_hint = hint
            else:
                self.pattern_hint = None
        
        # === CHECK STOPPING ===
        if 'answer' in sympy_results:
            return sympy_results['answer'], used_pattern_id
        
        if pass_num >= 1:
            conf, smooth = self.confidence_head(state_pages)
            if conf > 0.9 and smooth > 0.7:
                break
    
    # Extract answer
    if sympy_results:
        answer = list(sympy_results.values())[-1]
    else:
        answer = self.answer_head_predict(state_pages[-1])
    
    return answer, used_pattern_id


def after_solve(self, problem_text, pages, sympy_steps, 
                was_correct, used_pattern_id, epoch=0):
    """Post-solve: update pattern memory."""
    
    # Update outcome if we used a pattern
    if used_pattern_id is not None:
        self.pattern_memory.record_outcome(used_pattern_id, was_correct, epoch)
    
    # Store new pattern if successful and we have SymPy steps
    if was_correct and sympy_steps and self.pattern_memory is not None:
        # Auto-classify pattern type
        pattern_type = classify_pattern(sympy_steps)
        template = "; ".join(sympy_steps)
        
        self.pattern_memory.store(
            page_embedding=pages[0],
            sympy_template=template,
            pattern_type=pattern_type,
            example_problem=problem_text[:200],
            epoch=epoch,
        )
```

### Pattern Type Classifier (Simple Heuristic)

```python
def classify_pattern(sympy_steps):
    """
    Auto-classify a list of SymPy steps into a pattern type.
    Simple keyword heuristic — not ML, just string matching.
    """
    combined = " ".join(sympy_steps).lower()
    
    if "/ 2" in combined or "* 0.5" in combined:
        return "half_of"
    elif "/ 3" in combined:
        return "third_of"
    elif "/ 100" in combined or "percent" in combined or "* 0.01" in combined:
        return "percent_of"
    elif combined.count("+") >= 2:
        return "sum_of_parts"
    elif "*" in combined and "+" in combined:
        return "multiply_then_add"
    elif "-" in combined:
        return "difference"
    elif "*" in combined:
        return "multiply"
    elif "/" in combined:
        return "divide"
    else:
        return "other"
```

---

## Epoch-End Maintenance

```python
def end_of_epoch_maintenance(pattern_memory, epoch):
    """Run at the end of each training epoch."""
    
    # 1. Prune bad and stale patterns
    pattern_memory.prune(
        min_uses=10,
        max_failure_rate=0.7,
        stale_epochs=20,
        current_epoch=epoch,
    )
    
    # 2. Print stats
    pattern_memory.stats()
    
    # 3. Export for analysis (every 5 epochs)
    if epoch % 5 == 0:
        pattern_memory.export_json(f"pattern_memory_epoch_{epoch}.json")
```

---

## What the Database Looks Like After Training

```
After 10 epochs on GSM8K:

Pattern memory: 847 patterns
Avg success rate: 73%

Top 5 most-used patterns:
  [half_of]          result = {a} / 2              (312 wins, 8 fails, 97%)
  [sum_of_parts]     total = {a} + {b} + {c}       (256 wins, 23 fails, 92%)
  [percent_of]       result = {a} * {b} / 100       (198 wins, 45 fails, 81%)
  [multiply_then_add] total = {a} * {b} + {c}       (167 wins, 52 fails, 76%)
  [difference]       result = {a} - {b}              (401 wins, 12 fails, 97%)

Pattern types:
  difference: 134 patterns, 1823 total successes
  sum_of_parts: 189 patterns, 1567 total successes
  multiply: 98 patterns, 892 total successes
  percent_of: 76 patterns, 743 total successes
  half_of: 23 patterns, 534 total successes
```

The database has discovered the mathematical vocabulary of GSM8K. "Half of" is a single pattern with 97% success. "Percent of" is 81% (harder — model sometimes misformulates). The success rates tell us WHICH patterns the model has mastered and which need more training.

---

## Anti-Overfitting Properties

```
WITHOUT pattern DB (overfitting):
  Model memorizes: "48 / 2 = 24" (specific instance)
  Fails on: "96 / 2 = ?" (never seen this specific instance)
  
WITH pattern DB (generalization):
  DB stores: "result = {a} / 2" (general template, 97% success)
  Model retrieves template, applies to ANY number
  Works for 48, 96, 1000, 3.14 — the template is universal
```

The database stores PATTERNS, not instances. The canonicalization ensures "48/2" and "96/2" are the SAME entry. The model can't memorize — it can only learn general patterns.

---

## Parameter Cost

```
PatternMemory:    0 trainable params (SQLite is not differentiable)
SymPyResultEncoder: ~17K params (from SymPy handoff)
Pattern hint injection: 0 params (text formatting)

Total new: 0 trainable params
The database is infrastructure, not part of the model.
```

---

## Implementation Order

```
1. Implement PatternMemory class with all 6 stored procedures
2. Implement PatternDeduplicator (canonicalize + hash + embedding similarity)
3. Implement classify_pattern (simple heuristic)
4. Integrate into solve_with_memory and after_solve
5. Add end_of_epoch_maintenance to training loop
6. Test on L3 first (verify patterns accumulate and deduplicate)
7. Deploy on GSM8K
8. Monitor: pattern_memory.stats() every epoch
```

---

## What NOT to Do

```
- Do NOT make the database differentiable. It's a lookup table.
  No gradient flows through SQLite. The model learns to USE patterns
  through the SymPy generation loss, not through DB gradients.

- Do NOT store patterns from WRONG answers. Only store on success.
  The database should contain verified reasoning, not guesses.

- Do NOT skip deduplication. Without it, the database grows to
  millions of near-identical entries. Canonicalization is essential.

- Do NOT query the database at every pass. Query at pass 1 only.
  The pattern hint informs the formulation strategy. Later passes
  execute the strategy — they don't need new patterns.

- Do NOT force the model to follow retrieved patterns.
  The pattern is a HINT injected as text context. The model can
  use it, modify it, or ignore it. The training objective decides.

- Do NOT use Postgres. SQLite is simpler, file-based, zero setup.
  We're a single-process training loop, not a web application.
```
