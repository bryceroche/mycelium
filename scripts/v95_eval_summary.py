"""v95 eval summary aggregator — reads a DUMP_ALL_DECODES JSONL and emits the
core v95 metrics:
  - accuracy %
  - DAG executable (sympy returned a value) %
  - DAG parse rate (extractable DAG) %
  - failure breakdown
  - undefined_variable %

Run after `eval_v77_dag.py` was invoked with DUMP_ALL_DECODES=<path>.
Usage:  python scripts/v95_eval_summary.py <dump.jsonl>
"""
import sys
import json
import re

# Reuse the same classify_failure shape as eval_v77_dag.py.
_VAR_RE = re.compile(r"\bx\d+\b")


def classify(dag, sympy_val, correct):
    if correct:
        return "correct"
    if not dag:
        return "no_dag_found"
    if sympy_val is not None:
        return "parseable_but_wrong_answer"
    # Check substructure
    if "answer" not in dag:
        return "no_answer_statement"
    defined = set()
    statements = [s.strip() for s in dag.split(";") if s.strip()]
    for stmt in statements:
        if "=" not in stmt:
            return "malformed_statement"
        lhs, _, rhs = stmt.partition("=")
        lhs = lhs.strip()
        used = _VAR_RE.findall(rhs)
        for uv in used:
            if uv not in defined:
                return "undefined_variable"
        defined.add(lhs)
    if "/0" in dag.replace(" ", "") or "/ 0" in dag:
        return "div_by_zero"
    return "sympy_parse_error"


def main():
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <dump.jsonl>", file=sys.stderr)
        sys.exit(2)
    path = sys.argv[1]
    total = 0
    correct = 0
    parseable = 0  # has sympy_val (executable)
    extractable = 0  # has any dag string
    cats = {}
    samples_correct = []
    samples_wrong = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            total += 1
            dag = rec.get("dag", "")
            sympy_val = rec.get("sympy_val")
            ok = rec.get("correct", False)
            if ok:
                correct += 1
                if len(samples_correct) < 3:
                    samples_correct.append(rec)
            else:
                if len(samples_wrong) < 3:
                    samples_wrong.append(rec)
            if sympy_val is not None:
                parseable += 1
            if dag:
                extractable += 1
            cat = classify(dag, sympy_val, ok)
            cats[cat] = cats.get(cat, 0) + 1

    print(f"=== {path} ===")
    print(f"  total examples:        {total}")
    print(f"  accuracy:              {100.0 * correct / max(total, 1):.1f}%  ({correct}/{total})")
    print(f"  DAG executable rate:   {100.0 * parseable / max(total, 1):.1f}%  ({parseable}/{total})")
    print(f"  DAG extractable rate:  {100.0 * extractable / max(total, 1):.1f}%  ({extractable}/{total})")
    print(f"  failure / outcome breakdown:")
    for cat, n in sorted(cats.items(), key=lambda kv: -kv[1]):
        print(f"    {cat:30s}  {n:4d}  ({100.0 * n / max(total, 1):.1f}%)")
    if samples_correct:
        print(f"\n  Sample CORRECT decodes:")
        for r in samples_correct:
            print(f"    Q: {r['problem'][:100]!r}")
            print(f"    DAG: {r['dag']!r}")
            print(f"    val={r['sympy_val']} gold={r['gold']}")
    if samples_wrong:
        print(f"\n  Sample WRONG decodes:")
        for r in samples_wrong[:3]:
            print(f"    Q: {r['problem'][:100]!r}")
            print(f"    DAG: {r['dag']!r}")
            print(f"    val={r['sympy_val']} gold={r['gold']}")


if __name__ == "__main__":
    main()
