"""admit_annotation.py — THE ADMISSION GATE for silver annotations (2026-09-15,
the pivot to wild prose): an annotated row (Sonnet's or a human's) enters
the diet ONLY if (1) the rulebook holds — variables dense 0..n_vars-1 and
every referenced variable defined, values integers <= 300, at most one fdiv,
spans/mentions inside the text and non-empty, a query_var — and (2) the
answer key gates it: wild_certify.graph_verdict propagates the annotated
factor graph and it must FORCE the key at query_var (the gsm8k answer
field's "#### N"; the pen's solution fields are never gold). Admitted rows
are stamped gen.silver = <version> and written apart from the diet.
usage: admit_annotation.py annotations.jsonl admitted_out.jsonl [silver_version]
Self-test: --selftest runs the gate over mint rows with their own key."""
import json, re, sys
sys.path.insert(0, "."); sys.path.insert(0, "scripts")


def key_of(answer_field):
    m = re.search(r"####\s*(-?[\d,]+)", answer_field or "")
    return int(m.group(1).replace(",", "")) if m else None


def rulebook(row):
    """returns None when the row is well-formed, else the reason"""
    n = row.get("n_vars"); facs = row.get("factors") or []; text = row.get("text", "")
    if not isinstance(n, int) or n <= 0: return "n_vars"
    if not facs: return "no_factors"
    if "query_var" not in row or not (0 <= row["query_var"] < n): return "query_var"
    used = set(); defined = set(); n_fdiv = 0
    for f in facs:
        ft = f.get("ftype")
        if ft == "given":
            v = f.get("value")
            if not isinstance(v, int) or not (0 <= v <= 300): return f"value_{v}"
            if not (0 <= f.get("var", -1) < n): return "given_var"
            defined.add(f["var"]); used.add(f["var"])
        elif ft == "rel":
            if f.get("op") not in ("add", "sub", "mul", "div"): return f"op_{f.get('op')}"
            a = f.get("args", []); r = f.get("result")
            if len(a) != 2 or not all(0 <= x < n for x in a) or not (0 <= (r if r is not None else -1) < n): return "rel_pointers"
            used.update(a); used.add(r); defined.add(r)
        elif ft == "fdiv":
            n_fdiv += 1; used.update(f.get("args", [])); defined.add(f.get("result", -1))
        else:
            return f"ftype_{ft}"
        for s, e in f.get("spans") or []:
            if not (0 <= s < e <= len(text)): return "span_bounds"
    if n_fdiv > 1: return "fdiv_count"
    # the dialect is a CONSTRAINT graph, not a directed computation: an unknown may be
    # determined only through relations it enters as an argument (inverse moves) — so the
    # rulebook asks only that the query appears in some factor; DETERMINACY is the
    # certifier's verdict (graph_verdict: forces the key / contradicts / indeterminate)
    if row["query_var"] not in used: return "query_not_in_graph"
    for v_str, spans in (row.get("mentions") or {}).items():
        if not (0 <= int(v_str) < n): return "mention_var"
        for s, e in spans:
            if not (0 <= s < e <= len(text)): return "mention_bounds"
    return None


def admit(rows, version="sonnet_v1"):
    from wild_certify import graph_verdict
    out = []; why = {}
    for r in rows:
        reason = rulebook(r)
        if reason is None:
            key = r.get("key", key_of(r.get("answer_field")))
            if key is None: reason = "no_key"
            else:
                ok, detail = graph_verdict(r, key)
                reason = None if ok is True else ("key_contradicted" if ok is False else f"indeterminate:{detail}")
        if reason is None:
            r = dict(r); g = dict(r.get("gen") or {}); g.update({"src": "gsm8k", "silver": version, "admitted_by": "admit_annotation:key+rulebook"}); r["gen"] = g; out.append(r)
        else:
            why[reason.split(":")[0]] = why.get(reason.split(":")[0], 0) + 1
    return out, why


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        rows = [json.loads(l) for l in open(".cache/algebra_nl_test.jsonl")][:200]
        for r in rows: r["key"] = r["solution"][r["query_var"]]
        adm, why = admit(rows, "selftest"); print(f"[admit] self-test on 200 mint rows with their own key: admitted {len(adm)}; refused {why}")
        bad = dict(rows[0]); bad["factors"] = [dict(f) for f in bad["factors"]]; bad["factors"][0]["value"] = 999 if bad["factors"][0]["ftype"] == "given" else bad["factors"][0].get("value"); bad["key"] = -1
        a2, w2 = admit([bad], "selftest"); assert not a2, "a contradicted row was admitted"; print(f"[admit] a contradicted/out-of-range row is refused: {w2}")
        sys.exit(0)
    rows = [json.loads(l) for l in open(sys.argv[1])]; adm, why = admit(rows, sys.argv[3] if len(sys.argv) > 3 else "sonnet_v1")
    with open(sys.argv[2], "w") as f:
        for r in adm: f.write(json.dumps(r) + "\n")
    print(f"[admit] {len(rows)} annotated rows -> {len(adm)} admitted to {sys.argv[2]} (silver, versioned apart); refused: {why}")
