"""THE __main__ GUARD (ledger 2026-11-03, named; word given 11-04 "for the whole queue"): distill_reader.py gains
`if __name__ == "__main__":` around everything after its first imports.

Why: tinygrad's BEAM search runs a multiprocessing "spawn" Pool (PARALLEL defaults to the CPU count); spawned workers re-import the
parent's main module as __mp_main__, and with no guard every worker re-ran the whole training script. Since then the 8M runs train with
BEAM=2 PARALLEL=0. Under the guard a worker imports only the env defaults and the stdlib; the pool needs nothing else (its tasks are
tinygrad functions). Inside the if-block every name stays a module global, so nothing changes for a normal run.
Verified mechanically: the guarded body's AST equals the original statements, one for one. Asserts the anchor once.
Usage: dr_main_guard.py [path]   (default scripts/distill_reader.py; dry runs pass a COPY)
"""
import ast, os, sys

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
P = sys.argv[1] if len(sys.argv) > 1 else os.path.join(R, "scripts", "distill_reader.py")
A = "ap = argparse.ArgumentParser()\n"
s = open(P).read()
assert s.count("\n" + A) == 1 and 'if __name__ == "__main__":' not in s, "anchor not found once, or already guarded"
i = s.index("\n" + A) + 1
head, body = s[:i], s[i:]
new = head + 'if __name__ == "__main__":   # (the __main__ guard, ledger 11-04) spawned BEAM workers re-import this module: they get only the above\n' \
      + "".join(("    " + l) if l.strip() else l for l in body.splitlines(True))
old_t, new_t = ast.parse(s), ast.parse(new)
k = len(ast.parse(head).body)
assert [ast.dump(n) for n in old_t.body[:k]] == [ast.dump(n) for n in new_t.body[:k]]
g = new_t.body[k]; assert isinstance(g, ast.If) and len(new_t.body) == k + 1
assert [ast.dump(n) for n in old_t.body[k:]] == [ast.dump(n) for n in g.body], "guarded body differs from the original"
open(P, "w").write(new); print(f"[guard] {P}: {len(old_t.body) - k} top-level statements under __main__ (AST identical)")
