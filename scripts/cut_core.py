"""cut_core.py — THE CORE CUT (2026-08-28, word given): rebuild the orphan
branch `core` as a machine-cut minimal view of the deployed stack.

Laws of the cut:
  * `core` is an EXPORT TARGET, never a source — hand commits are clobbered
    by the next cut, by design (the word and the write are one atomic act).
  * The curation judgment is the ENTRYPOINT list only (the deployed-stack
    table, CLAUDE.md S1); the file list is DERIVED as the transitive local
    import closure — a new dependency rides into the next cut mechanically.
  * The manifest (.cache/GENERATION.json) is copied into the tree: the core
    branch states exactly which weights it composes with. Weights themselves
    never enter git.
  * The cut smoke-tests itself from a temp checkout of the new commit
    (ast-parse everything + import the pure-python organs); a broken export
    fails loudly and the ref is not moved.
  * Re-cut at chapter boundaries, same cadence as the gen-weights -> main
    sync. A stale core that looks authoritative is worse than none.

Usage: .venv/bin/python3 scripts/cut_core.py            (cut + smoke + update ref)
       .venv/bin/python3 scripts/cut_core.py --dry-run  (list the closure, touch nothing)
"""
import ast
import os
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The deployed stack (CLAUDE.md S1) + the fences at its doors. This list is
# the ONLY curated judgment in the cut; everything else is derived.
ENTRYPOINTS = [
    "mycelium/llama_loader.py",            # trunk (frozen)
    "scripts/phase1_algebra_head.py",      # parser head
    "scripts/phase1_algebra_nack.py",      # specialist / NACK
    "scripts/tta_views.py",                # TTA vote
    "scripts/tta_alg2_dials.py",
    "scripts/lattice_member_votes.py",     # cert-v2 panel
    "scripts/lattice_join.py",
    "scripts/recognition_mouth.py",        # the mouth
    "scripts/mouth_recal_gen9b.py",
    "mycelium/csp_core.py",                # the solving jaw
    "mycelium/csp_domains.py",
    "mycelium/custody_gold.py",            # fences
    "mycelium/diagnostic_register.py",
    "mycelium/complex_tensor.py",          # the bus's formal language
]
EXTRA_FILES = ["CLAUDE.md"]                # the agent brief rides along
MANIFEST = ".cache/GENERATION.json"        # copied into tree as GENERATION.json
BRANCH = "refs/heads/core"
# Organs importable without GPU/weights — the smoke's import set.
PURE_IMPORTS = ["mycelium.csp_core", "mycelium.csp_domains",
                "mycelium.custody_gold", "mycelium.complex_tensor"]


def sh(*cmd, env=None, input=None):
    r = subprocess.run(cmd, cwd=REPO, env=env, input=input,
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)}\n{r.stderr}")
    return r.stdout.strip()


def local_imports(path):
    """Module names imported by `path` that resolve to repo-local files.
    Scripts import flat (sys.path has scripts/); package imports are
    mycelium.x — both resolve here."""
    tree = ast.parse(open(os.path.join(REPO, path)).read())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
            if node.module == "mycelium":       # from mycelium import x
                names.update(f"mycelium.{a.name}" for a in node.names)
    out = set()
    for n in names:
        root = n.split(".")
        for cand in (f"{'/'.join(root)}.py", f"scripts/{root[0]}.py",
                     f"mycelium/{root[0]}.py" if not n.startswith("mycelium") else None):
            if cand and os.path.exists(os.path.join(REPO, cand)):
                out.add(cand)
                break
    return out


def closure():
    seen, todo = set(), list(ENTRYPOINTS)
    if os.path.exists(os.path.join(REPO, "mycelium/__init__.py")):
        todo.append("mycelium/__init__.py")   # a seed, not an afterthought:
    for ep in ENTRYPOINTS:                    # its imports must enter the walk
        assert os.path.exists(os.path.join(REPO, ep)), f"entrypoint missing: {ep}"
    while todo:
        f = todo.pop()
        if f in seen:
            continue
        seen.add(f)
        todo.extend(local_imports(f) - seen)
    return sorted(seen)


def readme(files, manifest_gen):
    return f"""# mycelium — core cut

Machine-cut minimal view of the deployed Mycelium stack ({len(files)} files;
generation: {manifest_gen}). This branch is an EXPORT TARGET rebuilt by
`scripts/cut_core.py` on main — never commit here by hand; the next cut
clobbers it by design.

Full history, ledger, and provenance live on the main branches of this
same repository. GENERATION.json (included) names the exact weight
artifacts this code composes with; weights are distributed separately
and are never in git.

Chain of custody: mouth (register) -> vote (diagram-invariance) ->
panel (landscape-invariance) -> key (truth).
"""


def main():
    dry = "--dry-run" in sys.argv
    files = closure()
    if dry:
        print("\n".join(files))
        print(f"[cut] closure = {len(files)} files (dry run; ref untouched)")
        return

    import json
    gen = json.load(open(os.path.join(REPO, MANIFEST))).get("gen_id", "?")
    index = tempfile.NamedTemporaryFile(prefix="core_index_", delete=False)
    index.close()
    os.unlink(index.name)      # git refuses a pre-existing empty index file
    env = dict(os.environ, GIT_INDEX_FILE=index.name)
    try:
        for f in files + [x for x in EXTRA_FILES
                          if os.path.exists(os.path.join(REPO, x))]:
            sh("git", "update-index", "--add", f, env=env)
        for name, content in (("GENERATION.json",
                               open(os.path.join(REPO, MANIFEST)).read()),
                              ("README.md", readme(files, gen))):
            blob = sh("git", "hash-object", "-w", "--stdin", input=content)
            sh("git", "update-index", "--add", "--cacheinfo",
               f"100644,{blob},{name}", env=env)
        tree = sh("git", "write-tree", env=env)
    finally:
        if os.path.exists(index.name):
            os.unlink(index.name)

    try:
        parent = sh("git", "rev-parse", "--verify", "-q", BRANCH)
    except RuntimeError:
        parent = None
    head = sh("git", "rev-parse", "--short", "HEAD")
    msg = f"core cut: {len(files)} files, gen {gen}, cut from {head}"
    args = ["git", "commit-tree", tree, "-m", msg]
    if parent:
        args[3:3] = ["-p", parent]
    commit = sh(*args)

    # THE SMOKE GATE: checkout the new commit somewhere clean; parse + import.
    with tempfile.TemporaryDirectory(prefix="core_smoke_") as td:
        tar = subprocess.run(["git", "archive", commit], cwd=REPO,
                             capture_output=True)
        assert tar.returncode == 0, tar.stderr.decode()
        subprocess.run(["tar", "-x"], cwd=td, input=tar.stdout, check=True)
        for root, _, fs in os.walk(td):
            for f in fs:
                if f.endswith(".py"):
                    ast.parse(open(os.path.join(root, f)).read())
        py = os.path.join(REPO, ".venv/bin/python3")
        r = subprocess.run([py, "-c",
                            "import " + ",".join(PURE_IMPORTS)],
                           cwd=td, capture_output=True, text=True)
        assert r.returncode == 0, f"smoke import failed:\n{r.stderr}"

    sh("git", "update-ref", BRANCH, commit)
    print(f"[cut] core -> {commit[:12]}  ({len(files)} files, gen {gen}, "
          f"smoke PASS: ast-parse all + import {len(PURE_IMPORTS)} organs)")


if __name__ == "__main__":
    main()
