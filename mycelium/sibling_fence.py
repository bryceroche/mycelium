"""sibling_fence.py -- THE MATH-500 SIBLING FENCE (2026-10-18; the Sonnet audit: no exact MATH-500 text is in any diet, but
MATH-train TEMPLATE SIBLINGS are: 'y varies inversely as sqrt x, x=24, y=15' (MATH-500) vs the same sentence with x=2, y=4
(train) at similarity 0.97). The train split is the lawful harvest; a near-copy of a measured item is a memorization trap.
THE RULE: a training text whose difflib similarity to any MATH-500 problem is >= THRESH (0.90) is a SIBLING and may not
enter a diet. Candidates are found by a shared word 8-gram (similar texts share one), then scored exactly.
API: siblings(texts) -> [(i, ratio, math500_index)] for every text at or above THRESH."""
import difflib, json, re

THRESH = 0.90
_M5 = None


def _tok(s):
    return re.findall(r"\w+", s.lower())


def _index():
    global _M5
    if _M5 is None:
        probs = [json.loads(l)["problem"] for l in open(".cache/math500_test.jsonl")]
        grams = {}
        for j, p in enumerate(probs):
            w = _tok(p)
            for k in range(len(w) - 7):
                grams.setdefault(" ".join(w[k:k + 8]), set()).add(j)
        _M5 = (probs, grams)
    return _M5


def siblings(texts, thresh=THRESH):
    probs, grams = _index()
    out = []
    for i, t in enumerate(texts):
        w = _tok(t); cand = set()
        for k in range(len(w) - 7):
            cand |= grams.get(" ".join(w[k:k + 8]), set())
        for j in cand:
            r = difflib.SequenceMatcher(None, t, probs[j]).ratio()
            if r >= thresh:
                out.append((i, r, j))
    return out


if __name__ == "__main__":
    import sys
    for path in sys.argv[1:]:
        T = [json.loads(l).get("text") or json.loads(l).get("problem") for l in open(path)]
        S = siblings(T)
        print(f"[sibling-fence] {path}: {len(T)} texts | SIBLINGS (>= {THRESH}) {len(S)} in {len({i for i, _, _ in S})} rows")
        for i, r, j in S[:10]:
            print(f"   row {i} ~ MATH-500 #{j} at {r:.3f}: {T[i][:100]!r}")
