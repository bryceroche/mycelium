"""stamp_value_spans.py — THE ZERO-TOKEN VALUE SPANS (2026-09-19, the breath-
sharpening arm item #2 from the 11:50 ledger entry): for a GIVEN factor the
value is a numeral already sitting in the text — its span costs no annotation
to find. Stamps `mentions[str(var)]` with every occurrence of the given's
value as a standalone numeral (regex, comma/decimal/currency/percent/sign
aware) or, when no numeral occurrence exists, a lexicon word span whose value
matches (mycelium.lexicon.constants — cardinal words + curated entries, the
chain lane's tier-2 givens). Keeps ALL occurrences (a multi-positive mask is
honest for the vspan BCE loss); never removes an existing mention.

NEVER touches a factor's own `spans` list and never reorders factors (THE
POSITIONAL LAW, docs/phase1_skeleton_spec.md 2026-09-17): the gold builder
sorts factors by span only on rows where every factor has spans and the row
is not stamped canonical=positional; `mentions` plays no part in that sort.

Usage: .venv/bin/python3 scripts/stamp_value_spans.py [IN] [OUT]
  (defaults: .cache/form_mix_pm35.jsonl -> .cache/form_mix_pm35v.jsonl)
"""
import json, re, sys

sys.path.insert(0, ".")
from mycelium.lexicon import constants as lex_constants

IN = sys.argv[1] if len(sys.argv) > 1 else ".cache/form_mix_pm35.jsonl"
OUT = sys.argv[2] if len(sys.argv) > 2 else ".cache/form_mix_pm35v.jsonl"

# A standalone numeral: comma-grouped ("1,200") or plain ("1200"), optional
# decimal tail ("2.5" — the greedy optional group already absorbs it, so the
# boundary check below never has to special-case "."; sentence-final "18."
# must still match on the digits alone, and it does since the group only
# consumes the dot when a digit follows it); never a substring of a longer
# digit run ("12" inside "120"), an ordinal suffix ("21st"), or an
# alphanumeric code ("A5") — those are blocked by the \w boundary. The sign
# is never part of the matched span — digits store |value|, sign is a
# separate gold channel (E1) — so a preceding "-" is permitted but not
# consumed; we match on abs(value) throughout and let the existing sign gold
# carry the rest.
NUM_RE = re.compile(r'(?<!\w)(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)(?!\w)')


def numeral_occurrences(text, target):
    out = []
    for m in NUM_RE.finditer(text):
        raw = m.group().replace(",", "")
        if "." in raw:
            fv = float(raw)
            if not fv.is_integer():
                continue
            val = int(fv)
        else:
            val = int(raw)
        if val == target:
            out.append([m.start(), m.end()])
    return out


def lexicon_occurrences(text, target):
    out = []
    for s, e, v in lex_constants(text):
        if v == target:
            out.append([s, e])
    return out


def add_spans(mentions, var, spans):
    lst = mentions.setdefault(str(var), [])
    existing = {tuple(sp) for sp in lst}
    for sp in spans:
        t = tuple(sp)
        if t not in existing:
            lst.append(list(sp))
            existing.add(t)


def main():
    n_numeral = n_lexicon = n_unresolved = n_givens = 0
    unresolved_examples = []
    rows = []
    with open(IN) as f:
        lines = f.readlines()
    for li, line in enumerate(lines):
        row = json.loads(line)
        text = row["text"]
        mentions = row.setdefault("mentions", {})
        for fac in row["factors"]:
            if fac.get("ftype") != "given":
                continue
            n_givens += 1
            v = fac["var"]
            x = int(fac["value"])
            target = abs(x)
            spans = numeral_occurrences(text, target)
            src = "numeral"
            if not spans:
                spans = lexicon_occurrences(text, target)
                src = "lexicon"
            if not spans:
                n_unresolved += 1
                if len(unresolved_examples) < 10:
                    unresolved_examples.append((li, v, x, text))
                continue
            if src == "numeral":
                n_numeral += 1
            else:
                n_lexicon += 1
            add_spans(mentions, v, spans)
        rows.append(row)

    with open(OUT, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    print(f"[stamp] {len(rows)} rows -> {OUT}")
    print(f"[stamp] given factors total: {n_givens}")
    print(f"[stamp] resolved by numeral:  {n_numeral}")
    print(f"[stamp] resolved by lexicon:  {n_lexicon}")
    print(f"[stamp] UNRESOLVED:           {n_unresolved}")
    print("[stamp] unresolved examples (up to 10):")
    for li, v, x, text in unresolved_examples:
        print(f"  row {li} var={v} value={x}: {text[:180]!r}")


if __name__ == "__main__":
    main()
