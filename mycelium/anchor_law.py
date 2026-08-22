"""anchor_law.py — THE SURFACE-ANCHOR LAW's one organ (2026-08-21).

Every gold given value must appear in the original text — as a digit
string or a number word (compounds included). Params (k, k1/k2, p) ride
op-lexical cues and are exempt; GIVENS are not.

Carved after the meter-divergence law struck twice in one day: the
reanno gate and the assembly door briefly held SEPARATE word maps and
disagreed on "twenty-seven". One organ, imported everywhere; a check
must call the organ it certifies.
"""
import re

_UNITS = ("zero one two three four five six seven eight nine ten eleven "
          "twelve thirteen fourteen fifteen sixteen seventeen eighteen "
          "nineteen").split()
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
         "seventy": 70, "eighty": 80, "ninety": 90}

WORDS = {w: v for v, w in enumerate(_UNITS)}
WORDS.update(_TENS)
for _tw, _tv in _TENS.items():
    for _uv in range(1, 10):
        WORDS[f"{_tw}-{_UNITS[_uv]}"] = _tv + _uv
WORDS.update({"half": 2, "doubled": 2, "twice": 2, "hundred": 100,
              "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
              "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9})


def surface_numbers(text):
    nums = set(int(x) for x in re.findall(r"\d+", text))
    low = text.lower()
    for w, v in WORDS.items():
        if re.search(rf"\b{re.escape(w)}\b", low):
            nums.add(v)
    return nums


def unanchored_givens(row):
    """Given values absent from the row's original text. Empty = lawful."""
    nums = surface_numbers(row["original"])
    return [f["value"] for f in row["factors"]
            if f["ftype"] == "given" and f["value"] not in nums]
