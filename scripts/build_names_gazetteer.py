"""build_names_gazetteer.py -- the photobooth's first-name gazetteer (2026-10-18). The audited photobooth swaps LISTED names
only; the hand lists missed most GSM8K names (Bill, Carrie, Tommy, Jeff, Katie...: 2,959 of 8,094 repeats changed). Source:
the NLTK names corpus (male.txt 2,943 / female.txt 5,001; a dataset download, .cache/names_corpus_dl/names). ADMITTED: a
name in exactly ONE gender list, alphabetic, whose lowercase form occurs lowercase < 3 times in GSM8K train questions (so
'bill', 'may', 'penny', 'joy' stay words), and not a month or weekday. Out: .cache/names_gazetteer.json {"M": [...], "F": [...]}"""
import collections, glob, json, re
import pyarrow.parquet as pq

D = ".cache/names_corpus_dl/names/"
M = {l.strip() for l in open(D + "male.txt") if l.strip()}
F = {l.strip() for l in open(D + "female.txt") if l.strip()}
snap = glob.glob(".cache/gsm8k/datasets--openai--gsm8k/snapshots/*/main/")[0]
low = collections.Counter(w for q in pq.read_table(snap + "train-00000-of-00001.parquet").column("question").to_pylist()
                          for w in re.findall(r"\b[a-z]+\b", q))
CAL = set("January February March April May June July August September October November December Monday Tuesday Wednesday "
          "Thursday Friday Saturday Sunday".split())
ok = lambda n: re.fullmatch(r"[A-Z][a-z]+", n) and low[n.lower()] < 3 and n not in CAL
out = {"M": sorted(n for n in M - F if ok(n)), "F": sorted(n for n in F - M if ok(n))}
C = collections.Counter(both=len(M & F), word=sum(1 for n in M ^ F if low[n.lower()] >= 3))
json.dump(out, open(".cache/names_gazetteer.json", "w"))
print(f"[gazetteer] M {len(out['M'])} F {len(out['F'])} | dropped: in both lists {C['both']}, a GSM8K lowercase word {C['word']} "
      f"(e.g. {sorted(n for n in M ^ F if low[n.lower()] >= 3)[:25]})")
