"""scope_mouth_exam.py — the doorman's testimony on the manufactured
scope specimens (2026-07-20). Gen-13 mouth, corrected threshold."""
import sys, os, json
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
os.environ.setdefault("ALG2", "1"); os.environ.setdefault("ALG_FTYPES", "6")
import numpy as np
from phase1_algebra_head import T_ALG, TOKENIZER_JSON
from beacon_closing_arm import recompute_states
from tokenizers import Tokenizer

mouth = np.load(".cache/recognition_mouth_gen13.npz")
bank, coef, thr = mouth["bank"].astype(np.float32), mouth["coef"], float(mouth["thr_knn"])
tok = Tokenizer.from_file(TOKENIZER_JSON)
VALS = [(7,4),(9,5),(8,3),(11,6),(12,7),(13,4),(10,3),(15,8),(14,5),(9,2)]
D = "Consider the numbers "
texts = []
for a, b in VALS:
    texts.append(D + f"a, b, c. a is {a}. b is {b}. The difference of the squares of a and b equals c. What is c?")
    texts.append(D + f"a, b, c. a is {a}. b is {b}. The square of the difference of a and b equals c. What is c?")
ids = np.zeros((len(texts), T_ALG), np.int32)
msk = np.zeros((len(texts), T_ALG), np.float32)
L = []
for i, t in enumerate(texts):
    e = tok.encode(t)
    Ln = min(len(e.ids), T_ALG)
    ids[i, :Ln] = e.ids[:Ln]; msk[i, :Ln] = 1.0; L.append(Ln)
sts = recompute_states(ids)
V = (sts.astype(np.float32) * msk[:, :, None]).sum(1) / np.maximum(msk.sum(1)[:, None], 1)
V /= np.linalg.norm(V, axis=1, keepdims=True)
d = np.sort(1.0 - V @ bank.T, axis=1)[:, :8].mean(1) - (coef[0] + coef[1] / np.array(L, float))
refused = d > thr
print(f"[mouth-exam] thr {thr:+.4f} | d in [{d.min():+.4f}, {d.max():+.4f}] | REFUSED {int(refused.sum())}/20")
json.dump({"d": d.tolist(), "refused": refused.tolist(), "thr": thr},
          open(".cache/scope_mouth_exam.json", "w"))
