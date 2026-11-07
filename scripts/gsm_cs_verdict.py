"""gsm_cs_verdict.py -- ledger 2026-11-06 17:07: THE CHART-SEALED POINTER (BT_ZQ=1 BT_CS=1) vs the comparator (same engine, states, seeds, steps).
CS-R the per-pointer read (RESTR TEST, scripts/gsm_reader_ptr_read.py) - comparator mean >= +0.020, both > 0; CS-1 TEST step ptr, CS-2 TEST
step all: mean >= +0.010, both > 0; CS-C the key-chart probe's hit64 (the pick's ST64 chart is a valid one's) mean >= +0.020, both > 0.
Usage: gsm_cs_verdict.py ARM COMP  -> .cache/gsm_cs_verdict.txt"""
import json, sys

ARM, COMP = sys.argv[1], sys.argv[2]
J = lambda t: [json.load(open(f".cache/gsmb_breath_{t}_s{s}.json")) for s in (0, 1)]
X, C = J(ARM), J(COMP)
rp = lambda t: json.load(open(f".cache/gsm_reader_ptr_read{'' if t == 'BT4RB8' else '_' + t}.json"))
kp = lambda t: json.load(open(f".cache/gsm_keychart_probe{'' if t == 'BT4RB8' else '_' + t}.json"))
RX, RC, KX, KC = rp(ARM), rp(COMP), kp(ARM), kp(COMP)


def gain(name, xs, cs, bar):
    z = [a - b for a, b in zip(xs, cs)]; m = sum(z) / 2; ok = m >= bar and min(z) > 0
    return f"{name}: {ARM} - {COMP} {z[0]:+.4f} / {z[1]:+.4f} (mean {m:+.4f}; bar {bar:+.3f}, both > 0) -> {'PASS' if ok else 'FAIL'}"


e = [KX[f"s{s}"]["addr64"] for s in (0, 1)]
L = [gain("CS-R per-pointer RESTR TEST", [RX[f"s{s}_TEST"]["RESTR"] for s in (0, 1)], [RC[f"s{s}_TEST"]["RESTR"] for s in (0, 1)], 0.020),
     gain("CS-1 TEST_step_ptr", [r["TEST_step_ptr"] for r in X], [r["TEST_step_ptr"] for r in C], 0.010),
     gain("CS-2 TEST_step_all", [r["TEST_step_all"] for r in X], [r["TEST_step_all"] for r in C], 0.010),
     gain("CS-C TEST chart-level hit64", [KX[f"s{s}"]["hit64"] for s in (0, 1)], [KC[f"s{s}"]["hit64"] for s in (0, 1)], 0.020),
     f"  beside KP-E wrong picks in a valid ST64 chart: {ARM} {e[0]:.4f} / {e[1]:.4f} vs {COMP} "
     + " / ".join(f"{KC[f's{s}']['addr64']:.4f}" for s in (0, 1))]
L.append(f"  beside per-pointer TRAIN RESTR: {ARM} " + " / ".join(f"{RX[f's{s}_TRAIN']['RESTR']:.4f}" for s in (0, 1))
         + f" vs {COMP} " + " / ".join(f"{RC[f's{s}_TRAIN']['RESTR']:.4f}" for s in (0, 1)))
L.append(f"  beside KP-E n wrong: {ARM} " + " / ".join(str(KX[f's{s}']['n_wrong']) for s in (0, 1)) + f" vs {COMP} "
         + " / ".join(str(KC[f's{s}']['n_wrong']) for s in (0, 1)) + " | ST512 " + " / ".join(f"{KX[f's{s}']['addr512']:.4f}" for s in (0, 1)))
for k in ("TEST_step_op", "TEST_type_ok", "TEST_ans", "ORIG_step_all", "CFNAME_step_all", "CFNUM_step_all"):
    L.append(f"  beside {k}: {ARM} " + " / ".join(f"{r[k]:.4f}" for r in X) + f" vs {COMP} " + " / ".join(f"{r[k]:.4f}" for r in C))
L.append("  [engine: tinygrad train, torch score, both sides]")
print("\n".join(L)); open(".cache/gsm_cs_verdict.txt", "w").write("\n".join(L) + "\n")
