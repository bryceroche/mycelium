"""gsm_zq_verdict.py -- ledger 2026-11-06 11:59: THE ZOOMING QUERY (BT_ZQ=1) vs the comparator (same engine, states, seeds, steps).
ZQ-R the per-pointer read (RESTR TEST, scripts/gsm_reader_ptr_read.py) - comparator mean >= +0.020, both > 0; ZQ-1 TEST step ptr, ZQ-2 TEST
step all: mean >= +0.010, both > 0; ZQ-E the key-chart probe's KP-E on the arm's weights (wrong picks in a valid ST64 chart) >= 0.15 both.
Usage: gsm_zq_verdict.py ARM COMP  -> .cache/gsm_zq_verdict.txt"""
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
L = [gain("ZQ-R per-pointer RESTR TEST", [RX[f"s{s}_TEST"]["RESTR"] for s in (0, 1)], [RC[f"s{s}_TEST"]["RESTR"] for s in (0, 1)], 0.020),
     gain("ZQ-1 TEST_step_ptr", [r["TEST_step_ptr"] for r in X], [r["TEST_step_ptr"] for r in C], 0.010),
     gain("ZQ-2 TEST_step_all", [r["TEST_step_all"] for r in X], [r["TEST_step_all"] for r in C], 0.010),
     f"ZQ-E: wrong picks in a valid ST64 chart {e[0]:.4f} / {e[1]:.4f} (bar >= 0.15 both; {COMP} "
     + " / ".join(f"{KC[f's{s}']['addr64']:.4f}" for s in (0, 1)) + f") -> {'PASS' if min(e) >= 0.15 else 'FAIL'}"]
L.append(f"  beside per-pointer TRAIN RESTR: {ARM} " + " / ".join(f"{RX[f's{s}_TRAIN']['RESTR']:.4f}" for s in (0, 1))
         + f" vs {COMP} " + " / ".join(f"{RC[f's{s}_TRAIN']['RESTR']:.4f}" for s in (0, 1)))
L.append(f"  beside KP-E n wrong: {ARM} " + " / ".join(str(KX[f's{s}']['n_wrong']) for s in (0, 1)) + f" vs {COMP} "
         + " / ".join(str(KC[f's{s}']['n_wrong']) for s in (0, 1)) + " | ST512 " + " / ".join(f"{KX[f's{s}']['addr512']:.4f}" for s in (0, 1)))
for k in ("TEST_step_op", "TEST_type_ok", "TEST_ans", "ORIG_step_all", "CFNAME_step_all", "CFNUM_step_all"):
    L.append(f"  beside {k}: {ARM} " + " / ".join(f"{r[k]:.4f}" for r in X) + f" vs {COMP} " + " / ".join(f"{r[k]:.4f}" for r in C))
L.append("  [engine: tinygrad train, torch score, both sides]")
print("\n".join(L)); open(".cache/gsm_zq_verdict.txt", "w").write("\n".join(L) + "\n")
