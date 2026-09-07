"""drift_autopsy.py — parameter-drift autopsy between two snapshots of one
run, against a reference run's drift over the same steps (2026-09-07; the
dose-0.15 death's fingerprint: whole-head drift 3.3x, alt21_W_bk_b 966x).
Zero GPU. Usage: DA_A=a.safetensors DA_B=b.safetensors DA_RA=ra.safetensors
DA_RB=rb.safetensors python drift_autopsy.py"""
import os, json, struct, numpy as np

def load(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]; hdr = json.loads(f.read(n)); base = 8 + n
        out = {}
        for k, v in hdr.items():
            if k == "__metadata__": continue
            s, e = v["data_offsets"]; f.seek(base + s)
            dt = {"F32": np.float32, "F16": np.float16, "I32": np.int32, "I64": np.int64}[v["dtype"]]
            out[k] = np.frombuffer(f.read(e - s), dtype=dt).reshape(v["shape"])
        return out

def rel(a, b):
    a = a.astype(np.float64); b = b.astype(np.float64)
    return np.linalg.norm(b - a) / (np.linalg.norm(a) + 1e-12)

def main():
    A, B = load(os.environ["DA_A"]), load(os.environ["DA_B"])
    RA, RB = load(os.environ["DA_RA"]), load(os.environ["DA_RB"])
    rows = []
    for k in A:
        if A[k].dtype != np.float32: continue
        rows.append((rel(A[k], B[k]), rel(RA[k], RB[k]), k, float(np.abs(B[k]).max())))
    rows.sort(reverse=True)
    ratios = [r[0] / max(r[1], 1e-9) for r in rows]
    tag = os.path.basename(os.environ["DA_B"])
    print(f"[drift] {tag}: median per-key ratio vs reference = {np.median(ratios):.2f}  "
          f"(failed d15 arm: 3.34); non-finite params: {any(not np.isfinite(B[k]).all() for k in B)}")
    for r_d, r_p, k, m in rows[:8]:
        print(f"[drift]   {k:22s} rel {r_d:8.4f} vs ref {r_p:8.4f}  ratio {r_d/max(r_p,1e-9):8.1f}  max|theta| {m:.3g}")
    kb = next((r for r in rows if r[2] == "alt21_W_bk_b"), None)
    if kb: print(f"[drift]   alt21_W_bk_b ratio = {kb[0]/max(kb[1],1e-9):.1f}  (failed arm: 965.9)")

if __name__ == "__main__":
    main()
