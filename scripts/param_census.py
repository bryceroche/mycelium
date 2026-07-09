"""param_census.py — the paper's parameter census (2026-07-10 flag).
Three rows, inclusion rules printed: TRAINED (gradient ever flowed) /
FROZEN-LEVERAGED (forward compute, no gradient) / ZERO-PARAMETER (mechanisms).
"""
import sys
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
from tinygrad.nn.state import safe_load

def count(path):
    try:
        return sum(int(v.shape[0]) * (int(v.shape[1]) if len(v.shape) > 1 else 1)
                   * (int(v.shape[2]) if len(v.shape) > 2 else 1)
                   for v in safe_load(path).values())
    except Exception as e:
        return f"?? {e}"

rows = {
    "parser head (breath/depth, ALG2)": ".cache/phase1_breath_depth.safetensors",
    "repair specialist (v2)": ".cache/phase1_algebra2_nack.safetensors",
    "deducer (fg_kenken_k16_reg)": ".cache/fg_ckpts/fg_kenken_k16_reg/fg_kenken_k16_reg_final.safetensors",
}
print("TRAINED (gradient ever flowed):")
tot = 0
for name, path in rows.items():
    c = count(path)
    print(f"  {name:38s} {c if isinstance(c, str) else f'{c/1e6:8.1f}M'}")
    if isinstance(c, int):
        tot += c
print(f"  {'TOTAL trained':38s} {tot/1e6:8.1f}M")
print("\nFROZEN-LEVERAGED (forward only): Llama-3.2-1B L0-L3 ~243.3M + embed 262.7M")
print("ZERO-PARAMETER: search tier, verifier, TTA vote, monitor centroids,")
print("  agreement signal, lattice policy — mechanisms, not weights.")
print("\nInclusion rule: 'trained' counts every checkpoint the deployed lattice")
print("loads; deducer counted though algebra pipeline doesn't call it (paper")
print("presents both phases). Token embeddings of the DEDUCER (~52M) are")
print("warm-started-then-trained: counted if its row is.")
