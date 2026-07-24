"""pad_warm_1024.py — the lawful width rite: crown_reader_v4 (H_W=512)
zero-embedded into H_W=1024 shapes. Old function preserved at init on
every 512-axis; new capacity revived by gradient."""
import os, sys
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
os.environ["DEV"] = "CPU"
os.environ["ALG_HW"] = "1024"
os.environ.setdefault("ALG2", "1"); os.environ["ALG_FTYPES"] = "8"
os.environ.setdefault("ALG_DUP", "1")
from phase1_algebra_head import build_params
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load, safe_save

wide = build_params(0)
sd = safe_load(".cache/crown_reader_v4.safetensors")
out = {}
n_pad = n_copy = 0
for k, wt in wide.items():
    tgt = wt.shape
    src = sd[k].numpy()
    if tuple(src.shape) == tuple(tgt):
        out[k] = Tensor(src.astype(np.float32), dtype=dtypes.float)
        n_copy += 1
        continue
    buf = np.zeros(tgt, np.float32)
    sl = tuple(slice(0, s) for s in src.shape)
    buf[sl] = src
    out[k] = Tensor(buf, dtype=dtypes.float)
    n_pad += 1
    print(f"  pad {k}: {tuple(src.shape)} -> {tuple(tgt)}")
safe_save(out, ".cache/g19_padwarm_init.safetensors")
print(f"[padwarm] {n_copy} copied, {n_pad} zero-embedded -> g19_padwarm_init.safetensors")
