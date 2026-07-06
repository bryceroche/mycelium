"""am_gpu_recover.py — self-service 7900 XTX recovery after an AM-driver device hang.

Uses the venv python's granted capabilities (CAP_DAC_OVERRIDE + CAP_SYS_ADMIN from
setup_am_driver.sh) to do what previously needed sudo:
  1. PCI remove + rescan (the kernel driver's re-init performs the GPU reset),
  2. unbind amdgpu from the dGPU (the iGPU keeps the display),
  3. verify with a tiny tinygrad matmul in a SUBPROCESS (so a hang can't take
     this script down with it).

Run after any "Device hang detected":
  .venv/bin/python3 scripts/am_gpu_recover.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

PCI = "0000:03:00.0"
DEV_PATH = f"/sys/bus/pci/devices/{PCI}"


def w(path: str, content: str) -> bool:
    try:
        with open(path, "w") as f:
            f.write(content)
        return True
    except OSError as e:
        print(f"  [recover] write {path} failed: {e}")
        return False


def main() -> int:
    print(f"[recover] PCI remove {PCI}...")
    if not w(f"{DEV_PATH}/remove", "1"):
        return 1
    time.sleep(2)
    print("[recover] PCI rescan...")
    if not w("/sys/bus/pci/rescan", "1"):
        return 1
    time.sleep(3)
    if not os.path.exists(DEV_PATH):
        print("[recover] device did not re-enumerate — REBOOT required")
        return 1
    # amdgpu will have re-claimed the card during rescan; unbind it.
    drv = f"{DEV_PATH}/driver"
    if os.path.islink(drv) and "amdgpu" in os.readlink(drv):
        print("[recover] unbinding amdgpu from the dGPU...")
        if not w(f"{drv}/unbind", PCI):
            return 1
        time.sleep(1)
    print("[recover] smoke (subprocess)...")
    r = subprocess.run(
        [sys.executable, "-c",
         "from tinygrad import Tensor; "
         "print('smoke:', (Tensor.ones(128,128)@Tensor.ones(128,128)).sum().item())"],
        env={**os.environ, "DEV": "AMD"}, capture_output=True, text=True, timeout=180)
    print("  " + (r.stdout.strip() or r.stderr.strip().splitlines()[-1]))
    ok = "smoke: 2097152.0" in r.stdout
    print(f"[recover] {'OK — device recovered' if ok else 'FAILED — reboot likely required'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
