#!/bin/bash
# Persistent setup for tinygrad's AM driver on the 7900 XTX (Shadow Glass).
# Run as root:  sudo bash scripts/setup_am_driver.sh
#
# After this runs:
#   - udev rule grants user 666 mode on 7900 XTX PCI sysfs files (every boot)
#   - systemd service unbinds amdgpu from the 7900 XTX at boot (every boot)
#   - python binary gets the capabilities needed for PCI BAR access
#     (CAP_DAC_OVERRIDE, CAP_SYS_RAWIO, CAP_SYS_ADMIN, CAP_IPC_LOCK)
#   - all also applied NOW so we don't need to reboot
#
# 7900 XTX is at PCI bus 0000:03:00.0 with vendor=0x1002 device=0x744c.
# The iGPU (Raphael at 12:00.0) stays bound to amdgpu so the display works.

set -e

if [ "$EUID" -ne 0 ]; then
  echo "must run as root: sudo bash $0"
  exit 1
fi

PCI_ID="0000:03:00.0"
VENDOR="0x1002"
DEVICE="0x744c"
# Resolve the venv python to its real binary (setcap doesn't follow symlinks)
VENV_PYTHON="/home/bryce/mycelium/.venv/bin/python"
REAL_PYTHON="$(readlink -f "$VENV_PYTHON")"
echo "venv python resolves to: $REAL_PYTHON"

echo "1. Writing udev rule for persistent permissions..."
cat > /etc/udev/rules.d/99-tinygrad-amd.rules <<EOF
SUBSYSTEM=="pci", ATTR{vendor}=="$VENDOR", ATTR{device}=="$DEVICE", MODE="0666"
EOF

echo "2. Writing systemd unit for persistent amdgpu unbind..."
cat > /etc/systemd/system/tinygrad-unbind.service <<EOF
[Unit]
Description=Unbind amdgpu from 7900 XTX so tinygrad AM driver can claim it
DefaultDependencies=no
After=systemd-modules-load.service
Before=multi-user.target

[Service]
Type=oneshot
ExecStart=/bin/sh -c "echo $PCI_ID > /sys/bus/pci/devices/$PCI_ID/driver/unbind || true"
ExecStart=/bin/sh -c "chmod 666 /sys/bus/pci/devices/$PCI_ID/enable /sys/bus/pci/devices/$PCI_ID/resource* /sys/bus/pci/devices/$PCI_ID/config"
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF

echo "3. Enabling + starting the systemd unit now..."
systemctl daemon-reload
systemctl enable tinygrad-unbind.service
systemctl start tinygrad-unbind.service

echo "4. Reloading + triggering udev rules..."
udevadm control --reload-rules
udevadm trigger

echo "5. Granting Linux capabilities to python (for PCI BAR access)..."
echo "   Target: $REAL_PYTHON"
setcap 'cap_dac_override,cap_sys_rawio,cap_sys_admin,cap_ipc_lock=ep' "$REAL_PYTHON"
echo "   Caps now:"
getcap "$REAL_PYTHON"

echo
echo "=== Done. AM driver should now be available. ==="
echo "Verify with: ls -la /sys/bus/pci/devices/$PCI_ID/enable"
echo "Run tinygrad with: DEV='PCI+AMD' python ..."
