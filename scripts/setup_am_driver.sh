#!/bin/bash
# Persistent setup for tinygrad's AM driver on the 7900 XTX (Shadow Glass).
# Run as root:  sudo bash scripts/setup_am_driver.sh
#
# After this runs:
#   - udev rule grants user 666 mode on 7900 XTX PCI sysfs files (every boot)
#   - systemd service unbinds amdgpu from the 7900 XTX at boot (every boot,
#     after the kernel has actually bound amdgpu to the device — earlier
#     versions ran too early and silently no-op'd)
#   - sysctl drop-in disables migration of locked pages (tinygrad pins
#     sysmem for AQL queues; the kernel would otherwise migrate them out
#     from under it -> "Failed to disable migration of locked pages")
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
# Ordering note: amdgpu is autoloaded by udev when it coldplugs the GPU,
# which happens after systemd-modules-load.service. The earlier version of
# this unit ran with After=systemd-modules-load and so the unbind fired
# before any driver was bound — silently no-op'd, leaving amdgpu owning
# the device after boot. Fix: wait for the driver symlink to appear (with
# a 30s timeout) before unbinding.
cat > /etc/systemd/system/tinygrad-unbind.service <<EOF
[Unit]
Description=Unbind amdgpu from 7900 XTX so tinygrad AM driver can claim it
After=systemd-udev-settle.service
Wants=systemd-udev-settle.service

[Service]
Type=oneshot
# Poll up to 30s for amdgpu to actually bind, then unbind.
ExecStart=/bin/sh -c 'for i in \$(seq 1 30); do [ -L /sys/bus/pci/devices/$PCI_ID/driver ] && break; sleep 1; done; if [ -L /sys/bus/pci/devices/$PCI_ID/driver ]; then echo $PCI_ID > /sys/bus/pci/devices/$PCI_ID/driver/unbind; fi'
ExecStart=/bin/sh -c "chmod 666 /sys/bus/pci/devices/$PCI_ID/enable /sys/bus/pci/devices/$PCI_ID/resource* /sys/bus/pci/devices/$PCI_ID/config"
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF

echo "3. Writing sysctl drop-in for vm.compact_unevictable_allowed=0..."
# tinygrad's AM driver pins sysmem pages for AQL queue rings. If the
# kernel migrates an evictable-but-locked page, the AM driver's mapping
# silently points at the wrong physical page. tinygrad refuses to start
# unless this is 0. Ubuntu defaults it to 1.
cat > /etc/sysctl.d/99-tinygrad-amd.conf <<EOF
# Required for tinygrad AM driver: don't migrate pinned pages.
vm.compact_unevictable_allowed = 0
EOF
sysctl --system > /dev/null
echo "   current value: $(cat /proc/sys/vm/compact_unevictable_allowed)"

echo "4. Enabling + starting the systemd unit now..."
systemctl daemon-reload
systemctl enable tinygrad-unbind.service
systemctl start tinygrad-unbind.service

echo "5. Reloading + triggering udev rules..."
udevadm control --reload-rules
udevadm trigger

echo "6. Granting Linux capabilities to python (for PCI BAR access)..."
echo "   Target: $REAL_PYTHON"
setcap 'cap_dac_override,cap_sys_rawio,cap_sys_admin,cap_ipc_lock=ep' "$REAL_PYTHON"
echo "   Caps now:"
getcap "$REAL_PYTHON"

echo "7. Ensuring iomem=relaxed on the kernel command line..."
# With CONFIG_STRICT_DEVMEM=y, the kernel refuses mmap of PCI BAR sysfs
# resource files (even for root) unless iomem=relaxed is on cmdline.
# Idempotent: add it only if not already present.
GRUB_FILE="/etc/default/grub"
REBOOT_REQUIRED=0
if ! grep -E '^GRUB_CMDLINE_LINUX_DEFAULT=.*iomem=relaxed' "$GRUB_FILE" > /dev/null; then
  # add iomem=relaxed inside the GRUB_CMDLINE_LINUX_DEFAULT="..." value
  sed -i 's|^\(GRUB_CMDLINE_LINUX_DEFAULT="[^"]*\)"|\1 iomem=relaxed"|' "$GRUB_FILE"
  echo "   added iomem=relaxed to $GRUB_FILE"
  update-grub
  REBOOT_REQUIRED=1
else
  echo "   iomem=relaxed already present, skipping"
fi
echo "   current cmdline line:"
grep '^GRUB_CMDLINE_LINUX_DEFAULT' "$GRUB_FILE" | head -1

echo
echo "=== Done. AM driver setup complete. ==="
echo "Verify with: ls -la /sys/bus/pci/devices/$PCI_ID/enable"
if [ "$REBOOT_REQUIRED" -eq 1 ]; then
  echo
  echo "*** REBOOT REQUIRED *** to pick up iomem=relaxed."
  echo "After reboot, run: DEV='PCI+AMD' python -c '...'"
else
  echo "Run tinygrad with: DEV='PCI+AMD' python ..."
fi
