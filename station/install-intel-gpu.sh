#!/bin/bash
# Install the Intel GPU drivers (Arc / Panther Lake) that PyTorch XPU needs, on Ubuntu.
# Source: https://dgpu-docs.intel.com/legacy/installation-guides/installing-packages-from-the-intel-ppa.html
# Run with: sudo bash station/install-intel-gpu.sh
set -e
USER_NAME="${SUDO_USER:-$USER}"

echo "== 1/5 package sources + tools"
apt-get update
apt-get install -y software-properties-common

echo "== 2/5 add the Intel graphics PPA"
add-apt-repository -y ppa:kobuk-team/intel-graphics
apt-get update

echo "== 3/5 compute (Level Zero + OpenCL) - needed by PyTorch XPU"
apt-get install -y libze-intel-gpu1 libze1 intel-metrics-discovery intel-opencl-icd clinfo intel-gsc

echo "== 4/5 media + AoT compiler"
apt-get install -y intel-media-va-driver-non-free libmfx-gen1.2 libvpl2 libvpl-tools libva-glx2 va-driver-all vainfo
apt-get install -y libze-dev intel-ocloc

echo "== 5/5 add user $USER_NAME to the render group"
gpasswd -a "$USER_NAME" render || true

echo
echo "== check:"
clinfo | grep "Device Name" || echo "clinfo shows no device - log out and in again, then re-check"
echo "Done."
