# Sourced by launch.sh before running the SYCL build.
# Sets up Intel oneAPI environment + B70-specific safety flags.
#
# IMPORTANT: oneAPI setvars.sh is not strict-mode-clean. We disable
# `set -e` and `set -u` while sourcing it, then re-enable. Without this,
# the launcher exits silently with code 127 right before exec.

set +eu
if [[ -f /opt/intel/oneapi/setvars.sh ]]; then
  source /opt/intel/oneapi/setvars.sh --force > /dev/null 2>&1
elif [[ -f "$HOME/intel/oneapi/setvars.sh" ]]; then
  source "$HOME/intel/oneapi/setvars.sh" --force > /dev/null 2>&1
else
  echo "warning: oneAPI setvars.sh not found; SYCL build may fail to load." >&2
fi
set -eu

# Issue #21893 corruption guard — was mandatory mid-2025, verified fixed
# in llama.cpp commit bbeb89d (2026-05-05) on B70 silicon. Leave OFF.
# If you see garbled chat output after a future llama.cpp pull, uncomment.
# export GGML_SYCL_DISABLE_OPT=1

# Pin to Level Zero device 0 (the B70). Avoids the SYCL runtime picking
# the AMD iGPU's OpenCL or CPU OpenCL ICD.
export ONEAPI_DEVICE_SELECTOR=level_zero:0
