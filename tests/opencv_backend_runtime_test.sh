#!/usr/bin/env bash
set -euo pipefail

backend="${1:-}"
bin_path="${2:-}"

if [[ -z "$backend" || -z "$bin_path" ]]; then
  printf 'usage: %s <cuda|rocm> <binary>\n' "$0" >&2
  exit 1
fi

if ! command -v ldd >/dev/null 2>&1; then
  printf 'missing required command: ldd\n' >&2
  exit 1
fi

if [[ ! -x "$bin_path" ]]; then
  printf 'missing executable binary: %s\n' "$bin_path" >&2
  exit 1
fi

ldd_out="$(ldd "$bin_path" 2>/dev/null || true)"
if [[ -z "$ldd_out" ]]; then
  printf 'ldd produced no output for %s\n' "$bin_path" >&2
  exit 1
fi

check_has() {
  local needle="$1"
  if ! grep -q -- "$needle" <<<"$ldd_out"; then
    printf 'expected linked OpenCV CUDA library not found: %s\n' "$needle" >&2
    exit 1
  fi
}

check_absent() {
  local needle="$1"
  if grep -q -- "$needle" <<<"$ldd_out"; then
    printf 'unexpected CUDA OpenCV library linked in ROCm mode: %s\n' "$needle" >&2
    exit 1
  fi
}

case "$backend" in
  cuda)
    check_has 'libopencv_cudacodec.so'
    check_has 'libopencv_cudaimgproc.so'
    check_has 'libopencv_cudawarping.so'
    ;;
  rocm)
    check_absent 'libopencv_cudacodec.so'
    check_absent 'libopencv_cudaimgproc.so'
    check_absent 'libopencv_cudawarping.so'
    ;;
  *)
    printf 'invalid backend: %s\n' "$backend" >&2
    exit 1
    ;;
esac
