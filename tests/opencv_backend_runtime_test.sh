#!/usr/bin/env bash
set -euo pipefail

backend="${1:-}"
bin_path="${2:-}"

if [[ -z "$backend" || -z "$bin_path" ]]; then
  printf 'usage: %s <cuda|rocm|vulkan> <binary>\n' "$0" >&2
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
    printf 'unexpected CUDA OpenCV library linked in non-CUDA mode: %s\n' "$needle" >&2
    exit 1
  fi
}

count_linked_cuda_opencv_libs() {
  local count=0
  grep -q -- 'libopencv_cudacodec.so' <<<"$ldd_out" && count=$((count + 1))
  grep -q -- 'libopencv_cudaimgproc.so' <<<"$ldd_out" && count=$((count + 1))
  grep -q -- 'libopencv_cudawarping.so' <<<"$ldd_out" && count=$((count + 1))
  printf '%s' "$count"
}

case "$backend" in
  cuda)
    linked_count="$(count_linked_cuda_opencv_libs)"
    if [[ "$linked_count" -eq 3 ]]; then
      exit 0
    fi
    if [[ "$linked_count" -eq 0 ]]; then
      printf 'CUDA backend built without OpenCV CUDA video modules; CPU OpenCV I/O fallback is active.\n'
      exit 0
    fi
    printf 'inconsistent OpenCV CUDA linkage for CUDA backend (%s/3 CUDA OpenCV libs linked)\n' "$linked_count" >&2
    exit 1
    ;;
  rocm)
    check_absent 'libopencv_cudacodec.so'
    check_absent 'libopencv_cudaimgproc.so'
    check_absent 'libopencv_cudawarping.so'
    ;;
  vulkan)
    check_absent 'libopencv_cudacodec.so'
    check_absent 'libopencv_cudaimgproc.so'
    check_absent 'libopencv_cudawarping.so'
    ;;
  *)
    printf 'invalid backend: %s\n' "$backend" >&2
    exit 1
    ;;
esac
