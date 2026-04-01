#!/usr/bin/env bash
set -euo pipefail

require_cmd() {
  local tool="$1"
  if ! command -v "$tool" >/dev/null 2>&1; then
    printf 'missing required command: %s\n' "$tool" >&2
    exit 1
  fi
}

ffmpeg_bin="${1:-}"
if [[ -z "$ffmpeg_bin" ]]; then
  ffmpeg_bin="$(command -v ffmpeg || true)"
fi

if [[ -z "$ffmpeg_bin" || ! -x "$ffmpeg_bin" ]]; then
  printf 'missing required ffmpeg binary path\n' >&2
  exit 1
fi

require_cmd nvidia-smi

if ! nvidia-smi --query-gpu=name --format=csv,noheader >/dev/null 2>&1; then
  printf 'nvidia-smi cannot query GPUs\n' >&2
  exit 1
fi

buildconf="$($ffmpeg_bin -hide_banner -buildconf 2>/dev/null || true)"
for required in \
  --enable-cuda-nvcc \
  --enable-cuvid \
  --enable-nvdec \
  --enable-nvenc \
  --enable-libx264 \
  --enable-encoder=h264_nvenc \
  --enable-decoder=h264_cuvid; do
  if ! grep -q -- "$required" <<<"$buildconf"; then
    printf 'ffmpeg build config is missing required flag: %s\n' "$required" >&2
    exit 1
  fi
done

if ! "$ffmpeg_bin" -hide_banner -hwaccels | awk '{print $1}' | grep -qx 'cuda'; then
  printf 'ffmpeg does not report CUDA hwaccel support\n' >&2
  exit 1
fi

if ! "$ffmpeg_bin" -hide_banner -decoders | awk '{print $2}' | grep -Eq '^(h264_cuvid|hevc_cuvid|av1_cuvid)$'; then
  printf 'ffmpeg does not report NVDEC (cuvid) decoders\n' >&2
  exit 1
fi

if ! "$ffmpeg_bin" -hide_banner -encoders | awk '{print $2}' | grep -Eq '^(h264_nvenc|hevc_nvenc|av1_nvenc)$'; then
  printf 'ffmpeg does not report NVENC encoders\n' >&2
  exit 1
fi

tmp_root="${TEST_TMPDIR:-/tmp}"
workdir="${tmp_root}/ffmpeg_cuda_runtime_${RANDOM}_$$"
mkdir -p "$workdir"
trap 'rm -rf "$workdir"' EXIT

input_mp4="${workdir}/input.mp4"

# NVENC smoke test: synthesize a clip and encode on GPU.
"$ffmpeg_bin" -hide_banner -loglevel error -y \
  -f lavfi -i "testsrc2=size=640x360:rate=30" \
  -frames:v 30 \
  -pix_fmt yuv420p \
  -c:v h264_nvenc \
  "$input_mp4"

# NVDEC smoke test: decode that clip on GPU and drain output.
"$ffmpeg_bin" -hide_banner -loglevel error -y \
  -hwaccel cuda \
  -c:v h264_cuvid \
  -i "$input_mp4" \
  -f null -
