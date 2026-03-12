#!/usr/bin/env bash
set -euo pipefail

if ! command -v iverilog >/dev/null 2>&1 || ! command -v vvp >/dev/null 2>&1; then
  echo "Skipping HDL simulation: iverilog/vvp not installed"
  exit 0
fi

ROOT="${TEST_SRCDIR:-$(pwd)}/${TEST_WORKSPACE:-}"
if [[ -z "${TEST_SRCDIR:-}" ]]; then
  ROOT="$(pwd)"
fi

run_tb() {
  local name="$1"
  shift
  local outdir
  outdir="$(mktemp -d)"
  iverilog -g2005-sv -o "$outdir/$name.out" "$@"
  vvp "$outdir/$name.out"
}

run_assets_pipeline_tb() {
  local outdir
  outdir="$(mktemp -d)"
  python3 "$BASE/tb/gen_two_image_assets_case.py" \
    generate \
    --left "$ROOT/assets/left.png" \
    --right "$ROOT/assets/right.png" \
    --outdir "$outdir"
  iverilog -g2005-sv -o "$outdir/pano_two_image_assets.out" \
    "$BASE/pano_remap_core.v" \
    "$BASE/zybo_pano_remap_engine.v" \
    "$BASE/pano_copy_roi_engine.v" \
    "$BASE/pano_downsample2x2.v" \
    "$BASE/pano_laplacian_core.v" \
    "$BASE/pano_blend_core.v" \
    "$BASE/pano_reconstruct_core.v" \
    "$BASE/pano_two_image_assets_pipeline.v" \
    "$BASE/tb/pano_two_image_assets_tb.v"
  (
    cd "$outdir"
    vvp "$outdir/pano_two_image_assets.out"
  )
  python3 "$BASE/tb/gen_two_image_assets_case.py" compare --outdir "$outdir"
}

BASE="$ROOT/src/fpga/hdl"
run_tb pano_downsample2x2_tb \
  "$BASE/pano_downsample2x2.v" \
  "$BASE/tb/pano_downsample2x2_tb.v"
run_tb pano_blend_core_tb \
  "$BASE/pano_blend_core.v" \
  "$BASE/tb/pano_blend_core_tb.v"
run_tb pano_reconstruct_core_tb \
  "$BASE/pano_reconstruct_core.v" \
  "$BASE/tb/pano_reconstruct_core_tb.v"
run_assets_pipeline_tb
