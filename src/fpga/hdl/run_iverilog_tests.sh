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
  local case_name="$1"
  local left_asset="$2"
  local right_asset="$3"
  local outdir
  outdir="$(mktemp -d)"
  python3 "$BASE/tb/gen_two_image_assets_case.py" \
    generate \
    --left "$ROOT/$left_asset" \
    --right "$ROOT/$right_asset" \
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

  local -a vvp_args=()
  if [[ "${FPGA_DUMP_VCD:-0}" == "1" ]]; then
    vvp_args+=("+dump_vcd")
  fi
  (
    cd "$outdir"
    vvp "$outdir/pano_two_image_assets.out" "${vvp_args[@]}"
  )
  python3 "$BASE/tb/gen_two_image_assets_case.py" compare --outdir "$outdir"

  if [[ "${FPGA_DUMP_VCD:-0}" == "1" && -f "$outdir/pano_two_image_assets.vcd" ]]; then
    local vcd_root="${FPGA_VCD_DIR:-${TEST_UNDECLARED_OUTPUTS_DIR:-}}"
    if [[ -n "$vcd_root" ]]; then
      mkdir -p "$vcd_root"
      cp "$outdir/pano_two_image_assets.vcd" "$vcd_root/${case_name}.vcd"
      echo "Wrote VCD: $vcd_root/${case_name}.vcd"
    else
      echo "Wrote VCD: $outdir/pano_two_image_assets.vcd"
    fi
  fi
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
run_assets_pipeline_tb left_right assets/left.png assets/right.png
run_assets_pipeline_tb s1_s2 assets/s1.jpg assets/s2.jpg
