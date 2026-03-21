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
  local control_dir="${4:-}"
  local control_arg=""
  local outdir
  outdir="$(mktemp -d)"
  if [[ -n "$control_dir" ]]; then
    if [[ "$control_dir" = /* ]]; then
      control_arg="$control_dir"
    else
      control_arg="$ROOT/$control_dir"
    fi
  fi
  local -a gen_args=(
    python3 "$BASE/tb/gen_two_image_assets_case.py"
    generate \
    --left "$ROOT/$left_asset" \
    --right "$ROOT/$right_asset" \
    --outdir "$outdir" \
    --pad "${FPGA_PAD:-2}"
  )
  if [[ -n "${FPGA_INPUT_W:-}" ]]; then
    gen_args+=(--width "$FPGA_INPUT_W")
  fi
  if [[ -n "${FPGA_INPUT_H:-}" ]]; then
    gen_args+=(--height "$FPGA_INPUT_H")
  fi
  if [[ -n "${FPGA_OVERLAP:-}" ]]; then
    gen_args+=(--overlap "$FPGA_OVERLAP")
  fi
  if [[ -n "$control_arg" ]]; then
    gen_args+=(--control-dir "$control_arg")
  fi
  "${gen_args[@]}"
  local input_w input_h overlap pad
  input_w="$(python3 - <<'PY' "$outdir/case_manifest.json"
import json, sys
with open(sys.argv[1], 'r', encoding='ascii') as f:
    meta = json.load(f)
print(meta['input_w'])
PY
)"
  input_h="$(python3 - <<'PY' "$outdir/case_manifest.json"
import json, sys
with open(sys.argv[1], 'r', encoding='ascii') as f:
    meta = json.load(f)
print(meta['input_h'])
PY
)"
  overlap="$(python3 - <<'PY' "$outdir/case_manifest.json"
import json, sys
with open(sys.argv[1], 'r', encoding='ascii') as f:
    meta = json.load(f)
print(meta['overlap'])
PY
)"
  pad="$(python3 - <<'PY' "$outdir/case_manifest.json"
import json, sys
with open(sys.argv[1], 'r', encoding='ascii') as f:
    meta = json.load(f)
print(meta['pad'])
PY
)"
  iverilog -g2005-sv -o "$outdir/pano_two_image_assets.out" \
    -P "pano_two_image_assets_tb.INPUT_W=$input_w" \
    -P "pano_two_image_assets_tb.INPUT_H=$input_h" \
    -P "pano_two_image_assets_tb.OVERLAP=$overlap" \
    -P "pano_two_image_assets_tb.PAD=$pad" \
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

  local artifact_root="${FPGA_ARTIFACT_DIR:-${TEST_UNDECLARED_OUTPUTS_DIR:-}}"
  if [[ -n "$artifact_root" ]]; then
    mkdir -p "$artifact_root/$case_name"
    cp -R "$outdir"/. "$artifact_root/$case_name/"
    echo "Wrote artifacts: $artifact_root/$case_name"
  fi

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
LEFT_RIGHT_CONTROL_DIR="${FPGA_CONTROL_DIR_LEFT_RIGHT:-}"
S1_S2_CONTROL_DIR="${FPGA_CONTROL_DIR_S1_S2:-}"
run_tb pano_downsample2x2_tb \
  "$BASE/pano_downsample2x2.v" \
  "$BASE/tb/pano_downsample2x2_tb.v"
run_tb pano_blend_core_tb \
  "$BASE/pano_blend_core.v" \
  "$BASE/tb/pano_blend_core_tb.v"
run_tb pano_reconstruct_core_tb \
  "$BASE/pano_reconstruct_core.v" \
  "$BASE/tb/pano_reconstruct_core_tb.v"
run_assets_pipeline_tb left_right assets/left.png assets/right.png "$LEFT_RIGHT_CONTROL_DIR"
run_assets_pipeline_tb s1_s2 assets/s1.jpg assets/s2.jpg "$S1_S2_CONTROL_DIR"
