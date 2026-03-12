#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
BASE="$ROOT/src/fpga/hdl"
TB_BASE="$BASE/tb"

left=""
right=""
outdir=""
control_dir=""
width=""
height=""
overlap=""
pad="2"
mode="both"
gem_root="${GEM_ROOT:-/tmp/GEM}"
gem_num_blocks="${GEM_NUM_BLOCKS:-64}"

usage() {
  cat <<USAGE
Usage: src/fpga/hdl/run_two_image_case.sh --left <path> --right <path> [options]

Options:
  --left <path>          Left/source image path.
  --right <path>         Right/source image path.
  --control-dir <path>   Optional full or pre-extracted control directory.
  --outdir <path>        Output directory. Defaults to a temp dir.
  --width <n>            Desired extracted/input width.
  --height <n>           Desired extracted/input height.
  --overlap <n>          Desired overlap width.
  --pad <n>              Blend padding. Default: 2.
  --mode <name>          one of: iverilog, gem, both. Default: both.
  --gem-root <path>      GEM checkout root. Default: ${gem_root}
  --gem-num-blocks <n>   GEM cuda_test NUM_BLOCKS. Default: ${gem_num_blocks}
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --left)
      left="$2"
      shift 2
      ;;
    --right)
      right="$2"
      shift 2
      ;;
    --control-dir)
      control_dir="$2"
      shift 2
      ;;
    --outdir)
      outdir="$2"
      shift 2
      ;;
    --width)
      width="$2"
      shift 2
      ;;
    --height)
      height="$2"
      shift 2
      ;;
    --overlap)
      overlap="$2"
      shift 2
      ;;
    --pad)
      pad="$2"
      shift 2
      ;;
    --mode)
      mode="$2"
      shift 2
      ;;
    --gem-root)
      gem_root="$2"
      shift 2
      ;;
    --gem-num-blocks)
      gem_num_blocks="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$left" || -z "$right" ]]; then
  usage >&2
  exit 2
fi
if [[ -z "$outdir" ]]; then
  outdir="$(mktemp -d)"
fi
mkdir -p "$outdir"

if [[ "$mode" != "iverilog" && "$mode" != "gem" && "$mode" != "both" ]]; then
  echo "Invalid mode: $mode" >&2
  exit 2
fi

case_args=(
  generate
  --left "$left"
  --right "$right"
  --outdir "$outdir"
  --pad "$pad"
)
if [[ -n "$control_dir" ]]; then
  case_args+=(--control-dir "$control_dir")
fi
if [[ -n "$width" ]]; then
  case_args+=(--width "$width")
fi
if [[ -n "$height" ]]; then
  case_args+=(--height "$height")
fi
if [[ -n "$overlap" ]]; then
  case_args+=(--overlap "$overlap")
fi
python3 "$TB_BASE/gen_two_image_assets_case.py" "${case_args[@]}"

read -r input_w input_h case_overlap case_pad canvas_words <<EOF_META
$(python3 - <<'PY' "$outdir/case_manifest.json"
import json, sys
with open(sys.argv[1], 'r', encoding='ascii') as f:
    meta = json.load(f)
print(meta['input_w'], meta['input_h'], meta['overlap'], meta['pad'], meta['canvas_w'] * meta['canvas_h'])
PY
)
EOF_META

iverilog_compile() {
  iverilog -g2005-sv -o "$outdir/pano_two_image_assets.out" \
    -P "pano_two_image_assets_tb.INPUT_W=$input_w" \
    -P "pano_two_image_assets_tb.INPUT_H=$input_h" \
    -P "pano_two_image_assets_tb.OVERLAP=$case_overlap" \
    -P "pano_two_image_assets_tb.PAD=$case_pad" \
    "$BASE/pano_remap_core.v" \
    "$BASE/zybo_pano_remap_engine.v" \
    "$BASE/pano_copy_roi_engine.v" \
    "$BASE/pano_downsample2x2.v" \
    "$BASE/pano_laplacian_core.v" \
    "$BASE/pano_blend_core.v" \
    "$BASE/pano_reconstruct_core.v" \
    "$BASE/pano_two_image_assets_pipeline.v" \
    "$BASE/tb/pano_two_image_assets_tb.v"
}

run_iverilog_case() {
  iverilog_compile
  (
    cd "$outdir"
    vvp "$outdir/pano_two_image_assets.out" +dump_vcd
  )
  python3 "$TB_BASE/gen_two_image_assets_case.py" compare --outdir "$outdir"
}

run_gem_case() {
  if [[ ! -d "$gem_root" ]]; then
    echo "GEM root not found: $gem_root" >&2
    exit 1
  fi
  command -v cargo >/dev/null 2>&1 || { echo "cargo is required for GEM" >&2; exit 1; }
  command -v yosys >/dev/null 2>&1 || { echo "yosys is required for GEM" >&2; exit 1; }
  command -v nvcc >/dev/null 2>&1 || { echo "nvcc is required for GEM" >&2; exit 1; }

  cat > "$outdir/memory_synth.ys" <<EOF_MEM
read_verilog \
  $BASE/pano_remap_core.v \
  $BASE/zybo_pano_remap_engine.v \
  $BASE/pano_copy_roi_engine.v \
  $BASE/pano_downsample2x2.v \
  $BASE/pano_laplacian_core.v \
  $BASE/pano_blend_core.v \
  $BASE/pano_reconstruct_core.v \
  $BASE/pano_two_image_assets_pipeline.v
chparam -set INPUT_W $input_w -set INPUT_H $input_h -set OVERLAP $case_overlap -set PAD $case_pad -set INIT_MEMS 0 pano_two_image_assets_pipeline
hierarchy -check -top pano_two_image_assets_pipeline
proc;;
opt_expr; opt_dff; opt_clean
memory -nomap
memory_libmap -lib $gem_root/aigpdk/memlib_yosys.txt -logic-cost-rom 100 -logic-cost-ram 100
write_verilog $outdir/memory_mapped.v
EOF_MEM

  cat > "$outdir/logic_synth.ys" <<EOF_LOGIC
read_verilog $outdir/memory_mapped.v
hierarchy -check -top pano_two_image_assets_pipeline
synth -flatten
delete t:\$print
dfflibmap -liberty $gem_root/aigpdk/aigpdk_nomem.lib
opt_clean -purge
abc -liberty $gem_root/aigpdk/aigpdk_nomem.lib
opt_clean -purge
techmap
abc -liberty $gem_root/aigpdk/aigpdk_nomem.lib
opt_clean -purge
write_verilog $outdir/gatelevel.gv
EOF_LOGIC

  yosys -s "$outdir/memory_synth.ys" > "$outdir/yosys_memory.log"
  yosys -s "$outdir/logic_synth.ys" > "$outdir/yosys_logic.log"

  (
    cd "$gem_root"
    cargo build -r --features cuda --bin cut_map_interactive --bin cuda_test > "$outdir/gem_build.log" 2>&1
    cargo run -r --features cuda --bin cut_map_interactive -- "$outdir/gatelevel.gv" "$outdir/result.gemparts" > "$outdir/gem_map.log" 2>&1
    cargo run -r --features cuda --bin cuda_test -- \
      "$outdir/gatelevel.gv" \
      "$outdir/result.gemparts" \
      "$outdir/pano_two_image_assets.vcd" \
      "$outdir/output.vcd" \
      "$gem_num_blocks" \
      --input-vcd-scope pano_two_image_assets_tb/uut \
      --output-vcd-scope pano_two_image_assets_tb/uut > "$outdir/gem_sim.log" 2>&1
  )

  python3 "$TB_BASE/parse_gem_output_vcd.py" \
    --vcd "$outdir/output.vcd" \
    --out-hex "$outdir/canvas_gem.hex" \
    --expected-count "$canvas_words"
  python3 "$TB_BASE/gen_two_image_assets_case.py" compare --outdir "$outdir" --actual "$outdir/canvas_gem.hex"
}

run_iverilog_case
if [[ "$mode" == "gem" || "$mode" == "both" ]]; then
  run_gem_case
fi

printf 'Artifacts: %s\n' "$outdir"
