# FPGA Subtree Guidelines

## Scope
- Applies to `src/fpga/` and all descendants unless a deeper `AGENTS.md` overrides it.
- Keep ARM-side orchestration here. Do not mix new FPGA runtime code into `src/pano/` unless the interface must be shared broadly.

## Architecture
- Treat the ARM processor as the owner of control, buffer allocation, and operation scheduling.
- Treat the FPGA as the owner of datapath-heavy kernels only: remap, ROI copy/blit, pyramid downsample, Laplacian, blend, reconstruct, and memory movers.
- Prefer explicit descriptors and register maps over implicit state.

## C++ Rules
- Use `cupano/fpga/...` include paths.
- Keep hardware-independent planning code separate from MMIO or platform-specific code.
- Prefer testable interfaces for operation execution and physical-buffer allocation so unit tests can run without hardware.
- If a type represents hardware state, include units in field names where possible, such as `_bytes`, `_bits`, `_ms`, `_addr`.

## Zybo/Zynq Rules
- Assume Zybo Z7 DDR is accessed from PL through PS HP ports, usually via AXI DMA/VDMA or a custom AXI master.
- Do not assume cache coherency between ARM and PL. Make cache-management requirements explicit in comments or APIs.
- Prefer 64-bit physical addresses in software even if the deployed design only decodes 32 bits.

## Fixed Point
- Default remap input/output stays `RGB888` or `RGBA8888` at the frame boundary.
- Default pyramid compute format is signed `Q9.8` in 18 bits unless a file documents a better reason.
- Default blend-mask weights are unsigned `U0.16`.
- Exposure/color adjustments use signed `Q8.8`.

## Testing
- Add focused unit tests for planning, descriptor packing, and orchestration.
- Add HDL testbenches for arithmetic/datapath modules when behavior changes.
- If the environment lacks `iverilog`, simulation runners should fail clearly or skip cleanly with a direct message.
