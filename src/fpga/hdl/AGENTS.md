# HDL Guidelines

## Scope
- Applies to `src/fpga/hdl/`.

## Style
- Use Verilog-2001 compatible syntax unless a file has a clear reason to require SystemVerilog.
- Keep modules single-purpose. Put DDR/AXI transport in separate modules from image arithmetic.
- Prefer parameters for data width, mask width, and channel count.
- Use synchronous resets unless the surrounding shell already requires an async reset.

## Interfaces
- Keep AXI-facing modules thin and descriptor-driven.
- Arithmetic cores should use simple request/response or streaming handshakes so they can be reused behind DMA, VDMA, or custom memory schedulers.
- Name clock/reset ports `clk` and `rstn` consistently.

## Arithmetic
- Default internal pixel representation is signed 18-bit `Q9.8`.
- Default mask representation is unsigned 16-bit `U0.16`.
- Saturate explicitly on reconstruct and Laplacian paths; do not rely on truncation.
- Preserve alpha as coverage/validity, not as another blend weight.

## Verification
- Every new arithmetic core should get a dedicated self-checking testbench when practical.
- Keep testbenches deterministic and avoid waveform-only validation.
