# Laplacian kernel occupancy

The reference two-, three-, and N-camera blend paths use 32-by-4 thread blocks.
Each warp follows one image row, and the 128-thread blocks permit finer register
allocation than the previous 16-by-16 blocks. The separate shared-memory tiled
three-camera implementation keeps its own launch geometry.

Reconstruction accumulates the alpha-valid interpolation weight once per pixel,
then reconstructs the color channels and copies alpha afterward. This preserves
the addition order and division, while shortening the live state across the
existing device-side alpha diagnostic. Standard and compact workspaces retain
the same allocation sizes, pixel formats, pyramid levels and ownership rules.

On SM 12.0, reconstruction uses `__launch_bounds__(128, 12)` to give the compiler
the register budget for twelve resident blocks (1536 threads). Other architectures
keep the compiler's default register budget. With CUDA 13.3 on an RTX 5090, the
two-camera FP16 reconstruction uses 40 registers per thread instead of 56. Its
reported local storage remains zero and its diagnostic stack remains 8 bytes.
This removes the register limit on theoretical occupancy for that kernel; it
does not guarantee 100% achieved occupancy or whole-program SM activity.

In hstream's `gse-16a` Program path, the 14,468-by-4,190 panorama already supplies
ample blocks without temporal batching. Matching native Blackwell builds gave:

| Measurement | Previous | Updated |
| --- | ---: | ---: |
| Completed Program FPS, two unprofiled runs | 90.05 / 90.60 | 93.72 / 94.77 |
| Steady-state resident compute warps | 72.58% | 78.62% |
| Steady-state SM activity | 96.43% | 96.25% |
| Full-resolution reconstruction | 981.9 us | 906.4 us |
| Full-resolution Laplacian kernel | 960.5 us | 871.4 us |

GPU metrics come from Nsight Systems over the first-to-last output conversion
window, excluding the first three and last one seconds. They are device-wide;
runs with unrelated GPU computations were excluded. The throughput runs use a
warmed FP16 detector, the same calibration and a FAKE sink after Program
processing, audio and telemetry. See hstream's `docs/gpu-throughput.md` for
end-to-end reproduction and archive/low-memory qualification.

`cudaBlend_test.cpp` covers odd tile boundaries and sub-tile images, two-frame
batches, RGB/RGBA, float/half, and standard/compact workspaces across the three
reference implementations. Poisoned outputs are checked against known pixels
so a shared launch-grid mistake cannot pass by comparing two incorrect outputs.
