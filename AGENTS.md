# Repository Guidelines

## Project Structure & Modules
- `src/`: C++/CUDA sources
  - `cuda/`: CUDA kernels and headers
  - `pano/`: panorama logic and public library targets
  - `utils/`: OpenGL/IO helpers
  - `stable/`: stabilization code
- `tests/`: runnable sample/test binaries (Bazel `cc_binary`) and GTest targets under `src/*`.
- `scripts/`: setup and tooling (e.g., `install_bazelisk.sh`, `create_control_points.py`).
- Bazel files: `WORKSPACE`, `MODULE.bazel`, `BUILD.bazel` files under dirs.
- Other: `assets/` sample images, `.clang-format` style, root `Makefile` for common Bazel workflows.

## Inbclusiveness
- Changes in any of cudaPano, cudaPano3, or cudaPanoN (or any of its dependent code and/or kernels) should be propagated to the other two (all three) in order to keep them all in lock-step.  This includes all changes, including but not limited to, structural, mathematical, or performance and kernel fusing).


## Build, Test, Run
- Install Bazelisk: `./scripts/install_bazelisk.sh` (Linux x86_64/aarch64).
- Debug build all: `make debug` (or `make bld`).
- Optimized build: `make perf`.
- Clean: `make clean` (or `make expunge` for a full clean).
- Specific targets: `bazelisk build //src/pano:cuda_pano`.
- GTest targets: `bazelisk test //src/pano:cudaPano3_test //src/cuda:cudaBlend3_test`.
- Stitching demo (after build): `./bazel-bin/tests/test_cuda_blend --show --perf --output=out.png --directory=<data_dir>` or `./laplacian_blend.sh <data_dir>`.

## Coding Style & Conventions
- Language: C++17 (most), CUDA C++14 in kernels.
- Formatting: enforce `.clang-format` (2-space indent, 120 cols, include sorting, left-aligned `*`). Run `clang-format -i` on touched files.
- Includes: use Bazel `include_prefix` paths (e.g., `#include <cupano/pano/cudaPano.h>`).
- Naming: follow existing patterns (camelCase for files like `cudaBlendShow.cpp`, `_test.cpp` for tests, Bazel targets `snake_case`).

## Testing Guidelines
- Framework: GoogleTest for `cc_test` under `src/*`.
- Test files: name `*_test.cpp` colocated with code (e.g., `src/cuda/cudaBlend3_test.cpp`).
- Run: `bazelisk test //src/...` or execute binaries in `tests/` from `bazel-bin/tests/` with flags.
- Visual checks: include output image diffs when relevant.

## Commit & PR Guidelines
- Commits: present tense, concise scope prefix (`cuda:`, `pano:`, `utils:`), e.g., `cuda: optimize laplacian blend path`.
- PRs: describe motivation, key changes, performance impact; link issues; include commands used to build/test and sample outputs (images if visual).
- CI expectation: all Bazel builds/tests pass for both x86_64 (`--cpu=k8`) and aarch64 when applicable.

## Environment & Dependencies
- Requires NVIDIA CUDA toolkit/driver and OpenCV dev headers (`/usr/include` by default).
- External tools for config: Hugin/Enblend (`sudo apt-get install hugin hugin-tools enblend`).
- Generate control points: `python scripts/create_control_points.py <left.mp4> <right.mp4>` before running demos.

## Blend modes
- `CudaStitchPano`, `CudaStitchPano3` and `CudaStitchPanoN` take a `BlendSettings` (`src/pano/blendMode.h`) where they used to take `int num_levels`. It converts implicitly from `int`, so existing callers keep compiling and keep the convention that `0` means a hard seam. `BlendSettings::Alpha(fraction)` selects a feathered crossfade instead of the Laplacian pyramid.
- Alpha mode is level 0 of the Laplacian blend run once at full resolution: same kernels, same mask layout, no pyramid and no per-frame scratch (the N path keeps one device pointer table built at construction). The feathered weight field is built on the CPU at construction by `src/pano/featherMask.h` and mirrored in `cupano/feather.py`; with the default exact distance transform the two agree bit for bit or to one float32 ULP, verified by `scripts/compare_feather_parity.py --all` rather than by construction. OpenCV's exact distance transform is not reproducible run to run, so bit-identity is not something either side can promise.
- The weight field needs a signed distance, not `distanceTransform(label == i)` — the seam labels are a total partition, so an unsigned distance normalizes back to a hard seam. The crossfade radius is capped **per pixel** at `min{ max(2*coverage_i, 2*outside_i - 1) : coverage_i > 0 }`, the widest radius under which every camera either has the depth to survive its own taper or is too far away to reach; a global cap collapses to zero on real rigs, "covered by >= 2 cameras" is wrong for N > 2, and testing reach against the *requested* radius is circular and collapses a 400 px seam to 4 px. The field is normalized on the host, because only `BatchedBlendKernelN` normalizes unconditionally. See `docs/alpha-blend-mode.md`.

## Low-memory panorama paths
- `CudaStitchPano`, `CudaStitchPano3`, and `CudaStitchPanoN` accept opt-in `compact_workspace=false`. It reuses writable remap scratch through Gaussian, Laplacian, blend, and reconstruction lifetimes without changing precision, dimensions, pyramid levels, or arithmetic.
- Low-level blend contexts expose `reuseInputs`; callers must refill writable input storage before each call and serialize context use. Intermediate pyramids no longer retain their original meanings; diagnostic display/dump entry points reject compact mode.
- A null panorama output requests managed allocation. With equal pipeline/compute pixel types and full-canvas soft blending in compact mode, the returned `CudaMat` is a non-owning view valid until the next process call or stitcher destruction. Consume on the same stream; do not retain across reloads. Other cases return owned output.
- The separate optimized three-image low-level blend API rejects compact contexts. The panorama fused remap route uses the supported reference blender. CUDA half4 hard/fused remap instantiations support half4 panorama linkage for all camera counts.

- Packed `CudaMat<Rgb10A2>` inputs fuse RGB10 unpacking into remapping for half4 panoramas across 2/3/N cameras. Existing half4 inputs retain the staged path; three-camera `fused` keeps its original route semantics. `CudaMat::owns_memory()` identifies reusable owned output versus borrowed compact scratch. See `docs/low-memory-stitching.md`.
