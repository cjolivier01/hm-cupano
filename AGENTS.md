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
- Other: `assets/` sample images, `.clang-format` style, helper scripts `bld`, `perf`.

## Build, Test, Run
- Install Bazelisk: `./scripts/install_bazelisk.sh` (Linux x86_64/aarch64).
- Debug build all: `./bld` (wraps `bazelisk build --config=debug //...`).
- Optimized build: `./perf` (then cleans).
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

