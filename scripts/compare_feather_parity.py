#!/usr/bin/env python3
"""Compares the C++ feather field against cupano.feather.build_weights element by element.

The parity claim in AGENTS.md and docs/alpha-blend-mode.md rests on this comparison. Run it as:

    bazel run //src/pano:featherMaskDump -- two_camera /tmp/two_camera.bin
    python3 scripts/compare_feather_parity.py two_camera /tmp/two_camera.bin

or over every fixture at once:

    python3 scripts/compare_feather_parity.py --all

Expect bit-identical weights with the exact distance transform. OpenCV's DIST_MASK_PRECISE is not
reproducible run to run on every build, so a one-ULP disagreement on a small fraction of elements
is possible and is reported separately from a real divergence.
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from cupano.feather import FeatherParams, build_weights  # noqa: E402

UNMAPPED = 65535


def _mapped(w: int, h: int) -> np.ndarray:
    return np.zeros((h, w), np.uint16)


def _unmapped(w: int, h: int) -> np.ndarray:
    return np.full((h, w), UNMAPPED, np.uint16)


def fixture(name: str):
    """Mirrors src/pano/featherMaskDump.cpp; keep the two in lockstep."""
    if name == "two_camera":
        labels = np.zeros((200, 600), np.uint8)
        labels[:, 250:] = 1
        mx = [_mapped(400, 200), _mapped(450, 200)]
        return labels, mx, [(0, 0), (150, 0)], 0.5
    if name == "empty_label_region":
        # Camera 2 covers the canvas but owns no label, so its weight is zero everywhere. Kept
        # because it is the only rig here with an empty label region, which is what carries the
        # saturated distance transform across the language boundary.
        labels = np.zeros((200, 600), np.uint8)
        labels[:, 250:] = 1
        mx = [_mapped(400, 200), _mapped(450, 200), _mapped(600, 200)]
        return labels, mx, [(0, 0), (150, 0), (0, 0)], 0.5
    if name == "blanketing_third":
        # Camera 2 covers the canvas and owns a sliver, so it does reach the cap.
        labels = np.zeros((200, 600), np.uint8)
        labels[:, 250:] = 1
        labels[:, 290:310] = 2
        mx = [_mapped(400, 200), _mapped(450, 200), _mapped(600, 200)]
        return labels, mx, [(0, 0), (150, 0), (0, 0)], 0.5
    if name == "grazing_neighbour":
        labels = np.zeros((400, 600), np.uint8)
        labels[:, 250:] = 1
        labels[198:203, 310:330] = 2
        sliver = _unmapped(400, 400)
        sliver[198:203, 0:134] = 0
        mx = [_mapped(400, 400), _mapped(400, 400), sliver]
        return labels, mx, [(0, 0), (150, 0), (198, 0)], 1.0
    if name == "coverage_hole":
        labels = np.zeros((300, 500), np.uint8)
        labels[:, 250:] = 1
        holed = _mapped(500, 300)
        holed[60:150, 120:210] = UNMAPPED
        mx = [holed, _mapped(500, 300)]
        return labels, mx, [(0, 0), (0, 0)], 0.2
    if name == "hard_fallback":
        labels = np.zeros((40, 128), np.uint8)
        labels[:, 60:] = 1
        holed = _mapped(128, 40)
        holed[8:28, 20:50] = UNMAPPED
        mx = [holed, _mapped(128, 40)]
        return labels, mx, [(0, 0), (0, 0)], 0.0
    if name == "eight_camera":
        w, h, n, stride = 300, 200, 8, 150
        canvas_w = w + stride * (n - 1)
        labels = np.repeat(
            np.minimum(n - 1, np.arange(canvas_w) * n // canvas_w).astype(np.uint8)[None], h, axis=0
        )
        mx = [_mapped(w, h) for _ in range(n)]
        return labels, mx, [(stride * i, 0) for i in range(n)], 0.15
    raise SystemExit(f"unknown fixture: {name}")


FIXTURES = (
    "two_camera",
    "empty_label_region",
    "blanketing_third",
    "grazing_neighbour",
    "coverage_hole",
    "hard_fallback",
    "eight_camera",
)


def compare(name: str, dump_path: pathlib.Path) -> bool:
    raw = dump_path.read_bytes()
    rows, cols, channels = np.frombuffer(raw, np.int32, 3)
    count = int(rows) * int(cols) * int(channels)
    cpp_weights = np.frombuffer(raw, np.float32, count, 12).reshape(rows, cols, channels)
    label_offset = 12 + 4 * count
    cpp_labels = np.frombuffer(raw, np.uint8, rows * cols, label_offset).reshape(rows, cols)
    cpp_meta = np.frombuffer(raw, np.float32, 5, label_offset + rows * cols)

    labels, mx, positions, frac = fixture(name)
    r = build_weights(labels, mx, [m.copy() for m in mx], positions, len(mx), FeatherParams(fraction=frac))
    py_meta = np.array(
        [
            r.radius_px,
            r.min_seam_radius_px,
            r.requested_radius_px,
            r.capped_seam_fraction,
            float(r.overlap_capped),
        ],
        np.float32,
    )

    differing = int((cpp_weights != r.weights).sum())
    max_abs = float(np.abs(cpp_weights - r.weights).max()) if differing else 0.0
    ulp_only = max_abs <= 2e-7
    labels_match = np.array_equal(cpp_labels, r.corrected_labels)
    # capped_seam_fraction is float in C++ and float64 in Python, so it can never match exactly.
    meta_match = np.allclose(cpp_meta, py_meta, rtol=0, atol=1e-6)

    status = "bit-identical" if differing == 0 else ("1 ULP" if ulp_only else "DIVERGED")
    print(
        f"{name:20s} {cpp_weights.size:>9d} elements  {status:14s} "
        f"differing={differing:<8d} maxabs={max_abs:.3e}  "
        f"labels={'ok' if labels_match else 'MISMATCH'}  meta={'ok' if meta_match else 'MISMATCH'}"
    )
    return ulp_only and labels_match and meta_match


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", nargs="?", help="fixture name, or omit with --all")
    parser.add_argument("dump", nargs="?", type=pathlib.Path, help="binary written by featherMaskDump")
    parser.add_argument("--all", action="store_true", help="build and compare every fixture")
    args = parser.parse_args()

    if args.all:
        repo = pathlib.Path(__file__).resolve().parent.parent
        subprocess.run(
            ["bazelisk", "build", "--config=opt", "//src/pano:featherMaskDump"], cwd=repo, check=True
        )
        binary = repo / "bazel-bin/src/pano/featherMaskDump"
        ok = True
        with tempfile.TemporaryDirectory() as tmp:
            for name in FIXTURES:
                out = pathlib.Path(tmp) / f"{name}.bin"
                subprocess.run([str(binary), name, str(out)], check=True)
                ok &= compare(name, out)
        print("\nparity holds" if ok else "\nPARITY BROKEN")
        return 0 if ok else 1

    if not args.fixture or not args.dump:
        parser.error("give a fixture and a dump path, or --all")
    return 0 if compare(args.fixture, args.dump) else 1


if __name__ == "__main__":
    raise SystemExit(main())
