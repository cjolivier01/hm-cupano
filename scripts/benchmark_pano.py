#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
import tifffile

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cupano import ControlMasks, CudaStitchPano
from cupano.triton_ops import triton_available


@dataclass
class BenchmarkResult:
    width: int
    height: int
    levels: int
    engine: str
    fps: float | None
    ms_per_frame: float | None
    ok: bool
    note: str = ""


def run_capture(cmd: list[str], cwd: Path | None = None) -> str:
    print("EXEC:", " ".join(str(part) for part in cmd))
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        check=False,
        text=True,
        capture_output=True,
    )
    output = proc.stdout + proc.stderr
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}\n{output}")
    return output


def ensure_binary(binary: Path, target: str, build_if_needed: bool) -> Path:
    if binary.exists():
        return binary
    if not build_if_needed:
        raise FileNotFoundError(f"Missing binary: {binary}")
    bazelisk = shutil.which("bazelisk") or shutil.which("bazel")
    if not bazelisk:
        raise FileNotFoundError("bazelisk/bazel not found; cannot build benchmark binary")
    run_capture([bazelisk, "build", target], cwd=REPO_ROOT)
    if not binary.exists():
        raise FileNotFoundError(f"Binary still missing after build: {binary}")
    return binary


def parse_size(value: str) -> tuple[int, int]:
    match = re.fullmatch(r"(\d+)x(\d+)", value)
    if not match:
        raise argparse.ArgumentTypeError(f"Invalid size '{value}', expected WIDTHxHEIGHT")
    return int(match.group(1)), int(match.group(2))


def identity_map_x(width: int, height: int) -> np.ndarray:
    return np.tile(np.arange(width, dtype=np.uint16), (height, 1))


def identity_map_y(width: int, height: int) -> np.ndarray:
    return np.tile(np.arange(height, dtype=np.uint16)[:, None], (1, width))


def patterned_bgr(width: int, height: int, seed: int) -> np.ndarray:
    xs = np.tile(np.arange(width, dtype=np.uint16), (height, 1))
    ys = np.tile(np.arange(height, dtype=np.uint16)[:, None], (1, width))
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[..., 0] = ((seed * 17) + xs) % 251
    image[..., 1] = ((seed * 29) + ys) % 253
    image[..., 2] = ((seed * 37) + xs + ys) % 255
    return image


def write_position_tiff(path: Path, width: int, height: int, xpos: int, ypos: int) -> None:
    tifffile.imwrite(
        path,
        np.zeros((height, width), dtype=np.uint8),
        resolution=(1.0, 1.0),
        extratags=[
            (286, 5, 1, (xpos, 1), False),
            (287, 5, 1, (ypos, 1), False),
        ],
    )


def prepare_benchmark_directory(root: Path, width: int, height: int, overlap: int) -> Path:
    if overlap <= 0 or overlap >= width:
        raise ValueError(f"Invalid overlap={overlap} for width={width}")
    x2 = width - overlap
    if x2 < 128:
        raise ValueError(f"Benchmark overlap implies x2={x2}, but two-image blend requires x2 >= 128")

    directory = root / f"w{width}_h{height}_ov{overlap}"
    directory.mkdir(parents=True, exist_ok=True)

    left = patterned_bgr(width, height, 0)
    right = patterned_bgr(width, height, 1)
    cv2.imwrite(str(directory / "left.png"), left)
    cv2.imwrite(str(directory / "right.png"), right)

    map_x = identity_map_x(width, height)
    map_y = identity_map_y(width, height)
    cv2.imwrite(str(directory / "mapping_0000_x.tif"), map_x)
    cv2.imwrite(str(directory / "mapping_0000_y.tif"), map_y)
    cv2.imwrite(str(directory / "mapping_0001_x.tif"), map_x)
    cv2.imwrite(str(directory / "mapping_0001_y.tif"), map_y)

    write_position_tiff(directory / "mapping_0000.tif", width, height, 0, 0)
    write_position_tiff(directory / "mapping_0001.tif", width, height, x2, 0)

    canvas_width = width + x2
    seam = np.zeros((height, canvas_width), dtype=np.uint8)
    boundary = x2 + overlap // 2
    seam[:, :boundary] = 255
    cv2.imwrite(str(directory / "seam_file.png"), seam)
    return directory


def load_python_inputs(directory: Path, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    left = cv2.imread(str(directory / "left.png"), cv2.IMREAD_COLOR)
    right = cv2.imread(str(directory / "right.png"), cv2.IMREAD_COLOR)
    if left is None or right is None:
        raise FileNotFoundError(f"Missing benchmark inputs in {directory}")
    left_bgra = cv2.cvtColor(left, cv2.COLOR_BGR2BGRA)
    right_bgra = cv2.cvtColor(right, cv2.COLOR_BGR2BGRA)
    left_tensor = torch.from_numpy(np.ascontiguousarray(left_bgra)).to(device=device, dtype=torch.uint8).unsqueeze(0)
    right_tensor = torch.from_numpy(np.ascontiguousarray(right_bgra)).to(device=device, dtype=torch.uint8).unsqueeze(0)
    return left_tensor, right_tensor


def benchmark_python(
    directory: Path,
    levels: int,
    backend: str,
    device: torch.device,
    iterations: int,
    warmup: int,
) -> BenchmarkResult:
    image = cv2.imread(str(directory / "left.png"), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(directory / "left.png")
    width = image.shape[1]
    height = image.shape[0]
    try:
        if backend == "triton" and not triton_available():
            return BenchmarkResult(width, height, levels, "python-triton", None, None, False, "Triton not installed")
        left, right = load_python_inputs(directory, device)
        masks = ControlMasks(str(directory))
        if not masks.is_valid():
            raise RuntimeError(f"Failed to load control masks from {directory}")
        stitcher = CudaStitchPano(
            batch_size=1,
            num_levels=levels,
            control_masks=masks,
            match_exposure=False,
            quiet=True,
            backend=backend,
        )

        with torch.inference_mode():
            for _ in range(max(warmup, 1)):
                _ = stitcher.process(left, right)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            for _ in range(iterations):
                _ = stitcher.process(left, right)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        fps = iterations / elapsed
        return BenchmarkResult(width, height, levels, f"python-{backend}", fps, 1000.0 / fps, True)
    except Exception as exc:  # pragma: no cover - exercised through command execution
        return BenchmarkResult(width, height, levels, f"python-{backend}", None, None, False, str(exc))
    finally:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()


def benchmark_cpp(binary: Path, directory: Path, levels: int) -> BenchmarkResult:
    image = cv2.imread(str(directory / "left.png"), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(directory / "left.png")
    width = image.shape[1]
    height = image.shape[0]
    try:
        output = run_capture(
            [
                str(binary),
                "--perf",
                f"--directory={directory}",
                f"--levels={levels}",
                "--adjust=0",
                "--batch-size=1",
            ],
            cwd=REPO_ROOT,
        )
        match = re.search(r"Blend speed:\s*([0-9.]+)fps", output)
        if not match:
            raise RuntimeError(f"Unable to parse C++ benchmark output:\n{output}")
        fps = float(match.group(1))
        return BenchmarkResult(width, height, levels, "cpp", fps, 1000.0 / fps, True)
    except Exception as exc:  # pragma: no cover - exercised through command execution
        return BenchmarkResult(width, height, levels, "cpp", None, None, False, str(exc))


def summarize_rows(results: Iterable[BenchmarkResult]) -> list[dict[str, object]]:
    grouped: dict[tuple[int, int, int], dict[str, object]] = {}
    for result in results:
        key = (result.width, result.height, result.levels)
        row = grouped.setdefault(
            key,
            {
                "width": result.width,
                "height": result.height,
                "levels": result.levels,
            },
        )
        row[f"{result.engine}_fps"] = result.fps
        row[f"{result.engine}_ms"] = result.ms_per_frame
        if not result.ok:
            row[f"{result.engine}_note"] = result.note
    for row in grouped.values():
        cpp_fps = row.get("cpp_fps")
        for engine in ("python-torch", "python-triton"):
            py_fps = row.get(f"{engine}_fps")
            key = f"cpp_over_{engine.replace('-', '_')}"
            row[key] = (cpp_fps / py_fps) if isinstance(cpp_fps, float) and isinstance(py_fps, float) and py_fps > 0 else None
    return sorted(grouped.values(), key=lambda item: (int(item["width"]), int(item["height"]), int(item["levels"])))


def format_value(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def print_table(rows: list[dict[str, object]]) -> None:
    columns = [
        ("size", lambda row: f"{row['width']}x{row['height']}"),
        ("levels", lambda row: row["levels"]),
        ("cpp_fps", lambda row: row.get("cpp_fps")),
        ("py_torch_fps", lambda row: row.get("python-torch_fps")),
        ("py_triton_fps", lambda row: row.get("python-triton_fps")),
        ("cpp/py_torch", lambda row: row.get("cpp_over_python_torch")),
        ("cpp/py_triton", lambda row: row.get("cpp_over_python_triton")),
    ]
    widths = []
    for name, getter in columns:
        values = [format_value(getter(row)) for row in rows]
        widths.append(max(len(name), *(len(value) for value in values)))
    header = "  ".join(name.ljust(width) for (name, _), width in zip(columns, widths, strict=True))
    print(header)
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print(
            "  ".join(
                format_value(getter(row)).ljust(width)
                for (_, getter), width in zip(columns, widths, strict=True)
            )
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the two-image C++ pano binary against Python torch/Triton backends")
    parser.add_argument("--sizes", nargs="+", type=parse_size, default=[(512, 256), (1024, 512), (2048, 1024)], help="Image sizes as WIDTHxHEIGHT")
    parser.add_argument("--levels", nargs="+", type=int, default=[0, 1, 6], help="Blend levels to benchmark")
    parser.add_argument("--python-backends", nargs="+", choices=("torch", "triton"), default=["torch", "triton"], help="Python backends to benchmark")
    parser.add_argument("--iterations", type=int, default=100, help="Python iterations per benchmark point")
    parser.add_argument("--warmup", type=int, default=10, help="Python warmup iterations per benchmark point")
    parser.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"), help="Torch device for the Python benchmark")
    parser.add_argument("--build-if-needed", action="store_true", help="Build the C++ binary if it is missing")
    parser.add_argument("--binary", default=None, help="Path to the C++ two-image benchmark binary")
    parser.add_argument("--output-json", default=None, help="Optional JSON file for the benchmark summary")
    parser.add_argument("--work-dir", default=None, help="Directory for generated synthetic benchmark inputs")
    parser.add_argument("--overlap-fraction", type=float, default=0.5, help="Fractional overlap between the two inputs")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    work_dir = Path(args.work_dir).resolve() if args.work_dir else (REPO_ROOT / ".benchmarks" / "pano")
    work_dir.mkdir(parents=True, exist_ok=True)

    binary = Path(args.binary).resolve() if args.binary else REPO_ROOT / "bazel-bin" / "tests" / "test_cuda_blend"
    binary = ensure_binary(binary, "//tests:test_cuda_blend", build_if_needed=args.build_if_needed)

    results: list[BenchmarkResult] = []
    for width, height in args.sizes:
        overlap = max(128, min(width - 1, int(round(width * args.overlap_fraction))))
        directory = prepare_benchmark_directory(work_dir, width, height, overlap)
        for levels in args.levels:
            results.append(benchmark_cpp(binary, directory, levels))
            for backend in args.python_backends:
                results.append(benchmark_python(directory, levels, backend, device, args.iterations, args.warmup))

    rows = summarize_rows(results)
    print_table(rows)

    if args.output_json:
        output_path = Path(args.output_json).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps({"results": [asdict(result) for result in results], "summary": rows}, indent=2) + "\n")
        print(f"JSON written to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
