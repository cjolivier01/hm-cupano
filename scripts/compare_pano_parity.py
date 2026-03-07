#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
import tifffile

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cupano import ControlMasks, ControlMasksN, CudaStitchPano, CudaStitchPanoN


@dataclass
class DiffMetrics:
    max_abs: int
    mean_abs: float
    rmse: float
    psnr: float
    shape: tuple[int, ...]
    dtype_cpp: str
    dtype_python: str


def run(cmd: Sequence[str], cwd: Path | None = None) -> None:
    print("EXEC:", " ".join(str(part) for part in cmd))
    subprocess.run(list(cmd), cwd=str(cwd) if cwd else None, check=True)


def ensure_binary(binary: Path, target: str, build_if_needed: bool) -> Path:
    if binary.exists():
        return binary
    if not build_if_needed:
        raise FileNotFoundError(f"Missing binary: {binary}")
    bazelisk = shutil.which("bazelisk") or shutil.which("bazel")
    if not bazelisk:
        raise FileNotFoundError("bazelisk/bazel not found; cannot build parity binary")
    run([bazelisk, "build", target], cwd=REPO_ROOT)
    if not binary.exists():
        raise FileNotFoundError(f"Binary still missing after build: {binary}")
    return binary


def copy_image(src: Path, dst: Path) -> None:
    image = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(src)
    if not cv2.imwrite(str(dst), image):
        raise RuntimeError(f"Failed to write {dst}")


def _tiff_position(path: Path) -> tuple[float, float]:
    with tifffile.TiffFile(str(path)) as tif:
        tags = tif.pages[0].tags

        def _to_float(value: object) -> float:
            if isinstance(value, tuple) and len(value) == 2:
                num, den = value
                return float(num) / float(den or 1)
            if isinstance(value, (list, tuple)) and value:
                return _to_float(value[0])
            return float(value)

        def _snap(value: float, eps: float = 1e-3) -> float:
            rounded = round(value)
            if abs(value - rounded) < eps:
                return float(rounded)
            return float(value)

        xres = _to_float(tags["XResolution"].value) if "XResolution" in tags else 0.0
        yres = _to_float(tags["YResolution"].value) if "YResolution" in tags else 0.0
        xpos = _to_float(tags["XPosition"].value) if "XPosition" in tags else 0.0
        ypos = _to_float(tags["YPosition"].value) if "YPosition" in tags else 0.0
    return _snap(xpos * xres), _snap(ypos * yres)


def ensure_two_image_seam(directory: Path) -> None:
    seam_path = directory / "seam_file.png"
    if seam_path.exists():
        return

    map0 = cv2.imread(str(directory / "mapping_0000_x.tif"), cv2.IMREAD_ANYDEPTH)
    map1 = cv2.imread(str(directory / "mapping_0001_x.tif"), cv2.IMREAD_ANYDEPTH)
    if map0 is None or map1 is None:
        raise FileNotFoundError("Missing mapping_000{0,1}_x.tif for seam fallback generation")

    pos0 = _tiff_position(directory / "mapping_0000.tif")
    pos1 = _tiff_position(directory / "mapping_0001.tif")
    min_x = min(pos0[0], pos1[0])
    min_y = min(pos0[1], pos1[1])
    x0 = int(round(pos0[0] - min_x))
    y0 = int(round(pos0[1] - min_y))
    x1 = int(round(pos1[0] - min_x))
    y1 = int(round(pos1[1] - min_y))

    canvas_w = max(x0 + map0.shape[1], x1 + map1.shape[1])
    canvas_h = max(y0 + map0.shape[0], y1 + map1.shape[0])
    seam = np.ones((canvas_h, canvas_w), dtype=np.uint8)

    only_img2 = np.zeros_like(seam, dtype=bool)
    only_img2[y1 : y1 + map1.shape[0], x1 : x1 + map1.shape[1]] = True
    only_img2[y0 : y0 + map0.shape[0], x0 : x0 + map0.shape[1]] = False
    seam[only_img2] = 0

    overlap_x0 = max(x0, x1)
    overlap_x1 = min(x0 + map0.shape[1], x1 + map1.shape[1])
    overlap_y0 = max(y0, y1)
    overlap_y1 = min(y0 + map0.shape[0], y1 + map1.shape[0])
    if overlap_x1 > overlap_x0 and overlap_y1 > overlap_y0:
        boundary = overlap_x0 + (overlap_x1 - overlap_x0) // 2
        seam[overlap_y0:overlap_y1, boundary:overlap_x1] = 0
    else:
        seam[y1 : y1 + map1.shape[0], x1 : x1 + map1.shape[1]] = 0

    if not cv2.imwrite(str(seam_path), seam):
        raise RuntimeError(f"Failed to write fallback seam to {seam_path}")


def prepare_two_image_directory(left: Path, right: Path, directory: Path, max_control_points: int, scale: float | None) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    left_dst = directory / "left.png"
    right_dst = directory / "right.png"
    copy_image(left, left_dst)
    copy_image(right, right_dst)

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "create_control_points.py"),
        "--left",
        str(left_dst),
        "--right",
        str(right_dst),
        "--max-control-points",
        str(max_control_points),
    ]
    if scale is not None:
        cmd.extend(["--scale", str(scale)])
    run(cmd, cwd=REPO_ROOT)
    ensure_two_image_seam(directory)
    return directory


def load_input_images(directory: Path, num_images: int) -> list[np.ndarray]:
    images: list[np.ndarray] = []
    if num_images == 2 and (directory / "left.png").exists() and (directory / "right.png").exists():
        names = ["left.png", "right.png"]
    else:
        names = [f"image{i}.png" for i in range(num_images)]

    for name in names:
        image = cv2.imread(str(directory / name), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(directory / name)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected 3-channel BGR input for {name}, got shape {image.shape}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        images.append(image)
    return images


def to_tensor_batch(images: Sequence[np.ndarray], device: torch.device) -> list[torch.Tensor]:
    tensors: list[torch.Tensor] = []
    for image in images:
        tensor = torch.from_numpy(np.ascontiguousarray(image)).to(device=device, dtype=torch.uint8).unsqueeze(0)
        tensors.append(tensor)
    return tensors


def torch_image_to_numpy(image: torch.Tensor) -> np.ndarray:
    image = image.detach().cpu()
    if image.ndim == 4:
        image = image[0]
    if image.dtype.is_floating_point:
        image = image.round().clamp(0, 255).to(torch.uint8)
    else:
        image = image.to(torch.uint8)
    return np.ascontiguousarray(image.numpy())


def save_python_output(output: torch.Tensor, path: Path) -> None:
    image = torch_image_to_numpy(output)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Failed to write {path}")


def run_python_stitch(
    directory: Path,
    num_images: int,
    levels: int,
    adjust: bool,
    device: torch.device,
    output_path: Path,
    backend: str,
) -> torch.Tensor:
    inputs = to_tensor_batch(load_input_images(directory, num_images), device)
    if num_images == 2:
        masks = ControlMasks(str(directory))
        stitcher = CudaStitchPano(
            batch_size=1,
            num_levels=levels,
            control_masks=masks,
            match_exposure=adjust,
            quiet=True,
            backend=backend,
        )
        output = stitcher.process(inputs[0], inputs[1])
    else:
        masks = ControlMasksN(str(directory), num_images)
        stitcher = CudaStitchPanoN(
            batch_size=1,
            num_levels=levels,
            control_masks=masks,
            match_exposure=adjust,
            quiet=True,
            minimize_blend=True,
            backend=backend,
        )
        output = stitcher.process(inputs)
    save_python_output(output, output_path)
    return output


def run_cpp_stitch(binary: Path, directory: Path, num_images: int, levels: int, adjust: bool, output_path: Path) -> None:
    cmd = [
        str(binary),
        f"--directory={directory}",
        f"--output={output_path}",
        f"--levels={levels}",
        f"--adjust={1 if adjust else 0}",
    ]
    if num_images != 2:
        cmd.append(f"--num-images={num_images}")
    run(cmd, cwd=REPO_ROOT)


def read_output(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    return image


def compute_diff_metrics(cpp_image: np.ndarray, python_image: np.ndarray) -> DiffMetrics:
    if cpp_image.shape != python_image.shape:
        raise ValueError(f"Shape mismatch: C++ {cpp_image.shape}, Python {python_image.shape}")
    diff = cpp_image.astype(np.float32) - python_image.astype(np.float32)
    abs_diff = np.abs(diff)
    mse = float(np.mean(diff * diff))
    rmse = math.sqrt(mse)
    psnr = float("inf") if mse == 0.0 else 20.0 * math.log10(255.0 / rmse)
    return DiffMetrics(
        max_abs=int(abs_diff.max(initial=0)),
        mean_abs=float(abs_diff.mean()),
        rmse=rmse,
        psnr=psnr,
        shape=cpp_image.shape,
        dtype_cpp=str(cpp_image.dtype),
        dtype_python=str(python_image.dtype),
    )


def save_diff_image(cpp_image: np.ndarray, python_image: np.ndarray, path: Path) -> None:
    diff = np.abs(cpp_image.astype(np.int16) - python_image.astype(np.int16)).astype(np.uint8)
    if diff.ndim == 3:
        if diff.shape[2] == 4:
            diff_vis = diff[..., :3].max(axis=2)
        else:
            diff_vis = diff.max(axis=2)
    else:
        diff_vis = diff
    diff_vis = cv2.normalize(diff_vis, None, 0, 255, cv2.NORM_MINMAX)
    diff_vis = cv2.applyColorMap(diff_vis, cv2.COLORMAP_INFERNO)
    if not cv2.imwrite(str(path), diff_vis):
        raise RuntimeError(f"Failed to write {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Python cupano output against the existing C++/CUDA stitchers")
    parser.add_argument("--directory", default=None, help="Prepared stitching directory containing images, mappings, and seam_file.png")
    parser.add_argument("--left", default=None, help="Left/source image for auto-generating a two-image parity directory")
    parser.add_argument("--right", default=None, help="Right/source image for auto-generating a two-image parity directory")
    parser.add_argument("--num-images", type=int, default=2, help="Number of images for the parity run")
    parser.add_argument("--levels", type=int, default=0, help="Number of Laplacian levels to compare")
    parser.add_argument("--adjust", action="store_true", help="Enable exposure adjustment when supported")
    parser.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"), help="Torch device to use for the Python path")
    parser.add_argument("--backend", choices=("auto", "torch", "triton"), default="auto", help="Python backend to use for the cupano path")
    parser.add_argument("--build-if-needed", action="store_true", help="Build the C++ parity binary with bazelisk if missing")
    parser.add_argument("--binary", default=None, help="Path to the C++ parity binary to run")
    parser.add_argument("--work-dir", default=None, help="Working directory for generated inputs and outputs")
    parser.add_argument("--max-control-points", type=int, default=120, help="Control points used when auto-generating a two-image directory")
    parser.add_argument("--scale", type=float, default=None, help="Optional scale passed through to create_control_points.py")
    parser.add_argument("--tolerance-max", type=float, default=0.0, help="Maximum allowed absolute channel difference")
    parser.add_argument("--tolerance-mean", type=float, default=0.0, help="Maximum allowed mean absolute channel difference")
    parser.add_argument("--keep-work-dir", action="store_true", help="Do not delete an auto-created temporary work directory")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    num_images = max(2, args.num_images)
    work_dir: Path | None = None
    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    try:
        if args.work_dir:
            work_dir = Path(args.work_dir).resolve()
            work_dir.mkdir(parents=True, exist_ok=True)
        elif args.directory:
            work_dir = Path(args.directory).resolve()
        else:
            temp_dir = tempfile.TemporaryDirectory(prefix="cupano_parity_")
            work_dir = Path(temp_dir.name)

        if args.left and args.right:
            if num_images != 2:
                raise ValueError("Auto-generation from --left/--right currently supports only two-image parity runs")
            prepare_two_image_directory(
                Path(args.left).resolve(),
                Path(args.right).resolve(),
                work_dir,
                max_control_points=args.max_control_points,
                scale=args.scale,
            )
        elif not args.directory:
            raise ValueError("Provide either --directory or both --left and --right")

        default_binary = REPO_ROOT / "bazel-bin" / "tests" / ("test_cuda_blend" if num_images == 2 else "test_cuda_blend_n")
        default_target = "//tests:test_cuda_blend" if num_images == 2 else "//tests:test_cuda_blend_n"
        binary = Path(args.binary).resolve() if args.binary else default_binary
        binary = ensure_binary(binary, default_target, build_if_needed=args.build_if_needed)

        backend_suffix = f"_{args.backend}"
        cpp_out = work_dir / f"cpp_levels{args.levels}_n{num_images}.png"
        py_out = work_dir / f"python_levels{args.levels}_n{num_images}{backend_suffix}.png"
        diff_out = work_dir / f"diff_levels{args.levels}_n{num_images}{backend_suffix}.png"
        metrics_out = work_dir / f"metrics_levels{args.levels}_n{num_images}{backend_suffix}.json"

        device = torch.device(args.device)
        run_python_stitch(work_dir, num_images, args.levels, args.adjust, device, py_out, args.backend)
        run_cpp_stitch(binary, work_dir, num_images, args.levels, args.adjust, cpp_out)

        cpp_image = read_output(cpp_out)
        python_image = read_output(py_out)
        metrics = compute_diff_metrics(cpp_image, python_image)
        save_diff_image(cpp_image, python_image, diff_out)
        metrics_out.write_text(json.dumps(asdict(metrics), indent=2) + "\n")

        print(json.dumps(asdict(metrics), indent=2))
        print(f"C++ output:    {cpp_out}")
        print(f"Python output: {py_out}")
        print(f"Diff heatmap:  {diff_out}")
        print(f"Metrics JSON:  {metrics_out}")

        if metrics.max_abs > args.tolerance_max or metrics.mean_abs > args.tolerance_mean:
            print(
                f"Parity thresholds failed: max_abs={metrics.max_abs} (limit {args.tolerance_max}), "
                f"mean_abs={metrics.mean_abs:.6f} (limit {args.tolerance_mean})",
                file=sys.stderr,
            )
            return 1
        return 0
    finally:
        if temp_dir is not None and not args.keep_work_dir:
            temp_dir.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
