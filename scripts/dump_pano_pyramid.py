#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import torch

from cupano.masks import ControlMasks
from cupano.pano import CudaStitchPano


def _load_rgba(path: Path) -> torch.Tensor:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    return torch.from_numpy(image)


def main() -> int:
    parser = argparse.ArgumentParser(description="Dump the Python two-image pano pyramid intermediates.")
    parser.add_argument("--directory", required=True, help="Control-mask directory containing left.png/right.png and mapping files.")
    parser.add_argument("--levels", type=int, required=True, help="Requested blend levels.")
    parser.add_argument("--output-dir", required=True, help="Directory to write TIFF pyramid dumps into.")
    parser.add_argument("--backend", choices=("auto", "triton"), default="auto")
    parser.add_argument("--device", default="cuda", help="Torch device, for example cuda or cpu.")
    args = parser.parse_args()

    base = Path(args.directory)
    control_masks = ControlMasks(str(base))
    if not control_masks.is_valid():
        raise SystemExit(f"Unable to load control masks from {base}")

    image1 = _load_rgba(base / "left.png").to(device=args.device)
    image2 = _load_rgba(base / "right.png").to(device=args.device)

    pano = CudaStitchPano(1, args.levels, control_masks, backend=args.backend, enable_cuda_graphs=False, quiet=True)
    pano.process(image1, image2)
    pano.dump_soft_blend_pyramid(Path(args.output_dir), device=image1.device, channels=int(image1.shape[-1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
