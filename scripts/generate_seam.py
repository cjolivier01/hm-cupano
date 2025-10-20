#!/usr/bin/env python3
"""
Generate seam_file.png for a stitched folder using multiblend (default for N>=3)
or enblend (supported for 2 and 3+). Prefers Bazel external tools when available.

Usage:
  python3 scripts/generate_seam.py --directory <dir> --num-images 3 --seam multiblend
  python3 scripts/generate_seam.py --directory <dir> --num-images 3 --seam enblend
  python3 scripts/generate_seam.py --directory <dir> --num-images 2 --seam enblend
"""

import argparse
import glob
import os
import shutil
import subprocess
from typing import List

from PIL import Image
import numpy as np
import cv2


def prefer_bazel_tool(name: str) -> List[str]:
    if name == "multiblend":
        if shutil.which("bazelisk") or shutil.which("bazel"):
            return ["bazelisk", "run", "@multiblend//:multiblend", "--"]
    if name == "enblend":
        if shutil.which("bazelisk") or shutil.which("bazel"):
            return ["bazelisk", "run", "@enblend//:enblend", "--"]
    return [name]


def run(cmd: List[str], cwd: str) -> None:
    print("EXEC:", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--directory", required=True, help="Folder with mapping_????.tif")
    ap.add_argument("--num-images", type=int, required=True, help="Number of input images (>=2)")
    ap.add_argument(
        "--seam",
        choices=["multiblend", "enblend"],
        default="multiblend",
        help="Seam generator to use (>=3 images: multiblend recommended)",
    )
    args = ap.parse_args()

    d = os.path.abspath(args.directory)
    os.makedirs(d, exist_ok=True)

    # Choose generator
    mapping_layers = sorted(glob.glob(os.path.join(d, "mapping_????.tif")))
    if not mapping_layers:
        raise SystemExit("No mapping_????.tif files found. Run nona first.")

    if args.num_images >= 3 and args.seam == "multiblend":
        tried_bazel = False
        cmd = prefer_bazel_tool("multiblend") + [
            f"--save-seams={os.path.join(d, 'seam_file.png')}",
            "-o",
            os.path.join(d, "panorama.tif"),
        ] + mapping_layers
        try:
            if cmd[0] in ("bazelisk", "bazel"):
                tried_bazel = True
            run(cmd, cwd=d)
            print("Wrote:", os.path.join(d, "seam_file.png"))
            return
        except subprocess.CalledProcessError:
            if tried_bazel and shutil.which("multiblend"):
                # Fallback to system multiblend
                cmd = ["multiblend", f"--save-seams={os.path.join(d, 'seam_file.png')}", "-o", os.path.join(d, "panorama.tif")] + mapping_layers
                run(cmd, cwd=d)
                print("Wrote:", os.path.join(d, "seam_file.png"))
                return
            raise

    # enblend path (2 or 3+)
    tried_bazel = False
    cmd = prefer_bazel_tool("enblend") + [
        f"--save-masks={os.path.join(d, 'seam-%i.png')}",
        "-o",
        os.path.join(d, "panorama.tif"),
    ] + mapping_layers
    try:
        if cmd[0] in ("bazelisk", "bazel"):
            tried_bazel = True
        run(cmd, cwd=d)
    except subprocess.CalledProcessError:
        if tried_bazel and shutil.which("enblend"):
            cmd = ["enblend", f"--save-masks={os.path.join(d, 'seam-%i.png')}", "-o", os.path.join(d, "panorama.tif")] + mapping_layers
            run(cmd, cwd=d)
        else:
            raise

    if args.num_images == 2:
        # For two, the seam is binary. Use the single mask directly.
        # Try seam-1.png then seam-0.png
        for candidate in ("seam-1.png", "seam-0.png"):
            src = os.path.join(d, candidate)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(d, "seam_file.png"))
                print("Wrote:", os.path.join(d, "seam_file.png"))
                return
        raise SystemExit("Could not find enblend seam mask for two images")

    # Compose paletted mask for N>=3 from enblend output masks
    mask_files = sorted(glob.glob(os.path.join(d, "seam-*.png")))
    if not mask_files:
        raise SystemExit("No seam-*.png produced by enblend")

    labels = (np.array(Image.open(mask_files[0])) >= 128).astype(np.uint8)
    for idx, mf in enumerate(mask_files[1:], start=2):
        arr = np.array(Image.open(mf))
        if arr.shape != labels.shape:
            arr = cv2.resize(arr, (labels.shape[1], labels.shape[0]), interpolation=cv2.INTER_NEAREST)
        labels[arr >= 128] = idx

    img = Image.fromarray(labels, mode="P")
    palette = []
    base = [(0,0,0), (0,255,0), (255,0,0), (0,0,255), (255,255,0), (255,0,255), (0,255,255), (128,128,128)]
    for i in range(256):
        c = base[i % len(base)] if i < args.num_images else (0,0,0)
        palette.extend(list(c))
    img.putpalette(palette)
    img.save(os.path.join(d, "seam_file.png"))
    print("Wrote:", os.path.join(d, "seam_file.png"))


if __name__ == "__main__":
    main()
