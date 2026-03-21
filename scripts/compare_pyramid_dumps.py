#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def _load(path: Path) -> np.ndarray:
    array = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if array is None:
        raise FileNotFoundError(path)
    return array.astype(np.float32, copy=False)


def _normalize_shape(array: np.ndarray) -> np.ndarray:
    if array.ndim == 3 and array.shape[-1] == 1:
        return array[..., 0]
    return array


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare two pano pyramid dump directories.")
    parser.add_argument("--reference", required=True, help="Reference dump directory.")
    parser.add_argument("--candidate", required=True, help="Candidate dump directory.")
    args = parser.parse_args()

    reference = Path(args.reference)
    candidate = Path(args.candidate)
    files = sorted(path.relative_to(reference) for path in reference.rglob("*.tiff"))
    if not files:
        raise SystemExit(f"No TIFF files found under {reference}")

    worst_rel: Path | None = None
    worst_max = -1.0
    worst_mean = -1.0
    first_nonzero: Path | None = None
    for rel in files:
        ref_path = reference / rel
        cand_path = candidate / rel
        if not cand_path.exists():
            raise SystemExit(f"Missing candidate file: {cand_path}")
        ref = _normalize_shape(_load(ref_path))
        cand = _normalize_shape(_load(cand_path))
        if ref.shape != cand.shape:
            raise SystemExit(f"Shape mismatch for {rel}: {ref.shape} vs {cand.shape}")
        diff = np.abs(ref - cand)
        max_abs = float(diff.max(initial=0.0))
        mean_abs = float(diff.mean()) if diff.size else 0.0
        print(f"{rel}: max_abs={max_abs:.9f} mean_abs={mean_abs:.9f}")
        if max_abs > 0.0 and first_nonzero is None:
            first_nonzero = rel
        if max_abs > worst_max or (max_abs == worst_max and mean_abs > worst_mean):
            worst_rel = rel
            worst_max = max_abs
            worst_mean = mean_abs

    if worst_rel is None:
        raise SystemExit("No files were compared")

    print(f"worst={worst_rel} max_abs={worst_max:.9f} mean_abs={worst_mean:.9f}")
    if first_nonzero is not None:
        print(f"first_nonzero={first_nonzero}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
