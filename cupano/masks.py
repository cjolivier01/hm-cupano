from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import tifffile
from PIL import Image

from .geometry import SpatialTiff


UNMAPPED_POSITION_VALUE = np.uint16(65535)


def _normalize_positions(positions: Iterable[SpatialTiff]) -> list[SpatialTiff]:
    positions = list(positions)
    min_x = min(p.xpos for p in positions)
    min_y = min(p.ypos for p in positions)
    return [SpatialTiff(p.xpos - min_x, p.ypos - min_y) for p in positions]


def _tag_to_float(value: object) -> float:
    if isinstance(value, tuple) and len(value) == 2:
        num, den = value
        den = den or 1
        return float(num) / float(den)
    if isinstance(value, (list, tuple)) and value:
        return _tag_to_float(value[0])
    return float(value)


def _snap_near_integer(value: float, eps: float = 1e-3) -> float:
    rounded = round(value)
    if abs(value - rounded) < eps:
        return float(rounded)
    return float(value)


def _get_geo_tiff(path: str | Path) -> SpatialTiff:
    with tifffile.TiffFile(str(path)) as tif:
        page = tif.pages[0]
        tags = page.tags
        xres = _tag_to_float(tags["XResolution"].value) if "XResolution" in tags else 0.0
        yres = _tag_to_float(tags["YResolution"].value) if "YResolution" in tags else 0.0
        xpos = _tag_to_float(tags["XPosition"].value) if "XPosition" in tags else 0.0
        ypos = _tag_to_float(tags["YPosition"].value) if "YPosition" in tags else 0.0
    return SpatialTiff(xpos=_snap_near_integer(xpos * xres), ypos=_snap_near_integer(ypos * yres))


def _read_indexed_png_or_grayscale(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        if image.mode == "P":
            return np.array(image, dtype=np.uint8)
    seam = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if seam is None:
        raise FileNotFoundError(path)
    return seam.astype(np.uint8, copy=False)


def _load_two_image_seam(path: str | Path) -> np.ndarray:
    seam = _read_indexed_png_or_grayscale(path)
    min_val = int(seam.min())
    max_val = int(seam.max())
    out = seam.copy()
    out[seam == max_val] = 0
    out[seam == min_val] = 1
    return out.astype(np.uint8, copy=False)


@dataclass
class ControlMasks:
    img1_col: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=np.uint16))
    img1_row: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=np.uint16))
    img2_col: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=np.uint16))
    img2_row: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=np.uint16))
    whole_seam_mask_image: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=np.uint8))
    positions: list[SpatialTiff] = field(default_factory=list)

    def __init__(self, game_dir: str | None = None):
        self.img1_col = np.empty((0, 0), dtype=np.uint16)
        self.img1_row = np.empty((0, 0), dtype=np.uint16)
        self.img2_col = np.empty((0, 0), dtype=np.uint16)
        self.img2_row = np.empty((0, 0), dtype=np.uint16)
        self.whole_seam_mask_image = np.empty((0, 0), dtype=np.uint8)
        self.positions = []
        if game_dir is not None:
            self.load(game_dir)

    def load(self, game_dir: str) -> bool:
        base = Path(game_dir)
        self.img1_col = cv2.imread(str(base / "mapping_0000_x.tif"), cv2.IMREAD_ANYDEPTH)
        self.img1_row = cv2.imread(str(base / "mapping_0000_y.tif"), cv2.IMREAD_ANYDEPTH)
        self.img2_col = cv2.imread(str(base / "mapping_0001_x.tif"), cv2.IMREAD_ANYDEPTH)
        self.img2_row = cv2.imread(str(base / "mapping_0001_y.tif"), cv2.IMREAD_ANYDEPTH)
        if any(x is None or x.size == 0 for x in (self.img1_col, self.img1_row, self.img2_col, self.img2_row)):
            return False
        self.whole_seam_mask_image = _load_two_image_seam(base / "seam_file.png")
        self.positions = _normalize_positions(
            [
                _get_geo_tiff(base / "mapping_0000.tif"),
                _get_geo_tiff(base / "mapping_0001.tif"),
            ]
        )
        return self.is_valid()

    def is_valid(self) -> bool:
        return (
            self.img1_col.size
            and self.img1_row.size
            and self.img2_col.size
            and self.img2_row.size
            and self.whole_seam_mask_image.size
            and len(self.positions) == 2
        )

    def canvas_width(self) -> int:
        return int(max(self.positions[0].xpos + self.img1_col.shape[1], self.positions[1].xpos + self.img2_col.shape[1]))

    def canvas_height(self) -> int:
        return int(max(self.positions[0].ypos + self.img1_col.shape[0], self.positions[1].ypos + self.img2_col.shape[0]))


@dataclass
class ControlMasksN:
    img_col: list[np.ndarray] = field(default_factory=list)
    img_row: list[np.ndarray] = field(default_factory=list)
    whole_seam_mask_indexed: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=np.uint8))
    positions: list[SpatialTiff] = field(default_factory=list)

    def __init__(self, directory: str | None = None, n_images: int | None = None):
        self.img_col = []
        self.img_row = []
        self.whole_seam_mask_indexed = np.empty((0, 0), dtype=np.uint8)
        self.positions = []
        if directory is not None and n_images is not None:
            self.load(directory, n_images)

    def load(self, directory: str, n_images: int) -> bool:
        base = Path(directory)
        self.img_col = []
        self.img_row = []
        self.positions = []
        for i in range(n_images):
            self.img_col.append(cv2.imread(str(base / f"mapping_{i:04d}_x.tif"), cv2.IMREAD_ANYDEPTH))
            self.img_row.append(cv2.imread(str(base / f"mapping_{i:04d}_y.tif"), cv2.IMREAD_ANYDEPTH))
            self.positions.append(_get_geo_tiff(base / f"mapping_{i:04d}.tif"))
        if any(x is None or x.size == 0 for x in self.img_col + self.img_row):
            return False
        self.positions = _normalize_positions(self.positions)
        self.whole_seam_mask_indexed = _read_indexed_png_or_grayscale(base / "seam_file.png")
        uniq = np.unique(self.whole_seam_mask_indexed)
        if uniq.size != n_images:
            return False
        if uniq[0] != 0 or uniq[-1] != n_images - 1:
            lut = np.zeros(256, dtype=np.uint8)
            for idx, value in enumerate(uniq.tolist()):
                lut[int(value)] = idx
            self.whole_seam_mask_indexed = lut[self.whole_seam_mask_indexed]
        return self.is_valid()

    def is_valid(self) -> bool:
        return (
            bool(self.img_col)
            and len(self.img_col) == len(self.img_row)
            and len(self.positions) == len(self.img_col)
            and self.whole_seam_mask_indexed.size > 0
        )

    def canvas_width(self) -> int:
        return int(max(pos.xpos + remap.shape[1] for pos, remap in zip(self.positions, self.img_col, strict=True)))

    def canvas_height(self) -> int:
        return int(max(pos.ypos + remap.shape[0] for pos, remap in zip(self.positions, self.img_row, strict=True)))

    @staticmethod
    def split_to_channels(indexed: np.ndarray, n_images: int) -> np.ndarray:
        if indexed.dtype != np.uint8:
            indexed = indexed.astype(np.uint8)
        channels = [(indexed == i).astype(np.uint8) for i in range(n_images)]
        return np.stack(channels, axis=-1)
