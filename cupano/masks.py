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


def _scaled_shape(shape: tuple[int, int], scale: float) -> tuple[int, int]:
    return (max(1, int(np.floor(shape[0] * scale))), max(1, int(np.floor(shape[1] * scale))))


def _scale_span(start: float, size: int, scale: float) -> tuple[float, int]:
    scaled_start = float(np.floor(start * scale))
    scaled_end = int(np.ceil((start + size) * scale))
    return scaled_start, max(1, scaled_end - int(scaled_start))


def _scaled_canvas_size(positions: list[SpatialTiff], shapes: list[tuple[int, int]]) -> tuple[int, int]:
    width = max(int(position.xpos) + shape[1] for position, shape in zip(positions, shapes, strict=True))
    height = max(int(position.ypos) + shape[0] for position, shape in zip(positions, shapes, strict=True))
    return max(1, width), max(1, height)


def _scale_to_fit_max_width(
    positions: list[SpatialTiff],
    shapes: list[tuple[int, int]],
    native_width: int,
    max_output_width: int,
) -> float:
    low = 0.0
    high = float(max_output_width) / float(native_width)
    direct_positions: list[SpatialTiff] = []
    direct_shapes: list[tuple[int, int]] = []
    for position, shape in zip(positions, shapes, strict=True):
        xpos, width = _scale_span(position.xpos, shape[1], high)
        ypos, height = _scale_span(position.ypos, shape[0], high)
        direct_positions.append(SpatialTiff(xpos=xpos, ypos=ypos))
        direct_shapes.append((height, width))
    if _scaled_canvas_size(direct_positions, direct_shapes)[0] <= max_output_width:
        return high
    for _ in range(32):
        mid = (low + high) / 2.0
        scaled_positions: list[SpatialTiff] = []
        scaled_shapes: list[tuple[int, int]] = []
        for position, shape in zip(positions, shapes, strict=True):
            xpos, width = _scale_span(position.xpos, shape[1], mid)
            ypos, height = _scale_span(position.ypos, shape[0], mid)
            scaled_positions.append(SpatialTiff(xpos=xpos, ypos=ypos))
            scaled_shapes.append((height, width))
        if _scaled_canvas_size(scaled_positions, scaled_shapes)[0] <= max_output_width:
            low = mid
        else:
            high = mid
    return low if low > 0.0 else high


def _resize_nearest(array: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    resized = cv2.resize(array, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return resized.astype(array.dtype, copy=False)


def _resize_remap_preserving_unmapped(array: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    resized = _resize_nearest(array, shape)
    invalid = np.zeros(shape, dtype=bool)
    scale_y = array.shape[0] / shape[0]
    scale_x = array.shape[1] / shape[1]
    for y in range(shape[0]):
        y0 = max(0, min(array.shape[0] - 1, int(np.floor(y * scale_y))))
        y1 = max(y0 + 1, min(array.shape[0], int(np.ceil((y + 1) * scale_y))))
        for x in range(shape[1]):
            x0 = max(0, min(array.shape[1] - 1, int(np.floor(x * scale_x))))
            x1 = max(x0 + 1, min(array.shape[1], int(np.ceil((x + 1) * scale_x))))
            invalid[y, x] = bool(np.any(array[y0:y1, x0:x1] == UNMAPPED_POSITION_VALUE))
    resized[invalid] = UNMAPPED_POSITION_VALUE
    return resized.astype(np.uint16, copy=False)


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

    def scale_to_max_output_width(self, max_output_width: int) -> None:
        if max_output_width <= 0 or not self.is_valid():
            return
        canvas_width = self.canvas_width()
        if canvas_width <= max_output_width:
            return
        native_shapes = [self.img1_col.shape, self.img2_col.shape]
        scale = _scale_to_fit_max_width(self.positions, native_shapes, canvas_width, max_output_width)
        scaled_positions: list[SpatialTiff] = []
        shapes: list[tuple[int, int]] = []
        for position, remap in (
            (self.positions[0], self.img1_col),
            (self.positions[1], self.img2_col),
        ):
            xpos, width = _scale_span(position.xpos, remap.shape[1], scale)
            ypos, height = _scale_span(position.ypos, remap.shape[0], scale)
            scaled_positions.append(SpatialTiff(xpos=xpos, ypos=ypos))
            shapes.append((height, width))
        img1_shape, img2_shape = shapes
        self.img1_col = _resize_remap_preserving_unmapped(self.img1_col, img1_shape)
        self.img1_row = _resize_remap_preserving_unmapped(self.img1_row, img1_shape)
        self.img2_col = _resize_remap_preserving_unmapped(self.img2_col, img2_shape)
        self.img2_row = _resize_remap_preserving_unmapped(self.img2_row, img2_shape)
        canvas_size = _scaled_canvas_size(scaled_positions, shapes)
        self.whole_seam_mask_image = _resize_nearest(self.whole_seam_mask_image, (canvas_size[1], canvas_size[0]))
        self.positions = scaled_positions


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

    def scale_to_max_output_width(self, max_output_width: int) -> None:
        if max_output_width <= 0 or not self.is_valid():
            return
        canvas_width = self.canvas_width()
        if canvas_width <= max_output_width:
            return
        native_shapes = [remap.shape for remap in self.img_col]
        scale = _scale_to_fit_max_width(self.positions, native_shapes, canvas_width, max_output_width)
        scaled_positions: list[SpatialTiff] = []
        shapes: list[tuple[int, int]] = []
        for position, remap in zip(self.positions, self.img_col, strict=True):
            xpos, width = _scale_span(position.xpos, remap.shape[1], scale)
            ypos, height = _scale_span(position.ypos, remap.shape[0], scale)
            scaled_positions.append(SpatialTiff(xpos=xpos, ypos=ypos))
            shapes.append((height, width))
        self.img_col = [
            _resize_remap_preserving_unmapped(remap, shape) for remap, shape in zip(self.img_col, shapes, strict=True)
        ]
        self.img_row = [
            _resize_remap_preserving_unmapped(remap, self.img_col[index].shape) for index, remap in enumerate(self.img_row)
        ]
        canvas_size = _scaled_canvas_size(scaled_positions, shapes)
        self.whole_seam_mask_indexed = _resize_nearest(
            self.whole_seam_mask_indexed, (canvas_size[1], canvas_size[0])
        )
        self.positions = scaled_positions

    @staticmethod
    def split_to_channels(indexed: np.ndarray, n_images: int) -> np.ndarray:
        if indexed.dtype != np.uint8:
            indexed = indexed.astype(np.uint8)
        channels = [(indexed == i).astype(np.uint8) for i in range(n_images)]
        return np.stack(channels, axis=-1)
