from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import CanvasInfo, Rect, Remapper


@dataclass
class CanvasManager:
    canvas_info: CanvasInfo
    minimize_blend: bool
    overlap_pad: int = 128

    def __post_init__(self) -> None:
        self._remapper_1 = Remapper()
        self._remapper_2 = Remapper()
        self._x1 = 0
        self._y1 = 0
        self._x2 = 0
        self._y2 = 0
        self._overlapping_width = 0
        self.remapped_image_roi_blend_1 = Rect()
        self.remapped_image_roi_blend_2 = Rect()

    def updateMinimizeBlend(self, remapped_size_1: tuple[int, int], remapped_size_2: tuple[int, int]) -> None:
        if len(self.canvas_info.positions) < 2:
            raise ValueError("CanvasManager requires two positions")

        self._x1, self._y1 = self.canvas_info.positions[0]
        self._x2, self._y2 = self.canvas_info.positions[1]
        self._remapper_1.width, self._remapper_1.height = remapped_size_1
        self._remapper_2.width, self._remapper_2.height = remapped_size_2

        width_1 = self._remapper_1.width
        self._overlapping_width = width_1 - self._x2
        if self._overlapping_width <= 0:
            raise ValueError("Images do not overlap; invalid two-image minimize_blend configuration")

        blend_width = self._overlapping_width + 2 * self.overlap_pad
        if self.minimize_blend:
            self._remapper_1.xpos = self._x1
            self._remapper_2.xpos = self._x1 + self.overlap_pad
            self.remapped_image_roi_blend_1 = Rect(
                self._x2 - self.overlap_pad - self._x1,
                0,
                blend_width - self.overlap_pad,
                remapped_size_1[1],
            )
            self.remapped_image_roi_blend_2 = Rect(0, 0, blend_width - self.overlap_pad, remapped_size_2[1])

    def convertMaskMat(self, mask: np.ndarray) -> np.ndarray:
        padded = mask
        padw = max(0, self.canvas_info.width - mask.shape[1])
        padh = max(0, self.canvas_info.height - mask.shape[0])
        if padw or padh:
            padded = np.pad(mask, ((0, padh), (0, padw)), mode="edge")

        if self.minimize_blend:
            x_start = self.canvas_info.positions[1][0] - self.overlap_pad
            x_end = self._remapper_1.width + self.overlap_pad
            return padded[:, x_start:x_end]
        return padded

    def overlap_padding(self) -> int:
        return self.overlap_pad

    def overlapping_width(self) -> int:
        return self._overlapping_width

    def canvas_width(self) -> int:
        return self.canvas_info.width

    def canvas_height(self) -> int:
        return self.canvas_info.height

    def canvas_positions(self) -> list[tuple[int, int]]:
        return self.canvas_info.positions


@dataclass
class CanvasManagerN:
    canvas_info: CanvasInfo
    minimize_blend: bool
    overlap_pad: int = 128

    def __post_init__(self) -> None:
        self._remappers = [Remapper() for _ in self.canvas_info.positions]

    def set_remap_size(self, idx: int, size: tuple[int, int]) -> None:
        self._remappers[idx].width, self._remappers[idx].height = size
        self._remappers[idx].xpos = self.canvas_info.positions[idx][0]

    def convertMaskMat(self, mask: np.ndarray) -> np.ndarray:
        padw = max(0, self.canvas_info.width - mask.shape[1])
        padh = max(0, self.canvas_info.height - mask.shape[0])
        if padw or padh:
            return np.pad(mask, ((0, padh), (0, padw)), mode="edge")
        return mask

    def overlap_padding(self) -> int:
        return self.overlap_pad

    def canvas_width(self) -> int:
        return self.canvas_info.width

    def canvas_height(self) -> int:
        return self.canvas_info.height

    def canvas_positions(self) -> list[tuple[int, int]]:
        return self.canvas_info.positions

    def remappers(self) -> list[Remapper]:
        return self._remappers
