from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import torch

from .canvas import CanvasManager, CanvasManagerN
from .geometry import CanvasInfo, Rect
from .masks import ControlMasks, ControlMasksN
from .ops import (
    cast_like,
    copy_roi,
    ensure_batched,
    laplacian_blend_n,
    remap_to_canvas,
    remap_to_canvas_with_dest_map,
    simple_make_full,
)
from .status import CudaStatus, CudaStatusError


@dataclass
class StitchingContext:
    batch_size: int
    is_hard_seam: bool
    remap_1_x: np.ndarray | None = None
    remap_1_y: np.ndarray | None = None
    remap_2_x: np.ndarray | None = None
    remap_2_y: np.ndarray | None = None
    blend_seam: np.ndarray | None = None


@dataclass
class StitchingContextN:
    batch_size: int
    is_hard_seam: bool
    n_images: int
    remap_x: list[np.ndarray]
    remap_y: list[np.ndarray]
    blend_mask: np.ndarray | None = None
    seam_index: np.ndarray | None = None


@dataclass
class RemapRoiInfo:
    roi: Rect = field(default_factory=Rect)
    offset_x: int = 0
    offset_y: int = 0



def _status_or_raise(status: CudaStatus) -> None:
    if not status.ok():
        raise CudaStatusError(status)


class _DeviceCache:
    def __init__(self) -> None:
        self._cache: dict[tuple[str, str], torch.Tensor] = {}

    def tensor(self, array: np.ndarray, device: torch.device, dtype: torch.dtype | None = None) -> torch.Tensor:
        use_dtype = dtype or torch.from_numpy(array).dtype
        key = (str(device), f"{array.__array_interface__['data'][0]}:{use_dtype}")
        if key not in self._cache:
            self._cache[key] = torch.as_tensor(array, device=device, dtype=use_dtype)
        return self._cache[key]


class CudaStitchPano:
    def __init__(
        self,
        batch_size: int,
        num_levels: int,
        control_masks: ControlMasks,
        match_exposure: bool = False,
        quiet: bool = False,
        minimize_blend: bool = True,
        max_output_width: int = 0,
    ) -> None:
        del max_output_width
        self._status = CudaStatus()
        self._num_levels = num_levels
        self._match_exposure = match_exposure
        self._image_adjustment: torch.Tensor | None = None
        self._whole_seam_mask_image: np.ndarray | None = None
        self._cache = _DeviceCache()
        if not control_masks.is_valid():
            self._status = CudaStatus(1, "Stitching masks were not able to be loaded")
            return

        self._context = StitchingContext(batch_size=batch_size, is_hard_seam=(num_levels == 0))
        canvas_w = control_masks.canvas_width()
        canvas_h = control_masks.canvas_height()
        if not quiet:
            print(f"Stitched canvas size: {canvas_w} x {canvas_h}")

        self._canvas_manager = CanvasManager(
            CanvasInfo(
                width=canvas_w,
                height=canvas_h,
                positions=[(int(control_masks.positions[0].xpos), int(control_masks.positions[0].ypos)), (int(control_masks.positions[1].xpos), int(control_masks.positions[1].ypos))],
            ),
            minimize_blend=not self._context.is_hard_seam,
        )
        self._canvas_manager._remapper_1.width = control_masks.img1_col.shape[1]
        self._canvas_manager._remapper_1.height = control_masks.img1_col.shape[0]
        self._canvas_manager._remapper_2.width = control_masks.img2_col.shape[1]
        self._canvas_manager._remapper_2.height = control_masks.img2_col.shape[0]
        self._canvas_manager.updateMinimizeBlend(
            (control_masks.img1_col.shape[1], control_masks.img1_col.shape[0]),
            (control_masks.img2_col.shape[1], control_masks.img2_col.shape[0]),
        )

        blend_seam = self._canvas_manager.convertMaskMat(control_masks.whole_seam_mask_image)
        self._context.remap_1_x = control_masks.img1_col
        self._context.remap_1_y = control_masks.img1_row
        self._context.remap_2_x = control_masks.img2_col
        self._context.remap_2_y = control_masks.img2_row
        self._context.blend_seam = blend_seam.astype(np.float32, copy=False)
        if match_exposure:
            self._whole_seam_mask_image = control_masks.whole_seam_mask_image.copy()

    @property
    def status(self) -> CudaStatus:
        return self._status

    def canvas_width(self) -> int:
        return self._canvas_manager.canvas_width()

    def canvas_height(self) -> int:
        return self._canvas_manager.canvas_height()

    def batch_size(self) -> int:
        return self._context.batch_size

    def _tensor(self, array: np.ndarray, device: torch.device, dtype: torch.dtype | None = None) -> torch.Tensor:
        return self._cache.tensor(array, device=device, dtype=dtype)

    def _compute_image_adjustment(self, image1: torch.Tensor, image2: torch.Tensor) -> torch.Tensor | None:
        if self._whole_seam_mask_image is None:
            return None
        seam = self._whole_seam_mask_image
        img1 = ensure_batched(image1)[0].detach().cpu().numpy().astype(np.float32)
        img2 = ensure_batched(image2)[0].detach().cpu().numpy().astype(np.float32)
        top_left_1 = self._canvas_manager.canvas_positions()[0]
        top_left_2 = self._canvas_manager.canvas_positions()[1]
        offset = match_seam_images(img1, img2, seam, 100, top_left_1, top_left_2)
        if offset is None:
            return None
        return torch.tensor(offset, device=image1.device, dtype=torch.float32)

    def process(self, input_image_1: torch.Tensor, input_image_2: torch.Tensor, canvas: torch.Tensor | None = None) -> torch.Tensor:
        _status_or_raise(self._status)
        input_image_1 = ensure_batched(input_image_1)
        input_image_2 = ensure_batched(input_image_2)
        if input_image_1.shape[0] != self._context.batch_size or input_image_2.shape[0] != self._context.batch_size:
            raise ValueError("Input batch size does not match stitcher batch size")
        if input_image_1.shape[-1] not in (3, 4):
            raise ValueError("Only 3- and 4-channel inputs are supported")

        device = input_image_1.device
        if canvas is None:
            canvas = torch.zeros(
                (self._context.batch_size, self.canvas_height(), self.canvas_width(), input_image_1.shape[-1]),
                device=device,
                dtype=input_image_1.dtype,
            )
        else:
            canvas = ensure_batched(canvas)
            canvas.zero_()

        if self._match_exposure and self._image_adjustment is None:
            self._image_adjustment = self._compute_image_adjustment(input_image_1, input_image_2)
            if self._image_adjustment is None:
                self._status = CudaStatus(2, "Unable to compute image adjustment")
                _status_or_raise(self._status)

        adjustment = self._image_adjustment
        if self._context.is_hard_seam:
            seam = self._tensor(self._context.blend_seam.astype(np.uint8), device, torch.uint8)
            remap_to_canvas_with_dest_map(
                input_image_1,
                canvas,
                self._tensor(self._context.remap_1_x, device, torch.int64),
                self._tensor(self._context.remap_1_y, device, torch.int64),
                1,
                seam,
                self._canvas_manager._x1,
                self._canvas_manager._y1,
                adjustment=adjustment.neg() if adjustment is not None else None,
            )
            remap_to_canvas_with_dest_map(
                input_image_2,
                canvas,
                self._tensor(self._context.remap_2_x, device, torch.int64),
                self._tensor(self._context.remap_2_y, device, torch.int64),
                0,
                seam,
                self._canvas_manager._x2,
                self._canvas_manager._y2,
                adjustment=adjustment,
            )
            return canvas

        compute_dtype = torch.float32
        full1 = torch.zeros(
            (self._context.batch_size, self._context.blend_seam.shape[0], self._context.blend_seam.shape[1], input_image_1.shape[-1]),
            device=device,
            dtype=compute_dtype,
        )
        full2 = torch.zeros_like(full1)
        remap_to_canvas(
            input_image_1,
            canvas,
            self._tensor(self._context.remap_1_x, device, torch.int64),
            self._tensor(self._context.remap_1_y, device, torch.int64),
            self._canvas_manager._x1,
            self._canvas_manager._y1,
            adjustment=adjustment.neg() if adjustment is not None else None,
            fill_invalid_alpha=True,
        )
        simple_make_full(
            canvas.to(compute_dtype),
            full1,
            self._canvas_manager.remapped_image_roi_blend_1.width,
            full1.shape[1],
            self._canvas_manager.remapped_image_roi_blend_1.x,
            0,
            self._canvas_manager._remapper_1.xpos,
            0,
        )
        remap_to_canvas(
            input_image_2,
            canvas,
            self._tensor(self._context.remap_2_x, device, torch.int64),
            self._tensor(self._context.remap_2_y, device, torch.int64),
            self._canvas_manager._x2,
            self._canvas_manager._y2,
            adjustment=adjustment,
            fill_invalid_alpha=True,
        )
        simple_make_full(
            canvas.to(compute_dtype),
            full2,
            self._canvas_manager.remapped_image_roi_blend_2.width,
            full2.shape[1],
            self._canvas_manager._x2,
            0,
            self._canvas_manager._remapper_2.xpos,
            0,
        )
        seam = self._tensor(self._context.blend_seam, device, torch.float32)
        mask = torch.stack((seam, 1.0 - seam), dim=-1)
        blended = laplacian_blend_n([full1, full2], mask, max(1, self._num_levels))
        copy_roi(
            blended,
            canvas,
            Rect(0, 0, blended.shape[2], blended.shape[1]),
            0,
            0,
            self._canvas_manager._x2 - self._canvas_manager.overlap_padding(),
            0,
        )
        return cast_like(canvas, input_image_1.dtype)


class CudaStitchPanoN:
    def __init__(
        self,
        batch_size: int,
        num_levels: int,
        control_masks: ControlMasksN,
        match_exposure: bool = False,
        quiet: bool = False,
        minimize_blend: bool = True,
    ) -> None:
        self._status = CudaStatus()
        self._num_levels = num_levels
        self._match_exposure = match_exposure
        self._minimize_blend = bool(minimize_blend and num_levels > 0)
        self._cache = _DeviceCache()
        self._blend_roi_canvas = Rect()
        self._write_roi_canvas = Rect()
        self._remap_rois: list[RemapRoiInfo] = []

        if not control_masks.is_valid():
            self._status = CudaStatus(1, "Stitching masks (N-image) could not be loaded")
            return

        n = len(control_masks.img_col)
        if n < 2 or n > 8:
            self._status = CudaStatus(2, "Unsupported N for blend (supported 2..8)")
            return

        canvas_w = control_masks.canvas_width()
        canvas_h = control_masks.canvas_height()
        if not quiet:
            print(f"Stitched (N-image) canvas size: {canvas_w} x {canvas_h}")
        positions = [(int(p.xpos), int(p.ypos)) for p in control_masks.positions]
        self._canvas_manager = CanvasManagerN(CanvasInfo(canvas_w, canvas_h, positions), minimize_blend=self._minimize_blend)
        for idx, remap in enumerate(control_masks.img_col):
            self._canvas_manager.set_remap_size(idx, (remap.shape[1], remap.shape[0]))

        seam_index_padded = self._canvas_manager.convertMaskMat(control_masks.whole_seam_mask_indexed.copy())
        seam_for_blend = seam_index_padded
        self._context = StitchingContextN(
            batch_size=batch_size,
            is_hard_seam=(num_levels == 0),
            n_images=n,
            remap_x=list(control_masks.img_col),
            remap_y=list(control_masks.img_row),
            seam_index=seam_index_padded,
        )

        if not self._context.is_hard_seam and self._minimize_blend:
            boundary_bbox = seam_boundary_bbox(seam_index_padded)
            if boundary_bbox is not None:
                self._write_roi_canvas = expand_and_clamp(boundary_bbox, self._canvas_manager.overlap_padding(), canvas_w, canvas_h)
                self._blend_roi_canvas = expand_and_clamp(self._write_roi_canvas, pyramid_margin(num_levels), canvas_w, canvas_h)
                self._blend_roi_canvas = align_and_clamp(self._blend_roi_canvas, pyramid_alignment(num_levels), canvas_w, canvas_h)
                seam_for_blend = seam_index_padded[
                    self._blend_roi_canvas.y : self._blend_roi_canvas.bottom,
                    self._blend_roi_canvas.x : self._blend_roi_canvas.right,
                ]
                self._remap_rois = [RemapRoiInfo() for _ in range(n)]
                for i, pos in enumerate(self._canvas_manager.canvas_positions()):
                    size = control_masks.img_col[i].shape[1], control_masks.img_col[i].shape[0]
                    img_rect = Rect(pos[0], pos[1], size[0], size[1])
                    inter = img_rect.intersect(self._blend_roi_canvas)
                    self._remap_rois[i].offset_x = pos[0] - self._blend_roi_canvas.x
                    self._remap_rois[i].offset_y = pos[1] - self._blend_roi_canvas.y
                    if not inter.empty:
                        self._remap_rois[i].roi = Rect(inter.x - pos[0], inter.y - pos[1], inter.width, inter.height)
            else:
                self._remap_rois = [RemapRoiInfo() for _ in range(n)]

        if not self._context.is_hard_seam:
            self._context.blend_mask = ControlMasksN.split_to_channels(seam_for_blend, n).astype(np.float32, copy=False)

    @property
    def status(self) -> CudaStatus:
        return self._status

    def canvas_width(self) -> int:
        return self._canvas_manager.canvas_width()

    def canvas_height(self) -> int:
        return self._canvas_manager.canvas_height()

    def batch_size(self) -> int:
        return self._context.batch_size

    def _tensor(self, array: np.ndarray, device: torch.device, dtype: torch.dtype | None = None) -> torch.Tensor:
        return self._cache.tensor(array, device=device, dtype=dtype)

    def process(self, inputs: Iterable[torch.Tensor], canvas: torch.Tensor | None = None) -> torch.Tensor:
        _status_or_raise(self._status)
        batched_inputs = [ensure_batched(inp) for inp in inputs]
        if len(batched_inputs) != self._context.n_images:
            raise ValueError("inputs size != N")
        batch = batched_inputs[0].shape[0]
        channels = batched_inputs[0].shape[-1]
        if any(inp.shape[0] != batch for inp in batched_inputs):
            raise ValueError("Mismatched batch sizes")
        if batch != self._context.batch_size:
            raise ValueError("Input batch size does not match stitcher batch size")
        device = batched_inputs[0].device

        if canvas is None:
            canvas = torch.zeros((batch, self.canvas_height(), self.canvas_width(), channels), device=device, dtype=batched_inputs[0].dtype)
        else:
            canvas = ensure_batched(canvas)
            canvas.zero_()

        seam_index = self._tensor(self._context.seam_index, device, torch.int64)
        if self._context.is_hard_seam:
            for i in range(self._context.n_images):
                remap_to_canvas_with_dest_map(
                    batched_inputs[i],
                    canvas,
                    self._tensor(self._context.remap_x[i], device, torch.int64),
                    self._tensor(self._context.remap_y[i], device, torch.int64),
                    i,
                    seam_index,
                    self._canvas_manager.canvas_positions()[i][0],
                    self._canvas_manager.canvas_positions()[i][1],
                )
            return canvas

        if self._minimize_blend and not self._blend_roi_canvas.empty and not self._write_roi_canvas.empty:
            for i in range(self._context.n_images):
                remap_to_canvas_with_dest_map(
                    batched_inputs[i],
                    canvas,
                    self._tensor(self._context.remap_x[i], device, torch.int64),
                    self._tensor(self._context.remap_y[i], device, torch.int64),
                    i,
                    seam_index,
                    self._canvas_manager.canvas_positions()[i][0],
                    self._canvas_manager.canvas_positions()[i][1],
                )
            compute_buffers: list[torch.Tensor] = []
            blend_h = self._blend_roi_canvas.height
            blend_w = self._blend_roi_canvas.width
            for i in range(self._context.n_images):
                buf = torch.zeros((batch, blend_h, blend_w, channels), device=device, dtype=torch.float32)
                ri = self._remap_rois[i]
                remap_to_canvas(
                    batched_inputs[i],
                    buf,
                    self._tensor(self._context.remap_x[i], device, torch.int64),
                    self._tensor(self._context.remap_y[i], device, torch.int64),
                    ri.offset_x,
                    ri.offset_y,
                    roi=ri.roi,
                    fill_invalid_alpha=True,
                )
                compute_buffers.append(buf)
            mask = self._tensor(self._context.blend_mask, device, torch.float32)
            blended = laplacian_blend_n(compute_buffers, mask, max(1, self._num_levels))
            copy_roi(
                blended,
                canvas,
                Rect(0, 0, self._write_roi_canvas.width, self._write_roi_canvas.height),
                self._write_roi_canvas.x - self._blend_roi_canvas.x,
                self._write_roi_canvas.y - self._blend_roi_canvas.y,
                self._write_roi_canvas.x,
                self._write_roi_canvas.y,
            )
            return cast_like(canvas, batched_inputs[0].dtype)

        compute_buffers = []
        for i in range(self._context.n_images):
            buf = torch.zeros((batch, self.canvas_height(), self.canvas_width(), channels), device=device, dtype=torch.float32)
            remap_to_canvas(
                batched_inputs[i],
                buf,
                self._tensor(self._context.remap_x[i], device, torch.int64),
                self._tensor(self._context.remap_y[i], device, torch.int64),
                self._canvas_manager.canvas_positions()[i][0],
                self._canvas_manager.canvas_positions()[i][1],
                fill_invalid_alpha=True,
            )
            compute_buffers.append(buf)
        mask = self._tensor(self._context.blend_mask, device, torch.float32)
        blended = laplacian_blend_n(compute_buffers, mask, max(1, self._num_levels))
        copy_roi(blended, canvas, Rect(0, 0, blended.shape[2], blended.shape[1]), 0, 0, 0, 0)
        return cast_like(canvas, batched_inputs[0].dtype)


def match_seam_images(
    image1: np.ndarray,
    image2: np.ndarray,
    seam: np.ndarray,
    n: int,
    top_left_1: tuple[int, int],
    top_left_2: tuple[int, int],
    verbose: bool = False,
) -> tuple[float, float, float] | None:
    if seam.dtype != np.uint8:
        raise ValueError("Seam mask must be uint8")

    sum_left = np.zeros(3, dtype=np.float64)
    sum_right = np.zeros(3, dtype=np.float64)
    count_left = 0
    count_right = 0
    seam_column = np.full(image1.shape[0], -1, dtype=np.int32)
    top_divisor = 4
    bottom_divisor = 10

    def _sample_left(img: np.ndarray, top_left: tuple[int, int]) -> tuple[np.ndarray, int]:
        total = np.zeros(3, dtype=np.float64)
        count = 0
        start_row = img.shape[0] // top_divisor
        end_row = ((bottom_divisor - 1) * img.shape[0]) // bottom_divisor
        for r in range(start_row, end_row):
            global_row = top_left[1] + r
            if global_row < 0 or global_row >= seam.shape[0]:
                continue
            col_start = top_left[0]
            col_end = top_left[0] + img.shape[1]
            seam_global_col = -1
            for c in range(col_start, col_end):
                if 0 <= c < seam.shape[1] and seam[global_row, c] == 0:
                    seam_global_col = c
                    seam_column[r] = c
                    break
            if seam_global_col == -1:
                continue
            seam_local_col = seam_global_col - top_left[0]
            sample_start = max(0, seam_local_col - n)
            pixels = img[r, sample_start:seam_local_col, :3]
            if pixels.size:
                total += pixels.reshape(-1, 3).sum(axis=0)
                count += pixels.shape[0]
        return total, count

    def _sample_right(img: np.ndarray, top_left: tuple[int, int]) -> tuple[np.ndarray, int]:
        total = np.zeros(3, dtype=np.float64)
        count = 0
        start_row = img.shape[0] // top_divisor
        end_row = ((bottom_divisor - 1) * img.shape[0]) // bottom_divisor
        for r in range(start_row, end_row):
            global_row = top_left[1] + r
            if global_row < 0 or global_row >= seam.shape[0]:
                continue
            col_start = top_left[0]
            col_end = top_left[0] + img.shape[1]
            seam_global_col = -1
            for c in range(col_start, col_end):
                if 0 <= c < seam.shape[1] and seam[global_row, c] == 0:
                    seam_global_col = c
                    break
            if seam_global_col == -1:
                continue
            seam_local_col = seam_global_col - top_left[0]
            sample_end = min(img.shape[1], seam_local_col + n)
            pixels = img[r, seam_local_col:sample_end, :3]
            if pixels.size:
                total += pixels.reshape(-1, 3).sum(axis=0)
                count += pixels.shape[0]
        return total, count

    sum_left, count_left = _sample_left(image1, top_left_1)
    sum_right, count_right = _sample_right(image2, top_left_2)
    if count_left == 0 or count_right == 0:
        return None
    avg_left = sum_left / count_left
    avg_right = sum_right / count_right
    offset = (avg_left - avg_right) * 0.5
    if verbose:
        print(f"Average values (Image1, left side): {avg_left}")
        print(f"Average values (Image2, right side): {avg_right}")
        print(f"Offset: {offset}")
    return float(offset[0]), float(offset[1]), float(offset[2])


def pyramid_margin(num_levels: int) -> int:
    if num_levels <= 0:
        return 0
    return 1 << min(num_levels, 30)


def pyramid_alignment(num_levels: int) -> int:
    if num_levels <= 1:
        return 1
    return 1 << min(num_levels - 1, 30)


def seam_boundary_bbox(seam_index: np.ndarray) -> Rect | None:
    if seam_index.dtype != np.uint8:
        raise ValueError("Expected indexed uint8 seam")
    h, w = seam_index.shape
    if w <= 1 or h <= 1:
        return None
    min_x = w
    min_y = h
    max_x = -1
    max_y = -1
    for y in range(h):
        row = seam_index[y]
        row_down = seam_index[y + 1] if y + 1 < h else None
        for x in range(w):
            value = row[x]
            if x + 1 < w and value != row[x + 1]:
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x + 1)
                max_y = max(max_y, y)
            if row_down is not None and value != row_down[x]:
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x)
                max_y = max(max_y, y + 1)
    if max_x < 0 or max_y < 0:
        return None
    return Rect(min_x, min_y, max_x - min_x + 1, max_y - min_y + 1)


def expand_and_clamp(rect: Rect, pad: int, max_w: int, max_h: int) -> Rect:
    if rect.empty:
        return Rect()
    x0 = max(0, rect.x - pad)
    y0 = max(0, rect.y - pad)
    x1 = min(max_w, rect.right + pad)
    y1 = min(max_h, rect.bottom + pad)
    return Rect(x0, y0, max(0, x1 - x0), max(0, y1 - y0))


def align_and_clamp(rect: Rect, align: int, max_w: int, max_h: int) -> Rect:
    if rect.empty or align <= 1:
        return rect
    x0 = max(0, (rect.x // align) * align)
    y0 = max(0, (rect.y // align) * align)
    x1 = min(max_w, ((rect.right + align - 1) // align) * align)
    y1 = min(max_h, ((rect.bottom + align - 1) // align) * align)
    return Rect(x0, y0, max(0, x1 - x0), max(0, y1 - y0))
