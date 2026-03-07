from __future__ import annotations

from typing import Final

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - optional dependency
    triton = None
    tl = None

from .geometry import Rect
from .masks import UNMAPPED_POSITION_VALUE

_HAS_TRITON: Final[bool] = triton is not None and tl is not None
_SUPPORTED_DTYPES: Final[set[torch.dtype]] = {torch.uint8, torch.float32}
_SUPPORTED_CHANNELS: Final[set[int]] = {3, 4}
_UNMAPPED_VALUE: Final[int] = int(UNMAPPED_POSITION_VALUE)


def triton_available() -> bool:
    return _HAS_TRITON


def _gpu_tensor(tensor: torch.Tensor) -> bool:
    return tensor.is_cuda


def _supported_tensor(tensor: torch.Tensor) -> bool:
    return tensor.dtype in _SUPPORTED_DTYPES and tensor.ndim == 4 and tensor.shape[-1] in _SUPPORTED_CHANNELS and tensor.is_contiguous()


def can_use_triton_backend(*tensors: torch.Tensor) -> bool:
    if not _HAS_TRITON:
        return False
    return all(_gpu_tensor(t) and _supported_tensor(t) for t in tensors)


if _HAS_TRITON:
    @triton.jit
    def _copy_roi_kernel(
        src_ptr,
        dest_ptr,
        src_stride_b,
        src_stride_h,
        src_stride_w,
        src_stride_c,
        dest_stride_b,
        dest_stride_h,
        dest_stride_w,
        dest_stride_c,
        src_height,
        src_width,
        dest_height,
        dest_width,
        src_roi_x,
        src_roi_y,
        dest_offset_x,
        dest_offset_y,
        region_width,
        region_height,
        BLOCK_W: tl.constexpr,
        BLOCK_H: tl.constexpr,
        CHANNELS: tl.constexpr,
        OUT_IS_FLOAT: tl.constexpr,
    ):
        pid_x = tl.program_id(0)
        pid_y = tl.program_id(1)
        pid_b = tl.program_id(2)

        offs_x = pid_x * BLOCK_W + tl.arange(0, BLOCK_W)
        offs_y = pid_y * BLOCK_H + tl.arange(0, BLOCK_H)
        offs_c = tl.arange(0, CHANNELS)

        src_x = src_roi_x + offs_x[None, :, None]
        src_y = src_roi_y + offs_y[:, None, None]
        dest_x = dest_offset_x + offs_x[None, :, None]
        dest_y = dest_offset_y + offs_y[:, None, None]

        mask = (
            (offs_x[None, :, None] < region_width)
            & (offs_y[:, None, None] < region_height)
            & (src_x >= 0)
            & (src_x < src_width)
            & (src_y >= 0)
            & (src_y < src_height)
            & (dest_x >= 0)
            & (dest_x < dest_width)
            & (dest_y >= 0)
            & (dest_y < dest_height)
        )

        src_ptrs = (
            src_ptr
            + pid_b * src_stride_b
            + src_y * src_stride_h
            + src_x * src_stride_w
            + offs_c[None, None, :] * src_stride_c
        )
        vals = tl.load(src_ptrs, mask=mask, other=0)

        if OUT_IS_FLOAT:
            out = vals.to(tl.float32)
        else:
            out = tl.minimum(tl.maximum(vals.to(tl.float32), 0.0), 255.0)
            out = (out + 0.5).to(tl.uint8)

        dest_ptrs = (
            dest_ptr
            + pid_b * dest_stride_b
            + dest_y * dest_stride_h
            + dest_x * dest_stride_w
            + offs_c[None, None, :] * dest_stride_c
        )
        tl.store(dest_ptrs, out, mask=mask)


    @triton.jit
    def _remap_kernel(
        src_ptr,
        dest_ptr,
        map_x_ptr,
        map_y_ptr,
        src_stride_b,
        src_stride_h,
        src_stride_w,
        src_stride_c,
        dest_stride_b,
        dest_stride_h,
        dest_stride_w,
        dest_stride_c,
        map_stride_h,
        map_stride_w,
        src_height,
        src_width,
        dest_height,
        dest_width,
        map_height,
        map_width,
        offset_x,
        offset_y,
        roi_x,
        roi_y,
        roi_width,
        roi_height,
        unmapped_value,
        adj0,
        adj1,
        adj2,
        BLOCK_W: tl.constexpr,
        BLOCK_H: tl.constexpr,
        CHANNELS: tl.constexpr,
        OUT_IS_FLOAT: tl.constexpr,
        HAS_ADJUSTMENT: tl.constexpr,
        NO_UNMAPPED_WRITE: tl.constexpr,
        FILL_INVALID_ALPHA: tl.constexpr,
    ):
        pid_x = tl.program_id(0)
        pid_y = tl.program_id(1)
        pid_b = tl.program_id(2)

        offs_x = pid_x * BLOCK_W + tl.arange(0, BLOCK_W)
        offs_y = pid_y * BLOCK_H + tl.arange(0, BLOCK_H)
        offs_c = tl.arange(0, CHANNELS)

        map_xpos = roi_x + offs_x[None, :]
        map_ypos = roi_y + offs_y[:, None]
        dest_x = offset_x + map_xpos
        dest_y = offset_y + map_ypos

        base_mask = (
            (offs_x[None, :] < roi_width)
            & (offs_y[:, None] < roi_height)
            & (map_xpos >= 0)
            & (map_xpos < map_width)
            & (map_ypos >= 0)
            & (map_ypos < map_height)
            & (dest_x >= 0)
            & (dest_x < dest_width)
            & (dest_y >= 0)
            & (dest_y < dest_height)
        )

        map_x_ptrs = map_x_ptr + map_ypos * map_stride_h + map_xpos * map_stride_w
        map_y_ptrs = map_y_ptr + map_ypos * map_stride_h + map_xpos * map_stride_w
        mx = tl.load(map_x_ptrs, mask=base_mask, other=unmapped_value).to(tl.int32)
        my = tl.load(map_y_ptrs, mask=base_mask, other=unmapped_value).to(tl.int32)

        unmapped = (mx == unmapped_value) | (my == unmapped_value)
        valid_xy = base_mask & (~unmapped) & (mx >= 0) & (mx < src_width) & (my >= 0) & (my < src_height)
        valid_src = valid_xy[:, :, None]
        store_mask = (base_mask & (~unmapped))[:, :, None] if NO_UNMAPPED_WRITE else base_mask[:, :, None]

        src_ptrs = (
            src_ptr
            + pid_b * src_stride_b
            + my[:, :, None] * src_stride_h
            + mx[:, :, None] * src_stride_w
            + offs_c[None, None, :] * src_stride_c
        )
        vals = tl.load(src_ptrs, mask=valid_src, other=0)
        vals_f = vals.to(tl.float32)

        if HAS_ADJUSTMENT:
            adj = tl.where(
                offs_c[None, None, :] == 0,
                adj0,
                tl.where(offs_c[None, None, :] == 1, adj1, tl.where(offs_c[None, None, :] == 2, adj2, 0.0)),
            )
            vals_f = vals_f + adj

        vals_f = tl.where(valid_src, vals_f, 0.0)
        if CHANNELS == 4 and FILL_INVALID_ALPHA:
            alpha_default = tl.where(unmapped, 0.0, 255.0)[:, :, None]
            alpha_mask = (~valid_xy)[:, :, None] & (offs_c[None, None, :] == 3)
            vals_f = tl.where(alpha_mask, alpha_default, vals_f)

        if OUT_IS_FLOAT:
            out = vals_f
        else:
            out = tl.minimum(tl.maximum(vals_f, 0.0), 255.0)
            out = (out + 0.5).to(tl.uint8)

        dest_ptrs = (
            dest_ptr
            + pid_b * dest_stride_b
            + dest_y[:, :, None] * dest_stride_h
            + dest_x[:, :, None] * dest_stride_w
            + offs_c[None, None, :] * dest_stride_c
        )
        tl.store(dest_ptrs, out, mask=store_mask)


    @triton.jit
    def _remap_with_dest_map_kernel(
        src_ptr,
        dest_ptr,
        map_x_ptr,
        map_y_ptr,
        dest_map_ptr,
        src_stride_b,
        src_stride_h,
        src_stride_w,
        src_stride_c,
        dest_stride_b,
        dest_stride_h,
        dest_stride_w,
        dest_stride_c,
        map_stride_h,
        map_stride_w,
        dest_map_stride_h,
        dest_map_stride_w,
        src_height,
        src_width,
        dest_height,
        dest_width,
        map_height,
        map_width,
        offset_x,
        offset_y,
        roi_x,
        roi_y,
        roi_width,
        roi_height,
        image_index,
        unmapped_value,
        adj0,
        adj1,
        adj2,
        BLOCK_W: tl.constexpr,
        BLOCK_H: tl.constexpr,
        CHANNELS: tl.constexpr,
        OUT_IS_FLOAT: tl.constexpr,
        HAS_ADJUSTMENT: tl.constexpr,
    ):
        pid_x = tl.program_id(0)
        pid_y = tl.program_id(1)
        pid_b = tl.program_id(2)

        offs_x = pid_x * BLOCK_W + tl.arange(0, BLOCK_W)
        offs_y = pid_y * BLOCK_H + tl.arange(0, BLOCK_H)
        offs_c = tl.arange(0, CHANNELS)

        map_xpos = roi_x + offs_x[None, :]
        map_ypos = roi_y + offs_y[:, None]
        dest_x = offset_x + map_xpos
        dest_y = offset_y + map_ypos

        base_mask = (
            (offs_x[None, :] < roi_width)
            & (offs_y[:, None] < roi_height)
            & (map_xpos >= 0)
            & (map_xpos < map_width)
            & (map_ypos >= 0)
            & (map_ypos < map_height)
            & (dest_x >= 0)
            & (dest_x < dest_width)
            & (dest_y >= 0)
            & (dest_y < dest_height)
        )

        dest_map_ptrs = dest_map_ptr + dest_y * dest_map_stride_h + dest_x * dest_map_stride_w
        class_mask = tl.load(dest_map_ptrs, mask=base_mask, other=-1).to(tl.int32) == image_index
        active_xy = base_mask & class_mask

        map_x_ptrs = map_x_ptr + map_ypos * map_stride_h + map_xpos * map_stride_w
        map_y_ptrs = map_y_ptr + map_ypos * map_stride_h + map_xpos * map_stride_w
        mx = tl.load(map_x_ptrs, mask=active_xy, other=unmapped_value).to(tl.int32)
        my = tl.load(map_y_ptrs, mask=active_xy, other=unmapped_value).to(tl.int32)
        valid_xy = active_xy & (mx != unmapped_value) & (my != unmapped_value) & (mx >= 0) & (mx < src_width) & (my >= 0) & (my < src_height)
        valid_src = valid_xy[:, :, None]

        src_ptrs = (
            src_ptr
            + pid_b * src_stride_b
            + my[:, :, None] * src_stride_h
            + mx[:, :, None] * src_stride_w
            + offs_c[None, None, :] * src_stride_c
        )
        vals = tl.load(src_ptrs, mask=valid_src, other=0)
        vals_f = tl.where(valid_src, vals.to(tl.float32), 0.0)

        if HAS_ADJUSTMENT:
            adj = tl.where(
                offs_c[None, None, :] == 0,
                adj0,
                tl.where(offs_c[None, None, :] == 1, adj1, tl.where(offs_c[None, None, :] == 2, adj2, 0.0)),
            )
            vals_f = vals_f + adj

        if OUT_IS_FLOAT:
            out = vals_f
        else:
            out = tl.minimum(tl.maximum(vals_f, 0.0), 255.0)
            out = (out + 0.5).to(tl.uint8)

        dest_ptrs = (
            dest_ptr
            + pid_b * dest_stride_b
            + dest_y[:, :, None] * dest_stride_h
            + dest_x[:, :, None] * dest_stride_w
            + offs_c[None, None, :] * dest_stride_c
        )
        tl.store(dest_ptrs, out, mask=active_xy[:, :, None])
else:  # pragma: no cover - exercised only when Triton is unavailable
    _copy_roi_kernel = None
    _remap_kernel = None
    _remap_with_dest_map_kernel = None


def _launch_grid(width: int, height: int, batch: int, block_w: int = 32, block_h: int = 8) -> tuple[int, int, int]:
    return triton.cdiv(width, block_w), triton.cdiv(height, block_h), batch


def _out_is_float(tensor: torch.Tensor) -> bool:
    if tensor.dtype == torch.float32:
        return True
    if tensor.dtype == torch.uint8:
        return False
    raise TypeError(f"Unsupported Triton output dtype: {tensor.dtype}")


def copy_roi_triton(src: torch.Tensor, dest: torch.Tensor, region: Rect, src_roi_x: int, src_roi_y: int, offset_x: int, offset_y: int) -> torch.Tensor:
    if region.width <= 0 or region.height <= 0:
        return dest
    if not can_use_triton_backend(src, dest):
        raise ValueError("Triton copy requested with unsupported tensors")
    channels = src.shape[-1]
    grid = _launch_grid(region.width, region.height, src.shape[0])
    _copy_roi_kernel[grid](
        src,
        dest,
        *src.stride(),
        *dest.stride(),
        src.shape[1],
        src.shape[2],
        dest.shape[1],
        dest.shape[2],
        src_roi_x,
        src_roi_y,
        offset_x,
        offset_y,
        region.width,
        region.height,
        BLOCK_W=32,
        BLOCK_H=8,
        CHANNELS=channels,
        OUT_IS_FLOAT=_out_is_float(dest),
    )
    return dest


def remap_to_canvas_triton(
    src: torch.Tensor,
    dest: torch.Tensor,
    map_x: torch.Tensor,
    map_y: torch.Tensor,
    offset_x: int,
    offset_y: int,
    *,
    adjustment: torch.Tensor | None = None,
    no_unmapped_write: bool = False,
    roi: Rect | None = None,
    fill_invalid_alpha: bool = True,
) -> torch.Tensor:
    if roi is None:
        roi = Rect(0, 0, int(map_x.shape[1]), int(map_x.shape[0]))
    if roi.empty:
        return dest
    if not can_use_triton_backend(src, dest):
        raise ValueError("Triton remap requested with unsupported source/destination tensors")
    if not (map_x.is_cuda and map_y.is_cuda and map_x.is_contiguous() and map_y.is_contiguous()):
        raise ValueError("Triton remap requires contiguous GPU map tensors")
    channels = src.shape[-1]
    grid = _launch_grid(roi.width, roi.height, src.shape[0])
    adj = adjustment.to(device=src.device, dtype=torch.float32) if adjustment is not None else None
    _remap_kernel[grid](
        src,
        dest,
        map_x,
        map_y,
        *src.stride(),
        *dest.stride(),
        *map_x.stride(),
        src.shape[1],
        src.shape[2],
        dest.shape[1],
        dest.shape[2],
        map_x.shape[0],
        map_x.shape[1],
        offset_x,
        offset_y,
        roi.x,
        roi.y,
        roi.width,
        roi.height,
        _UNMAPPED_VALUE,
        float(adj[0].item()) if adj is not None else 0.0,
        float(adj[1].item()) if adj is not None else 0.0,
        float(adj[2].item()) if adj is not None else 0.0,
        BLOCK_W=32,
        BLOCK_H=8,
        CHANNELS=channels,
        OUT_IS_FLOAT=_out_is_float(dest),
        HAS_ADJUSTMENT=adj is not None,
        NO_UNMAPPED_WRITE=no_unmapped_write,
        FILL_INVALID_ALPHA=fill_invalid_alpha,
    )
    return dest


def remap_to_canvas_with_dest_map_triton(
    src: torch.Tensor,
    dest: torch.Tensor,
    map_x: torch.Tensor,
    map_y: torch.Tensor,
    image_index: int,
    dest_image_map: torch.Tensor,
    offset_x: int,
    offset_y: int,
    *,
    adjustment: torch.Tensor | None = None,
    roi: Rect | None = None,
) -> torch.Tensor:
    if roi is None:
        roi = Rect(0, 0, int(map_x.shape[1]), int(map_x.shape[0]))
    if roi.empty:
        return dest
    if not can_use_triton_backend(src, dest):
        raise ValueError("Triton remap-with-dest-map requested with unsupported source/destination tensors")
    if not (map_x.is_cuda and map_y.is_cuda and dest_image_map.is_cuda):
        raise ValueError("Triton remap-with-dest-map requires GPU-resident maps")
    if not (map_x.is_contiguous() and map_y.is_contiguous() and dest_image_map.is_contiguous()):
        raise ValueError("Triton remap-with-dest-map requires contiguous maps")
    channels = src.shape[-1]
    grid = _launch_grid(roi.width, roi.height, src.shape[0])
    adj = adjustment.to(device=src.device, dtype=torch.float32) if adjustment is not None else None
    _remap_with_dest_map_kernel[grid](
        src,
        dest,
        map_x,
        map_y,
        dest_image_map,
        *src.stride(),
        *dest.stride(),
        *map_x.stride(),
        *dest_image_map.stride(),
        src.shape[1],
        src.shape[2],
        dest.shape[1],
        dest.shape[2],
        map_x.shape[0],
        map_x.shape[1],
        offset_x,
        offset_y,
        roi.x,
        roi.y,
        roi.width,
        roi.height,
        image_index,
        _UNMAPPED_VALUE,
        float(adj[0].item()) if adj is not None else 0.0,
        float(adj[1].item()) if adj is not None else 0.0,
        float(adj[2].item()) if adj is not None else 0.0,
        BLOCK_W=32,
        BLOCK_H=8,
        CHANNELS=channels,
        OUT_IS_FLOAT=_out_is_float(dest),
        HAS_ADJUSTMENT=adj is not None,
    )
    return dest
