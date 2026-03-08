from __future__ import annotations

from typing import Final, Sequence

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
_SUPPORTED_BLEND_INPUTS: Final[set[int]] = set(range(2, 9))
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


def _supported_blend_image_tensor(tensor: torch.Tensor) -> bool:
    return tensor.dtype == torch.float32 and tensor.ndim == 4 and tensor.shape[-1] in _SUPPORTED_CHANNELS and tensor.is_contiguous()


def _supported_blend_mask_tensor(mask: torch.Tensor, n_inputs: int) -> bool:
    return (
        mask.dtype == torch.float32
        and mask.ndim == 3
        and mask.shape[-1] == n_inputs
        and n_inputs in _SUPPORTED_BLEND_INPUTS
        and mask.is_contiguous()
    )


def can_use_triton_blend_backend(images: Sequence[torch.Tensor], mask: torch.Tensor) -> bool:
    if not _HAS_TRITON or not images:
        return False
    n_inputs = len(images)
    if not _supported_blend_mask_tensor(mask, n_inputs) or not mask.is_cuda:
        return False
    ref = images[0]
    if not ref.is_cuda or not _supported_blend_image_tensor(ref):
        return False
    shape = tuple(ref.shape)
    return all(img.is_cuda and _supported_blend_image_tensor(img) and tuple(img.shape) == shape for img in images)


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


    @triton.jit
    def _downsample_mask_kernel(
        mask_ptr,
        out_ptr,
        mask_stride_h,
        mask_stride_w,
        mask_stride_c,
        out_stride_h,
        out_stride_w,
        out_stride_c,
        in_height,
        in_width,
        out_height,
        out_width,
        BLOCK_W: tl.constexpr,
        BLOCK_H: tl.constexpr,
        CHANNELS: tl.constexpr,
    ):
        pid_x = tl.program_id(0)
        pid_y = tl.program_id(1)

        offs_x = pid_x * BLOCK_W + tl.arange(0, BLOCK_W)
        offs_y = pid_y * BLOCK_H + tl.arange(0, BLOCK_H)
        offs_c = tl.arange(0, CHANNELS)

        out_x = offs_x[None, :, None]
        out_y = offs_y[:, None, None]
        out_mask = (offs_x[None, :, None] < out_width) & (offs_y[:, None, None] < out_height)
        base_in_x = 2 * out_x
        base_in_y = 2 * out_y

        acc = tl.zeros((BLOCK_H, BLOCK_W, CHANNELS), dtype=tl.float32)
        count = tl.zeros((BLOCK_H, BLOCK_W, 1), dtype=tl.float32)

        for dy in range(2):
            for dx in range(2):
                in_x = base_in_x + dx
                in_y = base_in_y + dy
                tap_mask = out_mask & (in_x < in_width) & (in_y < in_height)
                ptrs = mask_ptr + in_y * mask_stride_h + in_x * mask_stride_w + offs_c[None, None, :] * mask_stride_c
                vals = tl.load(ptrs, mask=tap_mask, other=0.0)
                acc += vals
                count += tl.where(tap_mask, 1.0, 0.0)

        out_ptrs = out_ptr + out_y * out_stride_h + out_x * out_stride_w + offs_c[None, None, :] * out_stride_c
        out_vals = acc / tl.maximum(count, 1.0)
        tl.store(out_ptrs, out_vals, mask=out_mask)


    @triton.jit
    def _downsample_image_kernel(
        image_ptr,
        out_ptr,
        image_stride_b,
        image_stride_h,
        image_stride_w,
        image_stride_c,
        out_stride_b,
        out_stride_h,
        out_stride_w,
        out_stride_c,
        in_height,
        in_width,
        out_height,
        out_width,
        BLOCK_W: tl.constexpr,
        BLOCK_H: tl.constexpr,
        CHANNELS: tl.constexpr,
    ):
        pid_x = tl.program_id(0)
        pid_y = tl.program_id(1)
        pid_b = tl.program_id(2)

        offs_x = pid_x * BLOCK_W + tl.arange(0, BLOCK_W)
        offs_y = pid_y * BLOCK_H + tl.arange(0, BLOCK_H)
        out_x = offs_x[None, :]
        out_y = offs_y[:, None]
        out_mask = (offs_x[None, :] < out_width) & (offs_y[:, None] < out_height)
        base_in_x = 2 * out_x
        base_in_y = 2 * out_y

        if CHANNELS == 4:
            offs_rgb = tl.arange(0, 4)
            rgb_mask = offs_rgb[None, None, :] < 3
            sum_rgb = tl.zeros((BLOCK_H, BLOCK_W, 4), dtype=tl.float32)
            count = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
            alpha_max = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
            for dy in range(2):
                for dx in range(2):
                    in_x = base_in_x + dx
                    in_y = base_in_y + dy
                    tap_mask = out_mask & (in_x < in_width) & (in_y < in_height)
                    rgb_ptrs = (
                        image_ptr
                        + pid_b * image_stride_b
                        + in_y[:, :, None] * image_stride_h
                        + in_x[:, :, None] * image_stride_w
                        + offs_rgb[None, None, :] * image_stride_c
                    )
                    alpha_ptrs = (
                        image_ptr
                        + pid_b * image_stride_b
                        + in_y * image_stride_h
                        + in_x * image_stride_w
                        + 3 * image_stride_c
                    )
                    rgb = tl.load(rgb_ptrs, mask=tap_mask[:, :, None] & rgb_mask, other=0.0)
                    alpha = tl.load(alpha_ptrs, mask=tap_mask, other=0.0)
                    keep = tap_mask & (alpha != 0.0)
                    sum_rgb += tl.where(rgb_mask, tl.where(keep[:, :, None], rgb, 0.0), 0.0)
                    count += tl.where(keep, 1.0, 0.0)
                    alpha_max = tl.maximum(alpha_max, tl.where(tap_mask, alpha, 0.0))

            out_rgb_ptrs = (
                out_ptr
                + pid_b * out_stride_b
                + out_y[:, :, None] * out_stride_h
                + out_x[:, :, None] * out_stride_w
                + offs_rgb[None, None, :] * out_stride_c
            )
            out_alpha_ptrs = out_ptr + pid_b * out_stride_b + out_y * out_stride_h + out_x * out_stride_w + 3 * out_stride_c
            out_rgb = sum_rgb / tl.maximum(count[:, :, None], 1.0)
            tl.store(out_rgb_ptrs, out_rgb, mask=out_mask[:, :, None] & rgb_mask)
            tl.store(out_alpha_ptrs, alpha_max, mask=out_mask)
            return

        offs_c = tl.arange(0, CHANNELS)
        acc = tl.zeros((BLOCK_H, BLOCK_W, CHANNELS), dtype=tl.float32)
        count = tl.zeros((BLOCK_H, BLOCK_W, 1), dtype=tl.float32)
        for dy in range(2):
            for dx in range(2):
                in_x = base_in_x + dx
                in_y = base_in_y + dy
                tap_mask = out_mask & (in_x < in_width) & (in_y < in_height)
                ptrs = (
                    image_ptr
                    + pid_b * image_stride_b
                    + in_y[:, :, None] * image_stride_h
                    + in_x[:, :, None] * image_stride_w
                    + offs_c[None, None, :] * image_stride_c
                )
                vals = tl.load(ptrs, mask=tap_mask[:, :, None], other=0.0)
                acc += vals
                count += tl.where(tap_mask, 1.0, 0.0)[:, :, None]
        out_ptrs = (
            out_ptr
            + pid_b * out_stride_b
            + out_y[:, :, None] * out_stride_h
            + out_x[:, :, None] * out_stride_w
            + offs_c[None, None, :] * out_stride_c
        )
        tl.store(out_ptrs, acc / tl.maximum(count, 1.0), mask=out_mask[:, :, None])


    @triton.jit
    def _compute_laplacian_kernel(
        high_ptr,
        low_ptr,
        out_ptr,
        high_stride_b,
        high_stride_h,
        high_stride_w,
        high_stride_c,
        low_stride_b,
        low_stride_h,
        low_stride_w,
        low_stride_c,
        out_stride_b,
        out_stride_h,
        out_stride_w,
        out_stride_c,
        high_height,
        high_width,
        low_height,
        low_width,
        BLOCK_W: tl.constexpr,
        BLOCK_H: tl.constexpr,
        CHANNELS: tl.constexpr,
    ):
        pid_x = tl.program_id(0)
        pid_y = tl.program_id(1)
        pid_b = tl.program_id(2)

        offs_x = pid_x * BLOCK_W + tl.arange(0, BLOCK_W)
        offs_y = pid_y * BLOCK_H + tl.arange(0, BLOCK_H)
        out_x = offs_x[None, :]
        out_y = offs_y[:, None]
        out_mask = (offs_x[None, :] < high_width) & (offs_y[:, None] < high_height)

        x0 = out_x // 2
        y0 = out_y // 2
        x1 = tl.minimum(x0 + 1, low_width - 1)
        y1 = tl.minimum(y0 + 1, low_height - 1)

        w10 = tl.where((out_x & 1) != 0, 0.5, 0.0)
        w01 = tl.where((out_y & 1) != 0, 0.5, 0.0)
        w00 = (1.0 - w10) * (1.0 - w01)
        w10 = w10 * (1.0 - w01)
        w01 = (1.0 - tl.where((out_x & 1) != 0, 0.5, 0.0)) * w01
        w11 = tl.where((out_x & 1) != 0, 0.5, 0.0) * tl.where((out_y & 1) != 0, 0.5, 0.0)

        if CHANNELS == 4:
            offs_rgba = tl.arange(0, 4)
            rgb_mask = offs_rgba[None, None, :] < 3
            high_ptrs = (
                high_ptr
                + pid_b * high_stride_b
                + out_y[:, :, None] * high_stride_h
                + out_x[:, :, None] * high_stride_w
                + offs_rgba[None, None, :] * high_stride_c
            )
            high_vals = tl.load(high_ptrs, mask=out_mask[:, :, None], other=0.0)
            high_alpha_ptrs = high_ptr + pid_b * high_stride_b + out_y * high_stride_h + out_x * high_stride_w + 3 * high_stride_c
            high_alpha = tl.load(high_alpha_ptrs, mask=out_mask, other=0.0)

            sum_rgb = tl.zeros((BLOCK_H, BLOCK_W, 4), dtype=tl.float32)
            sum_w = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)

            p00_ptrs = (
                low_ptr
                + pid_b * low_stride_b
                + y0[:, :, None] * low_stride_h
                + x0[:, :, None] * low_stride_w
                + offs_rgba[None, None, :] * low_stride_c
            )
            p00 = tl.load(p00_ptrs, mask=out_mask[:, :, None], other=0.0)
            p00_alpha_ptrs = low_ptr + pid_b * low_stride_b + y0 * low_stride_h + x0 * low_stride_w + 3 * low_stride_c
            p00_alpha = tl.load(p00_alpha_ptrs, mask=out_mask, other=0.0)
            p00_valid = out_mask & (p00_alpha != 0.0)
            p00_weight = w00 * tl.where(p00_valid, 1.0, 0.0)
            sum_rgb += tl.where(rgb_mask, p00 * p00_weight[:, :, None], 0.0)
            sum_w += p00_weight

            p10_ptrs = (
                low_ptr
                + pid_b * low_stride_b
                + y0[:, :, None] * low_stride_h
                + x1[:, :, None] * low_stride_w
                + offs_rgba[None, None, :] * low_stride_c
            )
            p10 = tl.load(p10_ptrs, mask=out_mask[:, :, None], other=0.0)
            p10_alpha_ptrs = low_ptr + pid_b * low_stride_b + y0 * low_stride_h + x1 * low_stride_w + 3 * low_stride_c
            p10_alpha = tl.load(p10_alpha_ptrs, mask=out_mask, other=0.0)
            p10_valid = out_mask & (p10_alpha != 0.0)
            p10_weight = w10 * tl.where(p10_valid, 1.0, 0.0)
            sum_rgb += tl.where(rgb_mask, p10 * p10_weight[:, :, None], 0.0)
            sum_w += p10_weight

            p01_ptrs = (
                low_ptr
                + pid_b * low_stride_b
                + y1[:, :, None] * low_stride_h
                + x0[:, :, None] * low_stride_w
                + offs_rgba[None, None, :] * low_stride_c
            )
            p01 = tl.load(p01_ptrs, mask=out_mask[:, :, None], other=0.0)
            p01_alpha_ptrs = low_ptr + pid_b * low_stride_b + y1 * low_stride_h + x0 * low_stride_w + 3 * low_stride_c
            p01_alpha = tl.load(p01_alpha_ptrs, mask=out_mask, other=0.0)
            p01_valid = out_mask & (p01_alpha != 0.0)
            p01_weight = w01 * tl.where(p01_valid, 1.0, 0.0)
            sum_rgb += tl.where(rgb_mask, p01 * p01_weight[:, :, None], 0.0)
            sum_w += p01_weight

            p11_ptrs = (
                low_ptr
                + pid_b * low_stride_b
                + y1[:, :, None] * low_stride_h
                + x1[:, :, None] * low_stride_w
                + offs_rgba[None, None, :] * low_stride_c
            )
            p11 = tl.load(p11_ptrs, mask=out_mask[:, :, None], other=0.0)
            p11_alpha_ptrs = low_ptr + pid_b * low_stride_b + y1 * low_stride_h + x1 * low_stride_w + 3 * low_stride_c
            p11_alpha = tl.load(p11_alpha_ptrs, mask=out_mask, other=0.0)
            p11_valid = out_mask & (p11_alpha != 0.0)
            p11_weight = w11 * tl.where(p11_valid, 1.0, 0.0)
            sum_rgb += tl.where(rgb_mask, p11 * p11_weight[:, :, None], 0.0)
            sum_w += p11_weight

            up_rgb = sum_rgb / tl.maximum(sum_w[:, :, None], 1.0)
            out_vals = tl.where(rgb_mask, high_vals - up_rgb, 0.0)
            out_vals = tl.where(offs_rgba[None, None, :] == 3, high_alpha[:, :, None], out_vals)
            out_ptrs = (
                out_ptr
                + pid_b * out_stride_b
                + out_y[:, :, None] * out_stride_h
                + out_x[:, :, None] * out_stride_w
                + offs_rgba[None, None, :] * out_stride_c
            )
            tl.store(out_ptrs, out_vals, mask=out_mask[:, :, None])
            return

        offs_c = tl.arange(0, 4)
        channel_mask = offs_c[None, None, :] < CHANNELS
        high_ptrs = (
            high_ptr
            + pid_b * high_stride_b
            + out_y[:, :, None] * high_stride_h
            + out_x[:, :, None] * high_stride_w
            + offs_c[None, None, :] * high_stride_c
        )
        high_vals = tl.load(high_ptrs, mask=out_mask[:, :, None] & channel_mask, other=0.0)
        up = tl.zeros((BLOCK_H, BLOCK_W, 4), dtype=tl.float32)

        p00_ptrs = (
            low_ptr
            + pid_b * low_stride_b
            + y0[:, :, None] * low_stride_h
            + x0[:, :, None] * low_stride_w
            + offs_c[None, None, :] * low_stride_c
        )
        p00 = tl.load(p00_ptrs, mask=out_mask[:, :, None] & channel_mask, other=0.0)
        up += p00 * w00[:, :, None]

        p10_ptrs = (
            low_ptr
            + pid_b * low_stride_b
            + y0[:, :, None] * low_stride_h
            + x1[:, :, None] * low_stride_w
            + offs_c[None, None, :] * low_stride_c
        )
        p10 = tl.load(p10_ptrs, mask=out_mask[:, :, None] & channel_mask, other=0.0)
        up += p10 * w10[:, :, None]

        p01_ptrs = (
            low_ptr
            + pid_b * low_stride_b
            + y1[:, :, None] * low_stride_h
            + x0[:, :, None] * low_stride_w
            + offs_c[None, None, :] * low_stride_c
        )
        p01 = tl.load(p01_ptrs, mask=out_mask[:, :, None] & channel_mask, other=0.0)
        up += p01 * w01[:, :, None]

        p11_ptrs = (
            low_ptr
            + pid_b * low_stride_b
            + y1[:, :, None] * low_stride_h
            + x1[:, :, None] * low_stride_w
            + offs_c[None, None, :] * low_stride_c
        )
        p11 = tl.load(p11_ptrs, mask=out_mask[:, :, None] & channel_mask, other=0.0)
        up += p11 * w11[:, :, None]
        out_ptrs = (
            out_ptr
            + pid_b * out_stride_b
            + out_y[:, :, None] * out_stride_h
            + out_x[:, :, None] * out_stride_w
            + offs_c[None, None, :] * out_stride_c
        )
        tl.store(out_ptrs, high_vals - up, mask=out_mask[:, :, None] & channel_mask)


    @triton.jit
    def _reconstruct_level_kernel(
        low_ptr,
        lap_ptr,
        out_ptr,
        low_stride_b,
        low_stride_h,
        low_stride_w,
        low_stride_c,
        lap_stride_b,
        lap_stride_h,
        lap_stride_w,
        lap_stride_c,
        out_stride_b,
        out_stride_h,
        out_stride_w,
        out_stride_c,
        high_height,
        high_width,
        low_height,
        low_width,
        BLOCK_W: tl.constexpr,
        BLOCK_H: tl.constexpr,
        CHANNELS: tl.constexpr,
    ):
        pid_x = tl.program_id(0)
        pid_y = tl.program_id(1)
        pid_b = tl.program_id(2)

        offs_x = pid_x * BLOCK_W + tl.arange(0, BLOCK_W)
        offs_y = pid_y * BLOCK_H + tl.arange(0, BLOCK_H)
        out_x = offs_x[None, :]
        out_y = offs_y[:, None]
        out_mask = (offs_x[None, :] < high_width) & (offs_y[:, None] < high_height)

        x0 = out_x // 2
        y0 = out_y // 2
        x1 = tl.minimum(x0 + 1, low_width - 1)
        y1 = tl.minimum(y0 + 1, low_height - 1)

        dx = tl.where((out_x & 1) != 0, 0.5, 0.0)
        dy = tl.where((out_y & 1) != 0, 0.5, 0.0)
        w00 = (1.0 - dx) * (1.0 - dy)
        w10 = dx * (1.0 - dy)
        w01 = (1.0 - dx) * dy
        w11 = dx * dy

        if CHANNELS == 4:
            offs_rgba = tl.arange(0, 4)
            rgb_mask = offs_rgba[None, None, :] < 3
            sum_rgb = tl.zeros((BLOCK_H, BLOCK_W, 4), dtype=tl.float32)
            sum_w = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)

            p00_ptrs = (
                low_ptr
                + pid_b * low_stride_b
                + y0[:, :, None] * low_stride_h
                + x0[:, :, None] * low_stride_w
                + offs_rgba[None, None, :] * low_stride_c
            )
            p00 = tl.load(p00_ptrs, mask=out_mask[:, :, None], other=0.0)
            p00_alpha_ptrs = low_ptr + pid_b * low_stride_b + y0 * low_stride_h + x0 * low_stride_w + 3 * low_stride_c
            p00_alpha = tl.load(p00_alpha_ptrs, mask=out_mask, other=0.0)
            p00_valid = out_mask & (p00_alpha != 0.0)
            p00_weight = w00 * tl.where(p00_valid, 1.0, 0.0)
            sum_rgb += tl.where(rgb_mask, p00 * p00_weight[:, :, None], 0.0)
            sum_w += p00_weight

            p10_ptrs = (
                low_ptr
                + pid_b * low_stride_b
                + y0[:, :, None] * low_stride_h
                + x1[:, :, None] * low_stride_w
                + offs_rgba[None, None, :] * low_stride_c
            )
            p10 = tl.load(p10_ptrs, mask=out_mask[:, :, None], other=0.0)
            p10_alpha_ptrs = low_ptr + pid_b * low_stride_b + y0 * low_stride_h + x1 * low_stride_w + 3 * low_stride_c
            p10_alpha = tl.load(p10_alpha_ptrs, mask=out_mask, other=0.0)
            p10_valid = out_mask & (p10_alpha != 0.0)
            p10_weight = w10 * tl.where(p10_valid, 1.0, 0.0)
            sum_rgb += tl.where(rgb_mask, p10 * p10_weight[:, :, None], 0.0)
            sum_w += p10_weight

            p01_ptrs = (
                low_ptr
                + pid_b * low_stride_b
                + y1[:, :, None] * low_stride_h
                + x0[:, :, None] * low_stride_w
                + offs_rgba[None, None, :] * low_stride_c
            )
            p01 = tl.load(p01_ptrs, mask=out_mask[:, :, None], other=0.0)
            p01_alpha_ptrs = low_ptr + pid_b * low_stride_b + y1 * low_stride_h + x0 * low_stride_w + 3 * low_stride_c
            p01_alpha = tl.load(p01_alpha_ptrs, mask=out_mask, other=0.0)
            p01_valid = out_mask & (p01_alpha != 0.0)
            p01_weight = w01 * tl.where(p01_valid, 1.0, 0.0)
            sum_rgb += tl.where(rgb_mask, p01 * p01_weight[:, :, None], 0.0)
            sum_w += p01_weight

            p11_ptrs = (
                low_ptr
                + pid_b * low_stride_b
                + y1[:, :, None] * low_stride_h
                + x1[:, :, None] * low_stride_w
                + offs_rgba[None, None, :] * low_stride_c
            )
            p11 = tl.load(p11_ptrs, mask=out_mask[:, :, None], other=0.0)
            p11_alpha_ptrs = low_ptr + pid_b * low_stride_b + y1 * low_stride_h + x1 * low_stride_w + 3 * low_stride_c
            p11_alpha = tl.load(p11_alpha_ptrs, mask=out_mask, other=0.0)
            p11_valid = out_mask & (p11_alpha != 0.0)
            p11_weight = w11 * tl.where(p11_valid, 1.0, 0.0)
            sum_rgb += tl.where(rgb_mask, p11 * p11_weight[:, :, None], 0.0)
            sum_w += p11_weight

            lap_ptrs = (
                lap_ptr
                + pid_b * lap_stride_b
                + out_y[:, :, None] * lap_stride_h
                + out_x[:, :, None] * lap_stride_w
                + offs_rgba[None, None, :] * lap_stride_c
            )
            lap_vals = tl.load(lap_ptrs, mask=out_mask[:, :, None], other=0.0)
            lap_alpha_ptrs = lap_ptr + pid_b * lap_stride_b + out_y * lap_stride_h + out_x * lap_stride_w + 3 * lap_stride_c
            lap_alpha = tl.load(lap_alpha_ptrs, mask=out_mask, other=0.0)
            out_vals = tl.where(rgb_mask, sum_rgb / tl.maximum(sum_w[:, :, None], 1.0) + lap_vals, 0.0)
            out_vals = tl.where(offs_rgba[None, None, :] == 3, lap_alpha[:, :, None], out_vals)
            out_ptrs = (
                out_ptr
                + pid_b * out_stride_b
                + out_y[:, :, None] * out_stride_h
                + out_x[:, :, None] * out_stride_w
                + offs_rgba[None, None, :] * out_stride_c
            )
            tl.store(out_ptrs, out_vals, mask=out_mask[:, :, None])
            return

        offs_c = tl.arange(0, 4)
        channel_mask = offs_c[None, None, :] < CHANNELS
        up = tl.zeros((BLOCK_H, BLOCK_W, 4), dtype=tl.float32)

        p00_ptrs = (
            low_ptr
            + pid_b * low_stride_b
            + y0[:, :, None] * low_stride_h
            + x0[:, :, None] * low_stride_w
            + offs_c[None, None, :] * low_stride_c
        )
        p00 = tl.load(p00_ptrs, mask=out_mask[:, :, None] & channel_mask, other=0.0)
        up += p00 * w00[:, :, None]

        p10_ptrs = (
            low_ptr
            + pid_b * low_stride_b
            + y0[:, :, None] * low_stride_h
            + x1[:, :, None] * low_stride_w
            + offs_c[None, None, :] * low_stride_c
        )
        p10 = tl.load(p10_ptrs, mask=out_mask[:, :, None] & channel_mask, other=0.0)
        up += p10 * w10[:, :, None]

        p01_ptrs = (
            low_ptr
            + pid_b * low_stride_b
            + y1[:, :, None] * low_stride_h
            + x0[:, :, None] * low_stride_w
            + offs_c[None, None, :] * low_stride_c
        )
        p01 = tl.load(p01_ptrs, mask=out_mask[:, :, None] & channel_mask, other=0.0)
        up += p01 * w01[:, :, None]

        p11_ptrs = (
            low_ptr
            + pid_b * low_stride_b
            + y1[:, :, None] * low_stride_h
            + x1[:, :, None] * low_stride_w
            + offs_c[None, None, :] * low_stride_c
        )
        p11 = tl.load(p11_ptrs, mask=out_mask[:, :, None] & channel_mask, other=0.0)
        up += p11 * w11[:, :, None]
        lap_ptrs = (
            lap_ptr
            + pid_b * lap_stride_b
            + out_y[:, :, None] * lap_stride_h
            + out_x[:, :, None] * lap_stride_w
            + offs_c[None, None, :] * lap_stride_c
        )
        lap_vals = tl.load(lap_ptrs, mask=out_mask[:, :, None] & channel_mask, other=0.0)
        out_ptrs = (
            out_ptr
            + pid_b * out_stride_b
            + out_y[:, :, None] * out_stride_h
            + out_x[:, :, None] * out_stride_w
            + offs_c[None, None, :] * out_stride_c
        )
        tl.store(out_ptrs, up + lap_vals, mask=out_mask[:, :, None] & channel_mask)


    @triton.jit
    def _blend_laplacians_n_kernel(
        src0_ptr,
        src1_ptr,
        src2_ptr,
        src3_ptr,
        src4_ptr,
        src5_ptr,
        src6_ptr,
        src7_ptr,
        mask_ptr,
        out_ptr,
        src_stride_b,
        src_stride_h,
        src_stride_w,
        src_stride_c,
        mask_stride_h,
        mask_stride_w,
        mask_stride_c,
        out_stride_b,
        out_stride_h,
        out_stride_w,
        out_stride_c,
        height,
        width,
        BLOCK_W: tl.constexpr,
        BLOCK_H: tl.constexpr,
        CHANNELS: tl.constexpr,
        N_INPUTS: tl.constexpr,
    ):
        pid_x = tl.program_id(0)
        pid_y = tl.program_id(1)
        pid_b = tl.program_id(2)

        offs_x = pid_x * BLOCK_W + tl.arange(0, BLOCK_W)
        offs_y = pid_y * BLOCK_H + tl.arange(0, BLOCK_H)
        out_x = offs_x[None, :]
        out_y = offs_y[:, None]
        out_mask = (offs_x[None, :] < width) & (offs_y[:, None] < height)

        src_ptrs = (src0_ptr, src1_ptr, src2_ptr, src3_ptr, src4_ptr, src5_ptr, src6_ptr, src7_ptr)

        if CHANNELS == 4:
            offs_rgb = tl.arange(0, 4)
            rgb_mask = offs_rgb[None, None, :] < 3
            rgb_acc = tl.zeros((BLOCK_H, BLOCK_W, 4), dtype=tl.float32)
            valid_sum = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
            fallback_rgb = tl.zeros((BLOCK_H, BLOCK_W, 4), dtype=tl.float32)
            fallback_alpha = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
            raw_sum = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
            w0 = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
            w1 = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
            w2 = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
            w3 = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
            w4 = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
            w5 = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
            w6 = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
            w7 = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
            if N_INPUTS > 0:
                w0 = tl.load(mask_ptr + out_y * mask_stride_h + out_x * mask_stride_w + 0 * mask_stride_c, mask=out_mask, other=0.0)
                raw_sum += w0
            if N_INPUTS > 1:
                w1 = tl.load(mask_ptr + out_y * mask_stride_h + out_x * mask_stride_w + 1 * mask_stride_c, mask=out_mask, other=0.0)
                raw_sum += w1
            if N_INPUTS > 2:
                w2 = tl.load(mask_ptr + out_y * mask_stride_h + out_x * mask_stride_w + 2 * mask_stride_c, mask=out_mask, other=0.0)
                raw_sum += w2
            if N_INPUTS > 3:
                w3 = tl.load(mask_ptr + out_y * mask_stride_h + out_x * mask_stride_w + 3 * mask_stride_c, mask=out_mask, other=0.0)
                raw_sum += w3
            if N_INPUTS > 4:
                w4 = tl.load(mask_ptr + out_y * mask_stride_h + out_x * mask_stride_w + 4 * mask_stride_c, mask=out_mask, other=0.0)
                raw_sum += w4
            if N_INPUTS > 5:
                w5 = tl.load(mask_ptr + out_y * mask_stride_h + out_x * mask_stride_w + 5 * mask_stride_c, mask=out_mask, other=0.0)
                raw_sum += w5
            if N_INPUTS > 6:
                w6 = tl.load(mask_ptr + out_y * mask_stride_h + out_x * mask_stride_w + 6 * mask_stride_c, mask=out_mask, other=0.0)
                raw_sum += w6
            if N_INPUTS > 7:
                w7 = tl.load(mask_ptr + out_y * mask_stride_h + out_x * mask_stride_w + 7 * mask_stride_c, mask=out_mask, other=0.0)
                raw_sum += w7

            for i in range(8):
                if i < N_INPUTS:
                    weight_src = w0 if i == 0 else w1 if i == 1 else w2 if i == 2 else w3 if i == 3 else w4 if i == 4 else w5 if i == 5 else w6 if i == 6 else w7
                    weight = tl.where(raw_sum > 0.0, weight_src / raw_sum, 0.0)
                    rgb_ptrs = (
                        src_ptrs[i]
                        + pid_b * src_stride_b
                        + out_y[:, :, None] * src_stride_h
                        + out_x[:, :, None] * src_stride_w
                        + offs_rgb[None, None, :] * src_stride_c
                    )
                    alpha_ptrs = (
                        src_ptrs[i]
                        + pid_b * src_stride_b
                        + out_y * src_stride_h
                        + out_x * src_stride_w
                        + 3 * src_stride_c
                    )
                    rgb = tl.load(rgb_ptrs, mask=out_mask[:, :, None] & rgb_mask, other=0.0)
                    alpha = tl.load(alpha_ptrs, mask=out_mask, other=0.0)
                    valid = out_mask & (alpha != 0.0)
                    blend_weight = weight * tl.where(valid, 1.0, 0.0)
                    rgb_acc += tl.where(rgb_mask, rgb * blend_weight[:, :, None], 0.0)
                    valid_sum += blend_weight

                    take_fallback = alpha > fallback_alpha
                    fallback_rgb = tl.where(take_fallback[:, :, None] & rgb_mask, rgb, fallback_rgb)
                    fallback_alpha = tl.where(take_fallback, alpha, fallback_alpha)

            out_rgb = rgb_acc / tl.maximum(valid_sum[:, :, None], 1.0)
            out_alpha = fallback_alpha

            offs_all = tl.arange(0, 4)
            out_ptrs = (
                out_ptr
                + pid_b * out_stride_b
                + out_y[:, :, None] * out_stride_h
                + out_x[:, :, None] * out_stride_w
                + offs_all[None, None, :] * out_stride_c
            )
            zeros = tl.zeros((BLOCK_H, BLOCK_W, 4), dtype=tl.float32)
            fallback = tl.zeros((BLOCK_H, BLOCK_W, 4), dtype=tl.float32)
            fallback = tl.where(offs_all[None, None, :] == 3, fallback_alpha[:, :, None], fallback)
            fallback = tl.where(offs_all[None, None, :] < 3, fallback_rgb, fallback)
            blended = tl.zeros((BLOCK_H, BLOCK_W, 4), dtype=tl.float32)
            blended = tl.where(offs_all[None, None, :] == 3, out_alpha[:, :, None], blended)
            blended = tl.where(offs_all[None, None, :] < 3, out_rgb, blended)
            use_blend = valid_sum > 0.0
            has_fallback = fallback_alpha > 0.0
            out_vals = tl.where(use_blend[:, :, None], blended, tl.where(has_fallback[:, :, None], fallback, zeros))
            tl.store(out_ptrs, out_vals, mask=out_mask[:, :, None])
            return

        offs_c = tl.arange(0, 4)
        channel_mask = offs_c[None, None, :] < CHANNELS
        out_vals = tl.zeros((BLOCK_H, BLOCK_W, 4), dtype=tl.float32)
        raw_sum = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
        w0 = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
        w1 = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
        w2 = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
        w3 = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
        w4 = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
        w5 = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
        w6 = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
        w7 = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
        if N_INPUTS > 0:
            w0 = tl.load(mask_ptr + out_y * mask_stride_h + out_x * mask_stride_w + 0 * mask_stride_c, mask=out_mask, other=0.0)
            raw_sum += w0
        if N_INPUTS > 1:
            w1 = tl.load(mask_ptr + out_y * mask_stride_h + out_x * mask_stride_w + 1 * mask_stride_c, mask=out_mask, other=0.0)
            raw_sum += w1
        if N_INPUTS > 2:
            w2 = tl.load(mask_ptr + out_y * mask_stride_h + out_x * mask_stride_w + 2 * mask_stride_c, mask=out_mask, other=0.0)
            raw_sum += w2
        if N_INPUTS > 3:
            w3 = tl.load(mask_ptr + out_y * mask_stride_h + out_x * mask_stride_w + 3 * mask_stride_c, mask=out_mask, other=0.0)
            raw_sum += w3
        if N_INPUTS > 4:
            w4 = tl.load(mask_ptr + out_y * mask_stride_h + out_x * mask_stride_w + 4 * mask_stride_c, mask=out_mask, other=0.0)
            raw_sum += w4
        if N_INPUTS > 5:
            w5 = tl.load(mask_ptr + out_y * mask_stride_h + out_x * mask_stride_w + 5 * mask_stride_c, mask=out_mask, other=0.0)
            raw_sum += w5
        if N_INPUTS > 6:
            w6 = tl.load(mask_ptr + out_y * mask_stride_h + out_x * mask_stride_w + 6 * mask_stride_c, mask=out_mask, other=0.0)
            raw_sum += w6
        if N_INPUTS > 7:
            w7 = tl.load(mask_ptr + out_y * mask_stride_h + out_x * mask_stride_w + 7 * mask_stride_c, mask=out_mask, other=0.0)
            raw_sum += w7

        for i in range(8):
            if i < N_INPUTS:
                weight_src = w0 if i == 0 else w1 if i == 1 else w2 if i == 2 else w3 if i == 3 else w4 if i == 4 else w5 if i == 5 else w6 if i == 6 else w7
                weight = tl.where(raw_sum > 0.0, weight_src / raw_sum, 0.0)
                src_ptrs_i = (
                    src_ptrs[i]
                    + pid_b * src_stride_b
                    + out_y[:, :, None] * src_stride_h
                    + out_x[:, :, None] * src_stride_w
                    + offs_c[None, None, :] * src_stride_c
                )
                vals = tl.load(src_ptrs_i, mask=out_mask[:, :, None] & channel_mask, other=0.0)
                out_vals += vals * weight[:, :, None]

        dest_ptrs = (
            out_ptr
            + pid_b * out_stride_b
            + out_y[:, :, None] * out_stride_h
            + out_x[:, :, None] * out_stride_w
            + offs_c[None, None, :] * out_stride_c
        )
        tl.store(dest_ptrs, out_vals, mask=out_mask[:, :, None] & channel_mask)


    @triton.jit
    def _blend_laplacians_stacked_kernel(
        stack_ptr,
        mask_ptr,
        out_ptr,
        stack_stride_b,
        stack_stride_h,
        stack_stride_w,
        stack_stride_n,
        stack_stride_c,
        mask_stride_h,
        mask_stride_w,
        mask_stride_c,
        out_stride_b,
        out_stride_h,
        out_stride_w,
        out_stride_c,
        height,
        width,
        BLOCK_W: tl.constexpr,
        BLOCK_H: tl.constexpr,
        CHANNELS: tl.constexpr,
        N_INPUTS: tl.constexpr,
    ):
        pid_x = tl.program_id(0)
        pid_y = tl.program_id(1)
        pid_b = tl.program_id(2)

        offs_x = pid_x * BLOCK_W + tl.arange(0, BLOCK_W)
        offs_y = pid_y * BLOCK_H + tl.arange(0, BLOCK_H)
        out_x = offs_x[None, :]
        out_y = offs_y[:, None]
        out_mask = (offs_x[None, :] < width) & (offs_y[:, None] < height)

        offs_c = tl.arange(0, 4)
        channel_mask = offs_c[None, None, :] < CHANNELS
        rgb_mask = offs_c[None, None, :] < 3

        raw_sum = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
        w0 = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
        w1 = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
        w2 = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
        w3 = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
        w4 = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
        w5 = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
        w6 = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
        w7 = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
        if N_INPUTS > 0:
            w0 = tl.load(mask_ptr + out_y * mask_stride_h + out_x * mask_stride_w + 0 * mask_stride_c, mask=out_mask, other=0.0)
            raw_sum += w0
        if N_INPUTS > 1:
            w1 = tl.load(mask_ptr + out_y * mask_stride_h + out_x * mask_stride_w + 1 * mask_stride_c, mask=out_mask, other=0.0)
            raw_sum += w1
        if N_INPUTS > 2:
            w2 = tl.load(mask_ptr + out_y * mask_stride_h + out_x * mask_stride_w + 2 * mask_stride_c, mask=out_mask, other=0.0)
            raw_sum += w2
        if N_INPUTS > 3:
            w3 = tl.load(mask_ptr + out_y * mask_stride_h + out_x * mask_stride_w + 3 * mask_stride_c, mask=out_mask, other=0.0)
            raw_sum += w3
        if N_INPUTS > 4:
            w4 = tl.load(mask_ptr + out_y * mask_stride_h + out_x * mask_stride_w + 4 * mask_stride_c, mask=out_mask, other=0.0)
            raw_sum += w4
        if N_INPUTS > 5:
            w5 = tl.load(mask_ptr + out_y * mask_stride_h + out_x * mask_stride_w + 5 * mask_stride_c, mask=out_mask, other=0.0)
            raw_sum += w5
        if N_INPUTS > 6:
            w6 = tl.load(mask_ptr + out_y * mask_stride_h + out_x * mask_stride_w + 6 * mask_stride_c, mask=out_mask, other=0.0)
            raw_sum += w6
        if N_INPUTS > 7:
            w7 = tl.load(mask_ptr + out_y * mask_stride_h + out_x * mask_stride_w + 7 * mask_stride_c, mask=out_mask, other=0.0)
            raw_sum += w7

        if CHANNELS == 4:
            rgb_acc = tl.zeros((BLOCK_H, BLOCK_W, 4), dtype=tl.float32)
            valid_sum = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
            blend_alpha = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
            fallback_vals = tl.zeros((BLOCK_H, BLOCK_W, 4), dtype=tl.float32)
            fallback_alpha = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)

            if N_INPUTS > 0:
                weight = tl.where(raw_sum > 0.0, w0 / raw_sum, 0.0)
                ptrs = (
                    stack_ptr
                    + pid_b * stack_stride_b
                    + out_y[:, :, None] * stack_stride_h
                    + out_x[:, :, None] * stack_stride_w
                    + 0 * stack_stride_n
                    + offs_c[None, None, :] * stack_stride_c
                )
                vals = tl.load(ptrs, mask=out_mask[:, :, None] & channel_mask, other=0.0)
                alpha_ptrs = (
                    stack_ptr
                    + pid_b * stack_stride_b
                    + out_y * stack_stride_h
                    + out_x * stack_stride_w
                    + 0 * stack_stride_n
                    + 3 * stack_stride_c
                )
                alpha = tl.load(alpha_ptrs, mask=out_mask, other=0.0)
                valid = out_mask & (alpha != 0.0)
                blend_weight = weight * tl.where(valid, 1.0, 0.0)
                rgb_acc += tl.where(rgb_mask, vals * blend_weight[:, :, None], 0.0)
                valid_sum += blend_weight
                blend_alpha = tl.maximum(blend_alpha, tl.where(blend_weight > 0.0, alpha, 0.0))
                take_fallback = alpha > fallback_alpha
                fallback_vals = tl.where(take_fallback[:, :, None] & channel_mask, vals, fallback_vals)
                fallback_alpha = tl.where(take_fallback, alpha, fallback_alpha)

            if N_INPUTS > 1:
                weight = tl.where(raw_sum > 0.0, w1 / raw_sum, 0.0)
                ptrs = (
                    stack_ptr
                    + pid_b * stack_stride_b
                    + out_y[:, :, None] * stack_stride_h
                    + out_x[:, :, None] * stack_stride_w
                    + 1 * stack_stride_n
                    + offs_c[None, None, :] * stack_stride_c
                )
                vals = tl.load(ptrs, mask=out_mask[:, :, None] & channel_mask, other=0.0)
                alpha_ptrs = (
                    stack_ptr
                    + pid_b * stack_stride_b
                    + out_y * stack_stride_h
                    + out_x * stack_stride_w
                    + 1 * stack_stride_n
                    + 3 * stack_stride_c
                )
                alpha = tl.load(alpha_ptrs, mask=out_mask, other=0.0)
                valid = out_mask & (alpha != 0.0)
                blend_weight = weight * tl.where(valid, 1.0, 0.0)
                rgb_acc += tl.where(rgb_mask, vals * blend_weight[:, :, None], 0.0)
                valid_sum += blend_weight
                blend_alpha = tl.maximum(blend_alpha, tl.where(blend_weight > 0.0, alpha, 0.0))
                take_fallback = alpha > fallback_alpha
                fallback_vals = tl.where(take_fallback[:, :, None] & channel_mask, vals, fallback_vals)
                fallback_alpha = tl.where(take_fallback, alpha, fallback_alpha)

            if N_INPUTS > 2:
                weight = tl.where(raw_sum > 0.0, w2 / raw_sum, 0.0)
                ptrs = (
                    stack_ptr
                    + pid_b * stack_stride_b
                    + out_y[:, :, None] * stack_stride_h
                    + out_x[:, :, None] * stack_stride_w
                    + 2 * stack_stride_n
                    + offs_c[None, None, :] * stack_stride_c
                )
                vals = tl.load(ptrs, mask=out_mask[:, :, None] & channel_mask, other=0.0)
                alpha_ptrs = (
                    stack_ptr
                    + pid_b * stack_stride_b
                    + out_y * stack_stride_h
                    + out_x * stack_stride_w
                    + 2 * stack_stride_n
                    + 3 * stack_stride_c
                )
                alpha = tl.load(alpha_ptrs, mask=out_mask, other=0.0)
                valid = out_mask & (alpha != 0.0)
                blend_weight = weight * tl.where(valid, 1.0, 0.0)
                rgb_acc += tl.where(rgb_mask, vals * blend_weight[:, :, None], 0.0)
                valid_sum += blend_weight
                blend_alpha = tl.maximum(blend_alpha, tl.where(blend_weight > 0.0, alpha, 0.0))
                take_fallback = alpha > fallback_alpha
                fallback_vals = tl.where(take_fallback[:, :, None] & channel_mask, vals, fallback_vals)
                fallback_alpha = tl.where(take_fallback, alpha, fallback_alpha)

            if N_INPUTS > 3:
                weight = tl.where(raw_sum > 0.0, w3 / raw_sum, 0.0)
                ptrs = (
                    stack_ptr
                    + pid_b * stack_stride_b
                    + out_y[:, :, None] * stack_stride_h
                    + out_x[:, :, None] * stack_stride_w
                    + 3 * stack_stride_n
                    + offs_c[None, None, :] * stack_stride_c
                )
                vals = tl.load(ptrs, mask=out_mask[:, :, None] & channel_mask, other=0.0)
                alpha_ptrs = (
                    stack_ptr
                    + pid_b * stack_stride_b
                    + out_y * stack_stride_h
                    + out_x * stack_stride_w
                    + 3 * stack_stride_n
                    + 3 * stack_stride_c
                )
                alpha = tl.load(alpha_ptrs, mask=out_mask, other=0.0)
                valid = out_mask & (alpha != 0.0)
                blend_weight = weight * tl.where(valid, 1.0, 0.0)
                rgb_acc += tl.where(rgb_mask, vals * blend_weight[:, :, None], 0.0)
                valid_sum += blend_weight
                blend_alpha = tl.maximum(blend_alpha, tl.where(blend_weight > 0.0, alpha, 0.0))
                take_fallback = alpha > fallback_alpha
                fallback_vals = tl.where(take_fallback[:, :, None] & channel_mask, vals, fallback_vals)
                fallback_alpha = tl.where(take_fallback, alpha, fallback_alpha)

            if N_INPUTS > 4:
                weight = tl.where(raw_sum > 0.0, w4 / raw_sum, 0.0)
                ptrs = (
                    stack_ptr
                    + pid_b * stack_stride_b
                    + out_y[:, :, None] * stack_stride_h
                    + out_x[:, :, None] * stack_stride_w
                    + 4 * stack_stride_n
                    + offs_c[None, None, :] * stack_stride_c
                )
                vals = tl.load(ptrs, mask=out_mask[:, :, None] & channel_mask, other=0.0)
                alpha_ptrs = (
                    stack_ptr
                    + pid_b * stack_stride_b
                    + out_y * stack_stride_h
                    + out_x * stack_stride_w
                    + 4 * stack_stride_n
                    + 3 * stack_stride_c
                )
                alpha = tl.load(alpha_ptrs, mask=out_mask, other=0.0)
                valid = out_mask & (alpha != 0.0)
                blend_weight = weight * tl.where(valid, 1.0, 0.0)
                rgb_acc += tl.where(rgb_mask, vals * blend_weight[:, :, None], 0.0)
                valid_sum += blend_weight
                blend_alpha = tl.maximum(blend_alpha, tl.where(blend_weight > 0.0, alpha, 0.0))
                take_fallback = alpha > fallback_alpha
                fallback_vals = tl.where(take_fallback[:, :, None] & channel_mask, vals, fallback_vals)
                fallback_alpha = tl.where(take_fallback, alpha, fallback_alpha)

            if N_INPUTS > 5:
                weight = tl.where(raw_sum > 0.0, w5 / raw_sum, 0.0)
                ptrs = (
                    stack_ptr
                    + pid_b * stack_stride_b
                    + out_y[:, :, None] * stack_stride_h
                    + out_x[:, :, None] * stack_stride_w
                    + 5 * stack_stride_n
                    + offs_c[None, None, :] * stack_stride_c
                )
                vals = tl.load(ptrs, mask=out_mask[:, :, None] & channel_mask, other=0.0)
                alpha_ptrs = (
                    stack_ptr
                    + pid_b * stack_stride_b
                    + out_y * stack_stride_h
                    + out_x * stack_stride_w
                    + 5 * stack_stride_n
                    + 3 * stack_stride_c
                )
                alpha = tl.load(alpha_ptrs, mask=out_mask, other=0.0)
                valid = out_mask & (alpha != 0.0)
                blend_weight = weight * tl.where(valid, 1.0, 0.0)
                rgb_acc += tl.where(rgb_mask, vals * blend_weight[:, :, None], 0.0)
                valid_sum += blend_weight
                blend_alpha = tl.maximum(blend_alpha, tl.where(blend_weight > 0.0, alpha, 0.0))
                take_fallback = alpha > fallback_alpha
                fallback_vals = tl.where(take_fallback[:, :, None] & channel_mask, vals, fallback_vals)
                fallback_alpha = tl.where(take_fallback, alpha, fallback_alpha)

            if N_INPUTS > 6:
                weight = tl.where(raw_sum > 0.0, w6 / raw_sum, 0.0)
                ptrs = (
                    stack_ptr
                    + pid_b * stack_stride_b
                    + out_y[:, :, None] * stack_stride_h
                    + out_x[:, :, None] * stack_stride_w
                    + 6 * stack_stride_n
                    + offs_c[None, None, :] * stack_stride_c
                )
                vals = tl.load(ptrs, mask=out_mask[:, :, None] & channel_mask, other=0.0)
                alpha_ptrs = (
                    stack_ptr
                    + pid_b * stack_stride_b
                    + out_y * stack_stride_h
                    + out_x * stack_stride_w
                    + 6 * stack_stride_n
                    + 3 * stack_stride_c
                )
                alpha = tl.load(alpha_ptrs, mask=out_mask, other=0.0)
                valid = out_mask & (alpha != 0.0)
                blend_weight = weight * tl.where(valid, 1.0, 0.0)
                rgb_acc += tl.where(rgb_mask, vals * blend_weight[:, :, None], 0.0)
                valid_sum += blend_weight
                blend_alpha = tl.maximum(blend_alpha, tl.where(blend_weight > 0.0, alpha, 0.0))
                take_fallback = alpha > fallback_alpha
                fallback_vals = tl.where(take_fallback[:, :, None] & channel_mask, vals, fallback_vals)
                fallback_alpha = tl.where(take_fallback, alpha, fallback_alpha)

            if N_INPUTS > 7:
                weight = tl.where(raw_sum > 0.0, w7 / raw_sum, 0.0)
                ptrs = (
                    stack_ptr
                    + pid_b * stack_stride_b
                    + out_y[:, :, None] * stack_stride_h
                    + out_x[:, :, None] * stack_stride_w
                    + 7 * stack_stride_n
                    + offs_c[None, None, :] * stack_stride_c
                )
                vals = tl.load(ptrs, mask=out_mask[:, :, None] & channel_mask, other=0.0)
                alpha_ptrs = (
                    stack_ptr
                    + pid_b * stack_stride_b
                    + out_y * stack_stride_h
                    + out_x * stack_stride_w
                    + 7 * stack_stride_n
                    + 3 * stack_stride_c
                )
                alpha = tl.load(alpha_ptrs, mask=out_mask, other=0.0)
                valid = out_mask & (alpha != 0.0)
                blend_weight = weight * tl.where(valid, 1.0, 0.0)
                rgb_acc += tl.where(rgb_mask, vals * blend_weight[:, :, None], 0.0)
                valid_sum += blend_weight
                blend_alpha = tl.maximum(blend_alpha, tl.where(blend_weight > 0.0, alpha, 0.0))
                take_fallback = alpha > fallback_alpha
                fallback_vals = tl.where(take_fallback[:, :, None] & channel_mask, vals, fallback_vals)
                fallback_alpha = tl.where(take_fallback, alpha, fallback_alpha)

            blended = tl.where(rgb_mask, rgb_acc / tl.maximum(valid_sum[:, :, None], 1.0), 0.0)
            blended = tl.where(offs_c[None, None, :] == 3, blend_alpha[:, :, None], blended)
            use_blend = valid_sum > 0.0
            has_fallback = fallback_alpha > 0.0
            out_vals = tl.where(use_blend[:, :, None], blended, tl.where(has_fallback[:, :, None], fallback_vals, tl.zeros((BLOCK_H, BLOCK_W, 4), dtype=tl.float32)))
            out_ptrs = (
                out_ptr
                + pid_b * out_stride_b
                + out_y[:, :, None] * out_stride_h
                + out_x[:, :, None] * out_stride_w
                + offs_c[None, None, :] * out_stride_c
            )
            tl.store(out_ptrs, out_vals, mask=out_mask[:, :, None] & channel_mask)
            return

        out_vals = tl.zeros((BLOCK_H, BLOCK_W, 4), dtype=tl.float32)
        if N_INPUTS > 0:
            weight = tl.where(raw_sum > 0.0, w0 / raw_sum, 0.0)
            ptrs = (
                stack_ptr
                + pid_b * stack_stride_b
                + out_y[:, :, None] * stack_stride_h
                + out_x[:, :, None] * stack_stride_w
                + 0 * stack_stride_n
                + offs_c[None, None, :] * stack_stride_c
            )
            vals = tl.load(ptrs, mask=out_mask[:, :, None] & channel_mask, other=0.0)
            out_vals += vals * weight[:, :, None]
        if N_INPUTS > 1:
            weight = tl.where(raw_sum > 0.0, w1 / raw_sum, 0.0)
            ptrs = (
                stack_ptr
                + pid_b * stack_stride_b
                + out_y[:, :, None] * stack_stride_h
                + out_x[:, :, None] * stack_stride_w
                + 1 * stack_stride_n
                + offs_c[None, None, :] * stack_stride_c
            )
            vals = tl.load(ptrs, mask=out_mask[:, :, None] & channel_mask, other=0.0)
            out_vals += vals * weight[:, :, None]
        if N_INPUTS > 2:
            weight = tl.where(raw_sum > 0.0, w2 / raw_sum, 0.0)
            ptrs = (
                stack_ptr
                + pid_b * stack_stride_b
                + out_y[:, :, None] * stack_stride_h
                + out_x[:, :, None] * stack_stride_w
                + 2 * stack_stride_n
                + offs_c[None, None, :] * stack_stride_c
            )
            vals = tl.load(ptrs, mask=out_mask[:, :, None] & channel_mask, other=0.0)
            out_vals += vals * weight[:, :, None]
        if N_INPUTS > 3:
            weight = tl.where(raw_sum > 0.0, w3 / raw_sum, 0.0)
            ptrs = (
                stack_ptr
                + pid_b * stack_stride_b
                + out_y[:, :, None] * stack_stride_h
                + out_x[:, :, None] * stack_stride_w
                + 3 * stack_stride_n
                + offs_c[None, None, :] * stack_stride_c
            )
            vals = tl.load(ptrs, mask=out_mask[:, :, None] & channel_mask, other=0.0)
            out_vals += vals * weight[:, :, None]
        if N_INPUTS > 4:
            weight = tl.where(raw_sum > 0.0, w4 / raw_sum, 0.0)
            ptrs = (
                stack_ptr
                + pid_b * stack_stride_b
                + out_y[:, :, None] * stack_stride_h
                + out_x[:, :, None] * stack_stride_w
                + 4 * stack_stride_n
                + offs_c[None, None, :] * stack_stride_c
            )
            vals = tl.load(ptrs, mask=out_mask[:, :, None] & channel_mask, other=0.0)
            out_vals += vals * weight[:, :, None]
        if N_INPUTS > 5:
            weight = tl.where(raw_sum > 0.0, w5 / raw_sum, 0.0)
            ptrs = (
                stack_ptr
                + pid_b * stack_stride_b
                + out_y[:, :, None] * stack_stride_h
                + out_x[:, :, None] * stack_stride_w
                + 5 * stack_stride_n
                + offs_c[None, None, :] * stack_stride_c
            )
            vals = tl.load(ptrs, mask=out_mask[:, :, None] & channel_mask, other=0.0)
            out_vals += vals * weight[:, :, None]
        if N_INPUTS > 6:
            weight = tl.where(raw_sum > 0.0, w6 / raw_sum, 0.0)
            ptrs = (
                stack_ptr
                + pid_b * stack_stride_b
                + out_y[:, :, None] * stack_stride_h
                + out_x[:, :, None] * stack_stride_w
                + 6 * stack_stride_n
                + offs_c[None, None, :] * stack_stride_c
            )
            vals = tl.load(ptrs, mask=out_mask[:, :, None] & channel_mask, other=0.0)
            out_vals += vals * weight[:, :, None]
        if N_INPUTS > 7:
            weight = tl.where(raw_sum > 0.0, w7 / raw_sum, 0.0)
            ptrs = (
                stack_ptr
                + pid_b * stack_stride_b
                + out_y[:, :, None] * stack_stride_h
                + out_x[:, :, None] * stack_stride_w
                + 7 * stack_stride_n
                + offs_c[None, None, :] * stack_stride_c
            )
            vals = tl.load(ptrs, mask=out_mask[:, :, None] & channel_mask, other=0.0)
            out_vals += vals * weight[:, :, None]

        out_ptrs = (
            out_ptr
            + pid_b * out_stride_b
            + out_y[:, :, None] * out_stride_h
            + out_x[:, :, None] * out_stride_w
            + offs_c[None, None, :] * out_stride_c
        )
        tl.store(out_ptrs, out_vals, mask=out_mask[:, :, None] & channel_mask)
else:  # pragma: no cover - exercised only when Triton is unavailable
    _copy_roi_kernel = None
    _remap_kernel = None
    _remap_with_dest_map_kernel = None
    _downsample_mask_kernel = None
    _downsample_image_kernel = None
    _compute_laplacian_kernel = None
    _reconstruct_level_kernel = None
    _blend_laplacians_n_kernel = None
    _blend_laplacians_stacked_kernel = None


def _launch_grid(width: int, height: int, batch: int, block_w: int = 32, block_h: int = 8) -> tuple[int, int, int]:
    return triton.cdiv(width, block_w), triton.cdiv(height, block_h), batch


def _out_is_float(tensor: torch.Tensor) -> bool:
    if tensor.dtype == torch.float32:
        return True
    if tensor.dtype == torch.uint8:
        return False
    raise TypeError(f"Unsupported Triton output dtype: {tensor.dtype}")


def _launch_grid_2d(width: int, height: int, block_w: int = 32, block_h: int = 8) -> tuple[int, int]:
    return triton.cdiv(width, block_w), triton.cdiv(height, block_h)


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


def downsample_mask_triton(mask: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    if not _HAS_TRITON:
        raise ValueError("Triton is not available")
    if mask.dtype != torch.float32 or mask.ndim != 3 or not mask.is_cuda or not mask.is_contiguous():
        raise ValueError("Triton mask downsample requires a contiguous float32 HWC GPU tensor")
    out_h = (mask.shape[0] + 1) // 2
    out_w = (mask.shape[1] + 1) // 2
    if out is None:
        out = torch.empty((out_h, out_w, mask.shape[2]), device=mask.device, dtype=torch.float32)
    elif out.shape != (out_h, out_w, mask.shape[2]) or out.dtype != torch.float32 or not out.is_cuda or not out.is_contiguous():
        raise ValueError("Invalid Triton mask downsample output buffer")
    grid = _launch_grid_2d(out_w, out_h)
    _downsample_mask_kernel[grid](
        mask,
        out,
        *mask.stride(),
        *out.stride(),
        mask.shape[0],
        mask.shape[1],
        out_h,
        out_w,
        BLOCK_W=32,
        BLOCK_H=8,
        CHANNELS=mask.shape[2],
    )
    return out


def downsample_image_triton(image: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    if not _HAS_TRITON:
        raise ValueError("Triton is not available")
    if not _supported_blend_image_tensor(image) or not image.is_cuda:
        raise ValueError("Triton image downsample requires a contiguous float32 BHWC GPU tensor")
    out_h = (image.shape[1] + 1) // 2
    out_w = (image.shape[2] + 1) // 2
    if out is None:
        out = torch.empty((image.shape[0], out_h, out_w, image.shape[3]), device=image.device, dtype=torch.float32)
    elif out.shape != (image.shape[0], out_h, out_w, image.shape[3]) or out.dtype != torch.float32 or not out.is_cuda or not out.is_contiguous():
        raise ValueError("Invalid Triton image downsample output buffer")
    grid = _launch_grid(out_w, out_h, image.shape[0])
    _downsample_image_kernel[grid](
        image,
        out,
        *image.stride(),
        *out.stride(),
        image.shape[1],
        image.shape[2],
        out_h,
        out_w,
        BLOCK_W=32,
        BLOCK_H=8,
        CHANNELS=image.shape[3],
    )
    return out


def compute_laplacian_triton(high: torch.Tensor, low: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    if not _HAS_TRITON:
        raise ValueError("Triton is not available")
    if not (_supported_blend_image_tensor(high) and _supported_blend_image_tensor(low) and high.is_cuda and low.is_cuda):
        raise ValueError("Triton laplacian requires contiguous float32 BHWC GPU tensors")
    if high.shape[0] != low.shape[0] or high.shape[3] != low.shape[3]:
        raise ValueError("Mismatched tensor shapes for Triton laplacian")
    if out is None:
        out = torch.empty_like(high)
    elif out.shape != high.shape or out.dtype != torch.float32 or not out.is_cuda or not out.is_contiguous():
        raise ValueError("Invalid Triton laplacian output buffer")
    grid = _launch_grid(high.shape[2], high.shape[1], high.shape[0])
    _compute_laplacian_kernel[grid](
        high,
        low,
        out,
        *high.stride(),
        *low.stride(),
        *out.stride(),
        high.shape[1],
        high.shape[2],
        low.shape[1],
        low.shape[2],
        BLOCK_W=32,
        BLOCK_H=8,
        CHANNELS=high.shape[3],
    )
    return out


def reconstruct_level_triton(low: torch.Tensor, lap: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    if not _HAS_TRITON:
        raise ValueError("Triton is not available")
    if not (_supported_blend_image_tensor(low) and _supported_blend_image_tensor(lap) and low.is_cuda and lap.is_cuda):
        raise ValueError("Triton reconstruct requires contiguous float32 BHWC GPU tensors")
    if low.shape[0] != lap.shape[0] or low.shape[3] != lap.shape[3]:
        raise ValueError("Mismatched tensor shapes for Triton reconstruction")
    if out is None:
        out = torch.empty_like(lap)
    elif out.shape != lap.shape or out.dtype != torch.float32 or not out.is_cuda or not out.is_contiguous():
        raise ValueError("Invalid Triton reconstruction output buffer")
    grid = _launch_grid(lap.shape[2], lap.shape[1], lap.shape[0])
    _reconstruct_level_kernel[grid](
        low,
        lap,
        out,
        *low.stride(),
        *lap.stride(),
        *out.stride(),
        lap.shape[1],
        lap.shape[2],
        low.shape[1],
        low.shape[2],
        BLOCK_W=32,
        BLOCK_H=8,
        CHANNELS=lap.shape[3],
    )
    return out


def blend_laplacians_n_triton(laplacians: Sequence[torch.Tensor], mask: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    if not can_use_triton_blend_backend(laplacians, mask):
        raise ValueError("Triton N-way blend requires contiguous float32 GPU tensors")
    ref = laplacians[0]
    if out is None:
        out = torch.empty_like(ref)
    elif out.shape != ref.shape or out.dtype != torch.float32 or not out.is_cuda or not out.is_contiguous():
        raise ValueError("Invalid Triton blend output buffer")
    stacked = torch.stack(tuple(laplacians), dim=3).contiguous()
    grid = _launch_grid(ref.shape[2], ref.shape[1], ref.shape[0])
    _blend_laplacians_stacked_kernel[grid](
        stacked,
        mask,
        out,
        *stacked.stride(),
        *mask.stride(),
        *out.stride(),
        ref.shape[1],
        ref.shape[2],
        BLOCK_W=32,
        BLOCK_H=8,
        CHANNELS=ref.shape[3],
        N_INPUTS=len(laplacians),
    )
    return out
