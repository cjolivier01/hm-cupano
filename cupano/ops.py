from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal

import torch

from .geometry import Rect
from .masks import UNMAPPED_POSITION_VALUE
from .triton_ops import (
    blend_laplacians_n_triton,
    can_use_triton_backend,
    can_use_triton_blend_backend,
    compute_laplacian_triton,
    copy_roi_triton,
    downsample_image_triton,
    downsample_mask_triton,
    reconstruct_level_triton,
    remap_to_canvas_triton,
    remap_to_canvas_with_dest_map_triton,
    triton_available,
)

Backend = Literal["auto", "triton"]
ResolvedBackend = Literal["torch_impl", "triton"]


@dataclass
class LaplacianBlendWorkspace:
    key: tuple[object, ...] | None = None
    gauss: list[list[torch.Tensor]] = field(default_factory=list)
    laps: list[list[torch.Tensor]] = field(default_factory=list)
    blended: list[torch.Tensor] = field(default_factory=list)
    recon: list[torch.Tensor] = field(default_factory=list)
    mask_pyr: list[torch.Tensor] = field(default_factory=list)


def ensure_batched(image: torch.Tensor) -> torch.Tensor:
    if image.ndim == 3:
        return image.unsqueeze(0)
    if image.ndim != 4:
        raise ValueError(f"Expected HWC or BHWC tensor, got shape {tuple(image.shape)}")
    return image


def alpha_constant(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.tensor(255.0 if dtype.is_floating_point else 255, device=device, dtype=dtype)


def cast_like(value: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if value.dtype == dtype:
        return value
    if dtype.is_floating_point:
        return value.to(dtype)
    return value.clamp(0, 255).to(dtype)


def resolve_backend(backend: Backend, *tensors: torch.Tensor) -> ResolvedBackend:
    if backend == "triton":
        if not can_use_triton_backend(*tensors):
            raise ValueError("Triton backend requested for unsupported tensors")
        return "triton"
    if backend == "auto":
        if can_use_triton_backend(*tensors):
            return "triton"
        return "torch_impl"
    raise ValueError(f"Unsupported backend: {backend}")


def _region_intersection(offset_x: int, offset_y: int, roi: Rect, dest_w: int, dest_h: int) -> tuple[int, int, int, int, int, int] | None:
    dst_left = offset_x + roi.x
    dst_top = offset_y + roi.y
    dst_right = dst_left + roi.width
    dst_bottom = dst_top + roi.height
    x0 = max(0, dst_left)
    y0 = max(0, dst_top)
    x1 = min(dest_w, dst_right)
    y1 = min(dest_h, dst_bottom)
    if x0 >= x1 or y0 >= y1:
        return None
    sx0 = x0 - dst_left
    sy0 = y0 - dst_top
    return x0, y0, x1, y1, sx0, sy0


def _gather_pixels(src: torch.Tensor, mx: torch.Tensor, my: torch.Tensor) -> torch.Tensor:
    batch, _, width, channels = src.shape
    flat = src.reshape(batch, -1, channels)
    idx = (my * width + mx).reshape(-1)
    gathered = flat.index_select(1, idx)
    return gathered.reshape(batch, mx.shape[0], mx.shape[1], channels)


def _copy_roi_torch(src: torch.Tensor, dest: torch.Tensor, region: Rect, src_roi_x: int, src_roi_y: int, offset_x: int, offset_y: int) -> torch.Tensor:
    if region.width <= 0 or region.height <= 0:
        return dest
    x_start = max(0, -src_roi_x, -offset_x)
    y_start = max(0, -src_roi_y, -offset_y)
    x_end = min(region.width, src.shape[2] - src_roi_x, dest.shape[2] - offset_x)
    y_end = min(region.height, src.shape[1] - src_roi_y, dest.shape[1] - offset_y)
    if x_start >= x_end or y_start >= y_end:
        return dest
    src_x0 = src_roi_x + x_start
    src_y0 = src_roi_y + y_start
    src_x1 = src_roi_x + x_end
    src_y1 = src_roi_y + y_end
    dest_x0 = offset_x + x_start
    dest_y0 = offset_y + y_start
    dest_x1 = offset_x + x_end
    dest_y1 = offset_y + y_end
    src_patch = src[:, src_y0:src_y1, src_x0:src_x1, :]
    dest[:, dest_y0:dest_y1, dest_x0:dest_x1, :] = cast_like(src_patch, dest.dtype)
    return dest


def copy_roi(
    src: torch.Tensor,
    dest: torch.Tensor,
    region: Rect,
    src_roi_x: int,
    src_roi_y: int,
    offset_x: int,
    offset_y: int,
    *,
    backend: Backend = "auto",
) -> torch.Tensor:
    chosen_backend = resolve_backend(backend, src, dest)
    if chosen_backend == "triton":
        return copy_roi_triton(src, dest, region, src_roi_x, src_roi_y, offset_x, offset_y)
    return _copy_roi_torch(src, dest, region, src_roi_x, src_roi_y, offset_x, offset_y)


def simple_make_full(
    src: torch.Tensor,
    dest: torch.Tensor,
    region_width: int,
    region_height: int,
    src_roi_x: int,
    src_roi_y: int,
    dest_offset_x: int,
    dest_offset_y: int,
    *,
    backend: Backend = "auto",
) -> torch.Tensor:
    dest.zero_()
    return copy_roi(
        src,
        dest,
        Rect(0, 0, region_width, region_height),
        src_roi_x,
        src_roi_y,
        dest_offset_x,
        dest_offset_y,
        backend=backend,
    )


def _remap_to_canvas_torch(
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
    src = ensure_batched(src)
    if roi is None:
        roi = Rect(0, 0, int(map_x.shape[1]), int(map_x.shape[0]))
    if roi.empty:
        return dest
    inter = _region_intersection(offset_x, offset_y, roi, dest.shape[2], dest.shape[1])
    if inter is None:
        return dest
    x0, y0, x1, y1, sx0, sy0 = inter
    h = y1 - y0
    w = x1 - x0
    map_x0 = roi.x + sx0
    map_y0 = roi.y + sy0
    map_x1 = map_x0 + w
    map_y1 = map_y0 + h
    if map_x0 < 0:
        x0 -= map_x0
        map_x0 = 0
    if map_y0 < 0:
        y0 -= map_y0
        map_y0 = 0
    if map_x1 > map_x.shape[1]:
        x1 -= map_x1 - map_x.shape[1]
        map_x1 = map_x.shape[1]
    if map_y1 > map_x.shape[0]:
        y1 -= map_y1 - map_x.shape[0]
        map_y1 = map_x.shape[0]
    h = y1 - y0
    w = x1 - x0
    if h <= 0 or w <= 0:
        return dest
    mx = map_x[map_y0:map_y1, map_x0:map_x1]
    my = map_y[map_y0:map_y1, map_x0:map_x1]
    src_h = src.shape[1]
    src_w = src.shape[2]
    unmapped = (mx == int(UNMAPPED_POSITION_VALUE)) | (my == int(UNMAPPED_POSITION_VALUE))
    valid = (mx >= 0) & (my >= 0) & (mx < src_w) & (my < src_h) & ~unmapped
    mx_clamped = mx.clamp(0, max(src_w - 1, 0)).long()
    my_clamped = my.clamp(0, max(src_h - 1, 0)).long()
    sampled = cast_like(_gather_pixels(src, mx_clamped, my_clamped), dest.dtype)
    if adjustment is not None:
        sampled = sampled.clone()
        sampled[..., : min(3, sampled.shape[-1])] += adjustment.to(device=sampled.device, dtype=sampled.dtype)

    out = torch.zeros((src.shape[0], h, w, src.shape[-1]), device=dest.device, dtype=dest.dtype)
    if src.shape[-1] == 4 and fill_invalid_alpha:
        unmapped_3d = unmapped.unsqueeze(0)
        out[..., 3] = torch.where(
            unmapped_3d,
            torch.zeros_like(unmapped_3d, device=dest.device, dtype=dest.dtype),
            alpha_constant(dest.dtype, dest.device),
        )
    out = torch.where(valid.unsqueeze(0).unsqueeze(-1), sampled, out)

    dest_patch = dest[:, y0:y1, x0:x1, :]
    if no_unmapped_write:
        write_mask = (~unmapped).unsqueeze(0).unsqueeze(-1)
        dest[:, y0:y1, x0:x1, :] = torch.where(write_mask, out, dest_patch)
    else:
        dest[:, y0:y1, x0:x1, :] = out
    return dest


def remap_to_canvas(
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
    backend: Backend = "auto",
) -> torch.Tensor:
    chosen_backend = resolve_backend(backend, src, dest)
    if chosen_backend == "triton":
        try:
            return remap_to_canvas_triton(
                src,
                dest,
                map_x,
                map_y,
                offset_x,
                offset_y,
                adjustment=adjustment,
                no_unmapped_write=no_unmapped_write,
                roi=roi,
                fill_invalid_alpha=fill_invalid_alpha,
            )
        except ValueError:
            if backend == "triton":
                raise
    return _remap_to_canvas_torch(
        src,
        dest,
        map_x,
        map_y,
        offset_x,
        offset_y,
        adjustment=adjustment,
        no_unmapped_write=no_unmapped_write,
        roi=roi,
        fill_invalid_alpha=fill_invalid_alpha,
    )


def _remap_to_canvas_with_dest_map_torch(
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
    src = ensure_batched(src)
    if roi is None:
        roi = Rect(0, 0, int(map_x.shape[1]), int(map_x.shape[0]))
    if roi.empty:
        return dest
    inter = _region_intersection(offset_x, offset_y, roi, dest.shape[2], dest.shape[1])
    if inter is None:
        return dest
    x0, y0, x1, y1, sx0, sy0 = inter
    h = y1 - y0
    w = x1 - x0
    class_mask = dest_image_map[y0:y1, x0:x1] == image_index
    if not torch.any(class_mask):
        return dest
    map_x0 = roi.x + sx0
    map_y0 = roi.y + sy0
    map_x1 = map_x0 + w
    map_y1 = map_y0 + h
    if map_x0 < 0:
        x0 -= map_x0
        map_x0 = 0
    if map_y0 < 0:
        y0 -= map_y0
        map_y0 = 0
    if map_x1 > map_x.shape[1]:
        x1 -= map_x1 - map_x.shape[1]
        map_x1 = map_x.shape[1]
    if map_y1 > map_x.shape[0]:
        y1 -= map_y1 - map_x.shape[0]
        map_y1 = map_x.shape[0]
    h = y1 - y0
    w = x1 - x0
    if h <= 0 or w <= 0:
        return dest
    class_mask = dest_image_map[y0:y1, x0:x1] == image_index
    if not torch.any(class_mask):
        return dest
    mx = map_x[map_y0:map_y1, map_x0:map_x1]
    my = map_y[map_y0:map_y1, map_x0:map_x1]
    src_h = src.shape[1]
    src_w = src.shape[2]
    valid = (mx >= 0) & (my >= 0) & (mx < src_w) & (my < src_h)
    mx_clamped = mx.clamp(0, max(src_w - 1, 0)).long()
    my_clamped = my.clamp(0, max(src_h - 1, 0)).long()
    sampled = cast_like(_gather_pixels(src, mx_clamped, my_clamped), dest.dtype)
    if adjustment is not None:
        sampled = sampled.clone()
        sampled[..., : min(3, sampled.shape[-1])] += adjustment.to(device=sampled.device, dtype=sampled.dtype)
    out = torch.zeros((src.shape[0], h, w, src.shape[-1]), device=dest.device, dtype=dest.dtype)
    out = torch.where(valid.unsqueeze(0).unsqueeze(-1), sampled, out)
    write_mask = class_mask.unsqueeze(0).unsqueeze(-1)
    dest_patch = dest[:, y0:y1, x0:x1, :]
    dest[:, y0:y1, x0:x1, :] = torch.where(write_mask, out, dest_patch)
    return dest


def remap_to_canvas_with_dest_map(
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
    backend: Backend = "auto",
) -> torch.Tensor:
    chosen_backend = resolve_backend(backend, src, dest)
    if chosen_backend == "triton":
        try:
            return remap_to_canvas_with_dest_map_triton(
                src,
                dest,
                map_x,
                map_y,
                image_index,
                dest_image_map,
                offset_x,
                offset_y,
                adjustment=adjustment,
                roi=roi,
            )
        except ValueError:
            if backend == "triton":
                raise
    return _remap_to_canvas_with_dest_map_torch(
        src,
        dest,
        map_x,
        map_y,
        image_index,
        dest_image_map,
        offset_x,
        offset_y,
        adjustment=adjustment,
        roi=roi,
    )


def _downsample_mask_torch(mask: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    if mask.ndim != 3:
        raise ValueError(f"Expected HWC mask tensor, got {tuple(mask.shape)}")
    out_h = (mask.shape[0] + 1) // 2
    out_w = (mask.shape[1] + 1) // 2
    if out is None:
        out = torch.zeros((out_h, out_w, mask.shape[2]), device=mask.device, dtype=mask.dtype)
    else:
        out.zero_()
    count = torch.zeros((out_h, out_w, 1), device=mask.device, dtype=mask.dtype)
    for dy in range(2):
        for dx in range(2):
            patch = mask[dy::2, dx::2, :]
            ph, pw = patch.shape[:2]
            out[:ph, :pw, :] += patch
            count[:ph, :pw, :] += 1
    return out / count.clamp_min(1)


def downsample_mask(mask: torch.Tensor, *, out: torch.Tensor | None = None, backend: Backend = "auto") -> torch.Tensor:
    if backend == "triton":
        return downsample_mask_triton(mask, out=out)
    if backend == "auto" and triton_available() and mask.is_cuda and mask.is_contiguous() and mask.dtype == torch.float32:
        try:
            return downsample_mask_triton(mask, out=out)
        except ValueError:
            pass
    return _downsample_mask_torch(mask, out=out)


def _downsample_image_torch(image: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    image = ensure_batched(image)
    out_h = (image.shape[1] + 1) // 2
    out_w = (image.shape[2] + 1) // 2
    channels = image.shape[-1]
    if out is None:
        out = torch.zeros((image.shape[0], out_h, out_w, channels), device=image.device, dtype=image.dtype)
    else:
        out.zero_()
    if channels == 4:
        sums = torch.zeros((image.shape[0], out_h, out_w, 3), device=image.device, dtype=image.dtype)
        counts = torch.zeros((image.shape[0], out_h, out_w, 1), device=image.device, dtype=image.dtype)
        alpha = torch.zeros((image.shape[0], out_h, out_w, 1), device=image.device, dtype=image.dtype)
        for dy in range(2):
            for dx in range(2):
                patch = image[:, dy::2, dx::2, :]
                ph, pw = patch.shape[1:3]
                a = patch[..., 3:4]
                keep = (a != 0).to(image.dtype)
                sums[:, :ph, :pw, :] += patch[..., :3] * keep
                counts[:, :ph, :pw, :] += keep
                alpha[:, :ph, :pw, :] = torch.maximum(alpha[:, :ph, :pw, :], a)
        out[..., :3] = sums / counts.clamp_min(1)
        out[..., 3:4] = alpha
        return out

    count = torch.zeros((image.shape[0], out_h, out_w, 1), device=image.device, dtype=image.dtype)
    for dy in range(2):
        for dx in range(2):
            patch = image[:, dy::2, dx::2, :]
            ph, pw = patch.shape[1:3]
            out[:, :ph, :pw, :] += patch
            count[:, :ph, :pw, :] += 1
    return out / count.clamp_min(1)


def downsample_image(image: torch.Tensor, *, out: torch.Tensor | None = None, backend: Backend = "auto") -> torch.Tensor:
    if backend == "triton":
        return downsample_image_triton(image, out=out)
    if backend == "auto" and triton_available() and image.is_cuda and image.is_contiguous() and image.dtype == torch.float32:
        try:
            return downsample_image_triton(image, out=out)
        except ValueError:
            pass
    return _downsample_image_torch(image, out=out)


def _upsample_alpha_aware(low: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
    low = ensure_batched(low)
    batch, low_h, low_w, channels = low.shape
    ys = torch.arange(out_h, device=low.device, dtype=torch.float32) / 2.0
    xs = torch.arange(out_w, device=low.device, dtype=torch.float32) / 2.0
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    x0 = gx.floor().long().clamp(0, max(low_w - 1, 0))
    y0 = gy.floor().long().clamp(0, max(low_h - 1, 0))
    x1 = (x0 + 1).clamp(0, max(low_w - 1, 0))
    y1 = (y0 + 1).clamp(0, max(low_h - 1, 0))
    dx = (gx - x0.float()).unsqueeze(0).unsqueeze(-1)
    dy = (gy - y0.float()).unsqueeze(0).unsqueeze(-1)

    p00 = _gather_pixels(low, x0, y0)
    p10 = _gather_pixels(low, x1, y0)
    p01 = _gather_pixels(low, x0, y1)
    p11 = _gather_pixels(low, x1, y1)

    if channels == 4:
        out = torch.zeros((batch, out_h, out_w, channels), device=low.device, dtype=low.dtype)
        neighbors = (p00, p10, p01, p11)
        weights = ((1.0 - dx) * (1.0 - dy), dx * (1.0 - dy), (1.0 - dx) * dy, dx * dy)
        sum_rgb = torch.zeros((batch, out_h, out_w, 3), device=low.device, dtype=torch.float32)
        sum_w = torch.zeros((batch, out_h, out_w, 1), device=low.device, dtype=torch.float32)
        for patch, weight in zip(neighbors, weights, strict=True):
            valid = (patch[..., 3:4] != 0).to(torch.float32)
            sum_rgb += patch[..., :3].to(torch.float32) * weight * valid
            sum_w += weight * valid
        out[..., :3] = cast_like(sum_rgb / sum_w.clamp_min(1.0), low.dtype)
        return out

    w00 = (1.0 - dx) * (1.0 - dy)
    w10 = dx * (1.0 - dy)
    w01 = (1.0 - dx) * dy
    w11 = dx * dy
    return cast_like(
        p00.to(torch.float32) * w00 + p10.to(torch.float32) * w10 + p01.to(torch.float32) * w01 + p11.to(torch.float32) * w11,
        low.dtype,
    )


def _compute_laplacian_torch(high: torch.Tensor, low: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    high = ensure_batched(high)
    up = _upsample_alpha_aware(low, high.shape[1], high.shape[2]).to(torch.float32)
    if high.shape[-1] == 4:
        lap = torch.empty_like(high, dtype=torch.float32) if out is None else out
        lap.copy_(high.to(torch.float32))
        lap[..., :3] = high[..., :3].to(torch.float32) - up[..., :3]
        lap[..., 3] = high[..., 3].to(torch.float32)
        return lap
    result = high.to(torch.float32) - up
    if out is not None:
        out.copy_(result)
        return out
    return result


def compute_laplacian(
    high: torch.Tensor,
    low: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    backend: Backend = "auto",
) -> torch.Tensor:
    if backend == "triton":
        return compute_laplacian_triton(high, low, out=out)
    if backend == "auto" and triton_available() and high.is_cuda and low.is_cuda:
        try:
            return compute_laplacian_triton(high, low, out=out)
        except ValueError:
            pass
    return _compute_laplacian_torch(high, low, out=out)


def _blend_laplacians_n_torch(
    laplacians: Iterable[torch.Tensor],
    mask: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    laps = [ensure_batched(lap).to(torch.float32) for lap in laplacians]
    stack = torch.stack(laps, dim=3)
    weights = mask.to(device=stack.device, dtype=torch.float32)
    weights_sum = weights.sum(dim=-1, keepdim=True)
    weights = torch.where(weights_sum > 0, weights / weights_sum.clamp_min(1e-12), weights)
    weights = weights.unsqueeze(0).expand(stack.shape[0], -1, -1, -1)

    channels = stack.shape[-1]
    if channels == 4:
        alphas = stack[..., 3]
        weights = torch.where(alphas != 0, weights, torch.zeros_like(weights))
        valid_sum = weights.sum(dim=-1, keepdim=True)
        renorm = torch.where(valid_sum > 0, weights / valid_sum.clamp_min(1e-12), weights)
        rgb = (renorm.unsqueeze(-1) * stack[..., :3]).sum(dim=3)
        alpha = torch.where(renorm > 0, alphas, torch.zeros_like(alphas)).amax(dim=-1, keepdim=True)
        blend_out = torch.zeros((stack.shape[0], stack.shape[1], stack.shape[2], 4), device=stack.device, dtype=torch.float32)
        blend_out[..., :3] = rgb
        blend_out[..., 3:4] = alpha

        fallback_idx = alphas.argmax(dim=-1)
        fallback = torch.gather(
            stack,
            3,
            fallback_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, 1, stack.shape[-1]),
        ).squeeze(3)
        has_fallback = alphas.amax(dim=-1, keepdim=True) != 0
        use_blend = valid_sum > 0
        result = torch.where(use_blend.expand_as(blend_out), blend_out, torch.where(has_fallback.expand_as(blend_out), fallback, torch.zeros_like(blend_out)))
        if out is not None:
            out.copy_(result)
            return out
        return result

    result = (weights.unsqueeze(-1) * stack).sum(dim=3)
    if out is not None:
        out.copy_(result)
        return out
    return result


def _blend_laplacians_two_cpp_torch(
    lap1: torch.Tensor,
    lap2: torch.Tensor,
    mask: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    a = ensure_batched(lap1).to(torch.float32)
    b = ensure_batched(lap2).to(torch.float32)
    weights = mask.to(device=a.device, dtype=torch.float32)
    if weights.ndim == 2:
        weights = weights.unsqueeze(-1)
    if weights.ndim != 3 or weights.shape[-1] != 1:
        raise ValueError(f"Expected HWC single-channel mask, got {tuple(weights.shape)}")
    if a.shape != b.shape:
        raise ValueError("Two-image blend inputs must have the same shape")

    if a.shape[-1] != 4:
        result = weights.unsqueeze(0) * a + (1.0 - weights).unsqueeze(0) * b
        if out is not None:
            out.copy_(result)
            return out
        return result

    result = torch.empty_like(a) if out is None else out
    alpha1 = a[..., 3]
    alpha2 = b[..., 3]
    cond_copy_b = alpha1 == 0
    cond_copy_a = (~cond_copy_b) & (alpha2 == 0)
    cond_blend = (~cond_copy_b) & (~cond_copy_a)

    result.copy_(b)
    result[cond_copy_a] = a[cond_copy_a]

    m = weights[..., 0].unsqueeze(0)
    rgb = m.unsqueeze(-1) * a[..., :3] + (1.0 - m).unsqueeze(-1) * b[..., :3]
    result[..., :3] = torch.where(cond_blend.unsqueeze(-1), rgb, result[..., :3])
    result[..., 3] = torch.where(cond_blend, alpha2, result[..., 3])
    return result


def _laplacian_blend_two_cpp(
    input1: torch.Tensor,
    input2: torch.Tensor,
    mask: torch.Tensor,
    levels: int,
    *,
    backend: Backend = "auto",
    workspace: LaplacianBlendWorkspace | None = None,
) -> torch.Tensor:
    base1 = ensure_batched(input1).to(torch.float32)
    base2 = ensure_batched(input2).to(torch.float32)
    mask_f32 = mask.to(device=base1.device, dtype=torch.float32)
    if mask_f32.ndim == 3 and mask_f32.shape[-1] == 2:
        mask_f32 = mask_f32[..., :1]
    if mask_f32.ndim == 2:
        mask_f32 = mask_f32.unsqueeze(-1)
    if mask_f32.ndim != 3 or mask_f32.shape[-1] != 1:
        raise ValueError(f"Expected HWC single-channel mask, got {tuple(mask_f32.shape)}")

    use_backend: Backend = backend
    if backend == "auto":
        use_backend = "triton" if can_use_triton_blend_backend([base1, base2], mask_f32) else "auto"

    inputs = [base1, base2]
    if workspace is not None:
        workspace = _ensure_laplacian_workspace(workspace, inputs, mask_f32, levels)
        gauss = workspace.gauss
        laps = workspace.laps
        blended = workspace.blended
        recon_levels = workspace.recon
        mask_pyr = workspace.mask_pyr
        gauss[0][0].copy_(base1)
        gauss[1][0].copy_(base2)
        mask_pyr[0].copy_(mask_f32)
    else:
        gauss = [[base1.clone()], [base2.clone()]]
        laps = [[], []]
        blended = []
        recon_levels = []
        mask_pyr = [mask_f32.clone()]

    for lvl in range(1, levels):
        if workspace is not None:
            downsample_image(gauss[0][lvl - 1], out=gauss[0][lvl], backend=use_backend)
            downsample_image(gauss[1][lvl - 1], out=gauss[1][lvl], backend=use_backend)
            downsample_mask(mask_pyr[lvl - 1], out=mask_pyr[lvl], backend=use_backend)
        else:
            gauss[0].append(downsample_image(gauss[0][-1], backend=use_backend))
            gauss[1].append(downsample_image(gauss[1][-1], backend=use_backend))
            mask_pyr.append(downsample_mask(mask_pyr[-1], backend=use_backend))

    for lvl in range(levels - 1):
        if workspace is not None:
            compute_laplacian(gauss[0][lvl], gauss[0][lvl + 1], out=laps[0][lvl], backend=use_backend)
            compute_laplacian(gauss[1][lvl], gauss[1][lvl + 1], out=laps[1][lvl], backend=use_backend)
            _blend_laplacians_two_cpp_torch(laps[0][lvl], laps[1][lvl], mask_pyr[lvl], out=blended[lvl])
        else:
            lap1 = compute_laplacian(gauss[0][lvl], gauss[0][lvl + 1], backend=use_backend)
            lap2 = compute_laplacian(gauss[1][lvl], gauss[1][lvl + 1], backend=use_backend)
            laps[0].append(lap1)
            laps[1].append(lap2)
            blended.append(_blend_laplacians_two_cpp_torch(lap1, lap2, mask_pyr[lvl]))

    coarse1 = gauss[0][-1]
    coarse2 = gauss[1][-1]
    if workspace is not None:
        _blend_laplacians_two_cpp_torch(coarse1, coarse2, mask_pyr[-1], out=blended[-1])
        recon_levels[-1].copy_(blended[-1])
        for lvl in range(levels - 2, -1, -1):
            reconstruct_level(recon_levels[lvl + 1], blended[lvl], out=recon_levels[lvl], backend=use_backend)
        return recon_levels[0]

    blended.append(_blend_laplacians_two_cpp_torch(coarse1, coarse2, mask_pyr[-1]))
    recon = blended[-1]
    for lvl in range(levels - 2, -1, -1):
        recon = reconstruct_level(recon, blended[lvl], backend=use_backend)
    return recon


def blend_laplacians_n(
    laplacians: Iterable[torch.Tensor],
    mask: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    backend: Backend = "auto",
) -> torch.Tensor:
    laps = [ensure_batched(lap).to(torch.float32) for lap in laplacians]
    if backend == "triton":
        return blend_laplacians_n_triton(laps, mask, out=out)
    if backend == "auto" and can_use_triton_blend_backend(laps, mask):
        try:
            return blend_laplacians_n_triton(laps, mask, out=out)
        except ValueError:
            pass
    return _blend_laplacians_n_torch(laps, mask, out=out)


def _reconstruct_level_torch(low: torch.Tensor, lap: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    up = _upsample_alpha_aware(low, lap.shape[1], lap.shape[2]).to(torch.float32)
    if lap.shape[-1] == 4:
        next_recon = lap.clone() if out is None else out
        if out is not None:
            next_recon.copy_(lap)
        next_recon[..., :3] = up[..., :3] + lap[..., :3]
        next_recon[..., 3] = lap[..., 3]
        return next_recon
    result = up + lap
    if out is not None:
        out.copy_(result)
        return out
    return result


def reconstruct_level(
    low: torch.Tensor,
    lap: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    backend: Backend = "auto",
) -> torch.Tensor:
    if backend == "triton":
        return reconstruct_level_triton(low, lap, out=out)
    if backend == "auto" and triton_available() and low.is_cuda and lap.is_cuda:
        try:
            return reconstruct_level_triton(low, lap, out=out)
        except ValueError:
            pass
    return _reconstruct_level_torch(low, lap, out=out)


def compute_num_levels(width: int, height: int, requested_levels: int) -> int:
    levels = max(1, requested_levels)
    widths = [width]
    heights = [height]
    for _ in range(1, levels):
        if widths[-1] < 2 or heights[-1] < 2:
            break
        widths.append((widths[-1] + 1) // 2)
        heights.append((heights[-1] + 1) // 2)
    return len(widths)


def _level_shapes(width: int, height: int, levels: int) -> list[tuple[int, int]]:
    shapes = [(height, width)]
    for _ in range(1, levels):
        prev_h, prev_w = shapes[-1]
        shapes.append(((prev_h + 1) // 2, (prev_w + 1) // 2))
    return shapes


def _workspace_key(inputs: list[torch.Tensor], mask: torch.Tensor, levels: int) -> tuple[object, ...]:
    ref = ensure_batched(inputs[0]).to(torch.float32)
    return (
        str(ref.device),
        ref.shape[0],
        ref.shape[3],
        len(inputs),
        levels,
        tuple(_level_shapes(ref.shape[2], ref.shape[1], levels)),
        tuple(mask.shape),
    )


def _ensure_laplacian_workspace(
    workspace: LaplacianBlendWorkspace,
    inputs: list[torch.Tensor],
    mask: torch.Tensor,
    levels: int,
) -> LaplacianBlendWorkspace:
    key = _workspace_key(inputs, mask, levels)
    if workspace.key == key:
        return workspace

    ref = ensure_batched(inputs[0]).to(torch.float32)
    batch = ref.shape[0]
    channels = ref.shape[3]
    device = ref.device
    n_inputs = len(inputs)
    mask_channels = mask.shape[2]
    shapes = _level_shapes(ref.shape[2], ref.shape[1], levels)

    workspace.gauss = []
    workspace.laps = []
    for _ in range(n_inputs):
        gauss_levels = []
        lap_levels = []
        for level_h, level_w in shapes:
            gauss_levels.append(torch.empty((batch, level_h, level_w, channels), device=device, dtype=torch.float32))
            lap_levels.append(torch.empty((batch, level_h, level_w, channels), device=device, dtype=torch.float32))
        workspace.gauss.append(gauss_levels)
        workspace.laps.append(lap_levels)

    workspace.mask_pyr = [torch.empty((level_h, level_w, mask_channels), device=device, dtype=torch.float32) for level_h, level_w in shapes]
    workspace.blended = [torch.empty((batch, level_h, level_w, channels), device=device, dtype=torch.float32) for level_h, level_w in shapes]
    workspace.recon = [torch.empty((batch, level_h, level_w, channels), device=device, dtype=torch.float32) for level_h, level_w in shapes]
    workspace.key = key
    return workspace


def laplacian_blend_n(
    inputs: list[torch.Tensor],
    mask: torch.Tensor,
    num_levels: int,
    *,
    backend: Backend = "auto",
    workspace: LaplacianBlendWorkspace | None = None,
) -> torch.Tensor:
    if not inputs:
        raise ValueError("inputs must not be empty")
    batch = ensure_batched(inputs[0]).shape[0]
    levels = compute_num_levels(inputs[0].shape[2], inputs[0].shape[1], num_levels)

    if len(inputs) == 2 and mask.ndim == 3 and mask.shape[-1] == 2:
        return _laplacian_blend_two_cpp(inputs[0], inputs[1], mask, levels, backend=backend, workspace=workspace)

    base_inputs = [ensure_batched(inp).to(torch.float32) for inp in inputs]
    mask_f32 = mask.to(device=base_inputs[0].device, dtype=torch.float32)
    use_backend: Backend = backend
    if backend == "auto":
        use_backend = "triton" if can_use_triton_blend_backend(base_inputs, mask_f32) else "auto"

    if workspace is not None:
        workspace = _ensure_laplacian_workspace(workspace, base_inputs, mask_f32, levels)
        gauss = workspace.gauss
        laps = workspace.laps
        blended = workspace.blended
        recon_levels = workspace.recon
        mask_pyr = workspace.mask_pyr
        for i, inp in enumerate(base_inputs):
            gauss[i][0].copy_(inp)
        mask_pyr[0].copy_(mask_f32)
    else:
        gauss = [[inp.clone()] for inp in base_inputs]
        laps = [[] for _ in base_inputs]
        blended = []
        recon_levels = []
        mask_pyr = [mask_f32.clone()]

    for lvl in range(1, levels):
        for i in range(len(base_inputs)):
            if workspace is not None:
                downsample_image(gauss[i][lvl - 1], out=gauss[i][lvl], backend=use_backend)
            else:
                gauss[i].append(downsample_image(gauss[i][-1], backend=use_backend))
        if workspace is not None:
            downsample_mask(mask_pyr[lvl - 1], out=mask_pyr[lvl], backend=use_backend)
        else:
            mask_pyr.append(downsample_mask(mask_pyr[-1], backend=use_backend))

    for lvl in range(levels - 1):
        level_laps: list[torch.Tensor] = []
        for i in range(len(base_inputs)):
            if workspace is not None:
                compute_laplacian(gauss[i][lvl], gauss[i][lvl + 1], out=laps[i][lvl], backend=use_backend)
                level_laps.append(laps[i][lvl])
            else:
                lap = compute_laplacian(gauss[i][lvl], gauss[i][lvl + 1], backend=use_backend)
                laps[i].append(lap)
                level_laps.append(lap)
        if workspace is not None:
            blend_laplacians_n(level_laps, mask_pyr[lvl], out=blended[lvl], backend=use_backend)
        else:
            blended.append(blend_laplacians_n(level_laps, mask_pyr[lvl], backend=use_backend))

    coarse_inputs = [gauss[i][-1] for i in range(len(base_inputs))]
    if workspace is not None:
        blend_laplacians_n(coarse_inputs, mask_pyr[-1], out=blended[-1], backend=use_backend)
        recon_levels[-1].copy_(blended[-1])
        for lvl in range(levels - 2, -1, -1):
            reconstruct_level(recon_levels[lvl + 1], blended[lvl], out=recon_levels[lvl], backend=use_backend)
        recon = recon_levels[0]
    else:
        blended.append(blend_laplacians_n(coarse_inputs, mask_pyr[-1], backend=use_backend))
        recon = blended[-1]
        for lvl in range(levels - 2, -1, -1):
            recon = reconstruct_level(recon, blended[lvl], backend=use_backend)

    if recon.shape[0] != batch:
        raise AssertionError("Unexpected batch size change during reconstruction")
    return recon
