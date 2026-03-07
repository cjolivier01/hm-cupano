from __future__ import annotations

from typing import Iterable, Literal

import torch

from .geometry import Rect
from .masks import UNMAPPED_POSITION_VALUE
from .triton_ops import (
    can_use_triton_backend,
    copy_roi_triton,
    remap_to_canvas_triton,
    remap_to_canvas_with_dest_map_triton,
    triton_available,
)

Backend = Literal["auto", "triton"]
ResolvedBackend = Literal["torch_impl", "triton"]


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
    return value.round().clamp(0, 255).to(dtype)


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
    mx = map_x[roi.y + sy0 : roi.y + sy0 + h, roi.x + sx0 : roi.x + sx0 + w]
    my = map_y[roi.y + sy0 : roi.y + sy0 + h, roi.x + sx0 : roi.x + sx0 + w]
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
        out[..., 3] = torch.where(
            unmapped.unsqueeze(0),
            torch.zeros(1, h, w, device=dest.device, dtype=dest.dtype),
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
    mx = map_x[roi.y + sy0 : roi.y + sy0 + h, roi.x + sx0 : roi.x + sx0 + w]
    my = map_y[roi.y + sy0 : roi.y + sy0 + h, roi.x + sx0 : roi.x + sx0 + w]
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


def downsample_mask(mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim != 3:
        raise ValueError(f"Expected HWC mask tensor, got {tuple(mask.shape)}")
    out_h = (mask.shape[0] + 1) // 2
    out_w = (mask.shape[1] + 1) // 2
    out = torch.zeros((out_h, out_w, mask.shape[2]), device=mask.device, dtype=mask.dtype)
    count = torch.zeros((out_h, out_w, 1), device=mask.device, dtype=mask.dtype)
    for dy in range(2):
        for dx in range(2):
            patch = mask[dy::2, dx::2, :]
            ph, pw = patch.shape[:2]
            out[:ph, :pw, :] += patch
            count[:ph, :pw, :] += 1
    return out / count.clamp_min(1)


def downsample_image(image: torch.Tensor) -> torch.Tensor:
    image = ensure_batched(image)
    out_h = (image.shape[1] + 1) // 2
    out_w = (image.shape[2] + 1) // 2
    channels = image.shape[-1]
    out = torch.zeros((image.shape[0], out_h, out_w, channels), device=image.device, dtype=image.dtype)
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


def compute_laplacian(high: torch.Tensor, low: torch.Tensor) -> torch.Tensor:
    high = ensure_batched(high)
    up = _upsample_alpha_aware(low, high.shape[1], high.shape[2]).to(torch.float32)
    lap = high.to(torch.float32).clone()
    if high.shape[-1] == 4:
        lap[..., :3] = high[..., :3].to(torch.float32) - up[..., :3]
        lap[..., 3] = high[..., 3].to(torch.float32)
    else:
        lap = high.to(torch.float32) - up
    return lap


def blend_laplacians_n(laplacians: Iterable[torch.Tensor], mask: torch.Tensor) -> torch.Tensor:
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
        out = torch.zeros((stack.shape[0], stack.shape[1], stack.shape[2], 4), device=stack.device, dtype=torch.float32)
        out[..., :3] = rgb
        out[..., 3:4] = alpha

        fallback_idx = alphas.argmax(dim=-1)
        fallback = torch.gather(
            stack,
            3,
            fallback_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, 1, stack.shape[-1]),
        ).squeeze(3)
        has_fallback = alphas.amax(dim=-1, keepdim=True) != 0
        use_blend = valid_sum > 0
        return torch.where(use_blend.expand_as(out), out, torch.where(has_fallback.expand_as(out), fallback, torch.zeros_like(out)))

    return (weights.unsqueeze(-1) * stack).sum(dim=3)


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


def laplacian_blend_n(inputs: list[torch.Tensor], mask: torch.Tensor, num_levels: int) -> torch.Tensor:
    if not inputs:
        raise ValueError("inputs must not be empty")
    batch = ensure_batched(inputs[0]).shape[0]
    channels = ensure_batched(inputs[0]).shape[-1]
    levels = compute_num_levels(inputs[0].shape[2], inputs[0].shape[1], num_levels)

    gauss: list[list[torch.Tensor]] = [[ensure_batched(inp).to(torch.float32)] for inp in inputs]
    mask_pyr = [mask.to(torch.float32)]
    for _ in range(1, levels):
        for i in range(len(inputs)):
            gauss[i].append(downsample_image(gauss[i][-1]))
        mask_pyr.append(downsample_mask(mask_pyr[-1]))

    laps: list[list[torch.Tensor]] = [[] for _ in inputs]
    for i in range(len(inputs)):
        for lvl in range(levels - 1):
            laps[i].append(compute_laplacian(gauss[i][lvl], gauss[i][lvl + 1]))
        laps[i].append(gauss[i][-1].to(torch.float32))

    blended = [blend_laplacians_n([laps[i][lvl] for i in range(len(inputs))], mask_pyr[lvl]) for lvl in range(levels)]
    recon = blended[-1]
    for lvl in range(levels - 2, -1, -1):
        up = _upsample_alpha_aware(recon, blended[lvl].shape[1], blended[lvl].shape[2]).to(torch.float32)
        next_recon = blended[lvl].clone()
        if channels == 4:
            next_recon[..., :3] = up[..., :3] + blended[lvl][..., :3]
            next_recon[..., 3] = blended[lvl][..., 3]
        else:
            next_recon = up + blended[lvl]
        recon = next_recon
    if recon.shape[0] != batch:
        raise AssertionError("Unexpected batch size change during reconstruction")
    return recon
