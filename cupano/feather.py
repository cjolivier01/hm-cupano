"""Feathered per-camera blend weights for the alpha blend mode.

Mirror of ``src/pano/featherMask.{h,cpp}``. Both sides call ``cv2.distanceTransform`` with the same
parameters and follow the same steps. With the default ``DIST_MASK_PRECISE`` transform the two
produced bit-identical weights on most configurations compared and agreed to one float32 ULP
(1.2e-07) on the rest, across two OpenCV major versions. All reported metadata matched exactly.

``FeatherParams.fast`` (``DIST_MASK_5``) is the exception: that chamfer approximation itself differs
between OpenCV builds by up to 3e-04, so the two sides can diverge by ~1e-06 there. Use the default
when the two implementations have to agree.

A weight field derived from ``distanceTransform(label == i)`` alone is inert: the seam labels are a
total partition, so that distance is zero everywhere outside region ``i`` and, once the blend kernel
normalizes, the owner always wins and the seam stays hard. Widening a camera's influence past its
own label needs a *signed* distance.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

UNMAPPED_POSITION_VALUE = 65535
# Below one pixel the crossfade cannot be represented and the mask degenerates to the hard seam.
MIN_USEFUL_RADIUS = 1.0

DEFAULT_FEATHER_FRACTION = 0.05
MAX_FEATHER_PX = 512.0


@dataclass
class FeatherParams:
    """Crossfade width controls. ``fraction`` is of the narrowest camera footprint width."""

    fraction: float = DEFAULT_FEATHER_FRACTION
    max_px: float = MAX_FEATHER_PX
    fast: bool = False


@dataclass
class FeatherResult:
    weights: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    radius_px: float = 0.0
    min_seam_radius_px: float = 0.0
    requested_radius_px: float = 0.0
    overlap_capped: bool = False
    capped_seam_fraction: float = 0.0
    # The labels the field was actually built from: seam_index with every pixel whose owner has no
    # data there reassigned to the camera covering it most deeply. A coverage hole inside an
    # overlap moves a seam, so a blend ROI derived from the original labels can miss the crossfade.
    corrected_labels: np.ndarray | None = None
    hard: bool = False


def _distance_mask(fast: bool) -> int:
    return cv2.DIST_MASK_5 if fast else cv2.DIST_MASK_PRECISE


def _smoothstep(t: np.ndarray) -> np.ndarray:
    """``smoothstep(clamp(t, 0, 1))``."""
    clamped = np.clip(t, 0.0, 1.0)
    return clamped * clamped * (3.0 - 2.0 * clamped)


def coverage_masks(
    canvas_shape: tuple[int, int],
    remap_x: list[np.ndarray],
    remap_y: list[np.ndarray],
    positions: list[tuple[int, int]],
) -> list[np.ndarray]:
    """Canvas-sized uint8 masks, 255 where camera ``i`` has a mapped source pixel."""
    height, width = canvas_shape
    masks: list[np.ndarray] = []
    for index, (mx, my) in enumerate(zip(remap_x, remap_y)):
        mask = np.zeros((height, width), dtype=np.uint8)
        x0, y0 = positions[index]
        x1 = min(x0 + mx.shape[1], width)
        y1 = min(y0 + mx.shape[0], height)
        cx0, cy0 = max(x0, 0), max(y0, 0)
        if x1 > cx0 and y1 > cy0:
            sub_x = mx[cy0 - y0 : y1 - y0, cx0 - x0 : x1 - x0]
            sub_y = my[cy0 - y0 : y1 - y0, cx0 - x0 : x1 - x0]
            mapped = (sub_x != UNMAPPED_POSITION_VALUE) & (sub_y != UNMAPPED_POSITION_VALUE)
            mask[cy0:y1, cx0:x1] = np.where(mapped, np.uint8(255), np.uint8(0))
        masks.append(mask)
    return masks


def build_weights(
    seam_index: np.ndarray,
    remap_x: list[np.ndarray],
    remap_y: list[np.ndarray],
    positions: list[tuple[int, int]],
    n_images: int,
    params: FeatherParams | None = None,
) -> FeatherResult:
    """Feathered ``[H, W, N]`` float32 blend weights from a hard seam label image.

    ``seam_index`` must be the padded, canonicalized canvas-sized label image, never a blend-ROI
    crop: distance transforms are non-local, so cropping first produces a hard edge at the crop
    boundary. Crop the returned weights instead.
    """
    params = params or FeatherParams()
    if seam_index.size == 0 or seam_index.dtype != np.uint8 or seam_index.ndim != 2:
        raise ValueError("Feather mask requires a non-empty 2-D uint8 seam index image")
    if n_images < 2 or not (len(remap_x) == len(remap_y) == len(positions) == n_images):
        raise ValueError("Feather mask input sizes do not match n_images")
    # Written as `not (x >= 0)` so NaN is rejected, matching the C++.
    if not params.fraction >= 0.0 or not params.max_px >= 0.0:
        raise ValueError("Feather mask requires a non-negative fraction and max_px")
    for mx, my in zip(remap_x, remap_y):
        if mx.dtype != np.uint16 or my.dtype != np.uint16:
            raise ValueError("Feather mask requires uint16 remap maps")
        # The C++ rejects a multi-channel map because CV_16UC2 != CV_16U; without this a
        # (H, W, 2) pair reaches coverage_masks and dies on a numpy broadcast instead.
        if mx.ndim != 2 or my.ndim != 2:
            raise ValueError("Feather mask requires single-channel remap maps")
        if mx.shape != my.shape:
            raise ValueError("Feather mask remap x and y maps must be the same size")

    if int(seam_index.max(initial=0)) >= n_images:
        raise ValueError("Feather mask seam index contains a label outside [0, n_images)")

    canvas_shape = seam_index.shape
    narrowest = min(int(m.shape[1]) for m in remap_x)
    # Match the C++ float32 product so overlap_capped agrees across implementations.
    radius = float(
        min(
            np.float32(params.fraction) * np.float32(narrowest),
            np.float32(params.max_px),
        )
    )
    if not radius >= MIN_USEFUL_RADIUS:
        # Subpixel widths reproduce the hard seam exactly, including its uncorrected labels.
        # No coverage or distance fields are needed for this one-hot partition.
        one_hot = np.zeros(canvas_shape + (n_images,), dtype=np.float32)
        for i in range(n_images):
            one_hot[..., i] = (seam_index == i).astype(np.float32)
        return FeatherResult(weights=one_hot, corrected_labels=seam_index, hard=True)

    mask_type = _distance_mask(params.fast)

    coverage = coverage_masks(canvas_shape, remap_x, remap_y, positions)
    coverage_distance = [cv2.distanceTransform(c, cv2.DIST_L2, mask_type) for c in coverage]

    # Coverage-corrected labels. Where a pixel's owner has no source data, hand it to the camera
    # that covers it most deeply, keeping the label image a total partition so the blend kernel
    # always sees a non-zero weight sum. Pixels no camera covers keep their original label; every
    # contributor there is transparent, so the kernel emits transparent black.
    labels = seam_index.copy()
    covered = np.stack([c != 0 for c in coverage], axis=0)
    depth = np.stack(coverage_distance, axis=0)
    owner_covered = np.take_along_axis(covered, labels[None].astype(np.intp), axis=0)[0]
    any_covered = covered.any(axis=0)
    needs_fix = (~owner_covered) & any_covered
    if needs_fix.any():
        masked_depth = np.where(covered, depth, -np.inf)
        labels[needs_fix] = np.argmax(masked_depth, axis=0).astype(np.uint8)[needs_fix]

    # Cap the crossfade per pixel at what the cameras that actually contribute there can support.
    #
    # A global minimum does not survive real data: it samples pixels no camera covers and lets one
    # pinhole collapse the whole canvas. "Covered by at least two cameras" is not equivalent for
    # N > 2 either: a third camera blanketing the area inflates the count and the cap never
    # engages.
    #
    # Camera i only bends the ramp at p if it is inside the band, and from the weight formula
    # below it is inside exactly when outside_i < R/2 + 0.5. So i is satisfied by either of two
    # things: R small enough to keep it out (R <= 2*outside_i - 1), or enough depth to survive its
    # own taper (R <= 2*coverage_i). Taking the larger of the two per camera and the minimum
    # across cameras is the widest R that holds at R itself. Testing reach against the *requested*
    # radius instead would be circular, and not conservatively so: a camera 60 px away whose
    # weight is exactly zero once the band narrows would still pin the cap, collapsing a 400 px
    # seam to 4 px.
    cap = np.full(canvas_shape, np.finfo(np.float32).max, dtype=np.float32)
    for i in range(n_images):
        not_owner = ((labels != i).astype(np.uint8)) * 255
        outside = cv2.distanceTransform(not_owner, cv2.DIST_L2, mask_type)
        allowed = np.maximum(
            coverage_distance[i] * np.float32(2.0), outside * np.float32(2.0) - np.float32(1.0)
        )
        # A camera with no data at p contributes nothing there for any R, because its taper reads
        # coverage_distance == 0. Letting it into the minimum would strangle a good seam; a
        # seam-label island near another seam is enough to trigger that.
        cap = np.where(covered[i], np.minimum(cap, allowed), cap)

    # cap holds a radius already, and stays at FLT_MAX where no camera covers.
    radius_map = np.minimum(cap, np.float32(radius)).astype(np.float32)

    # Report against the seam, where the crossfade actually happens. Both sides of a label change,
    # matching blend_roi::seam_boundary_bbox.
    seam_boundary = np.zeros(canvas_shape, dtype=bool)
    right = labels[:, 1:] != labels[:, :-1]
    seam_boundary[:, :-1] |= right
    seam_boundary[:, 1:] |= right
    down = labels[1:, :] != labels[:-1, :]
    seam_boundary[:-1, :] |= down
    seam_boundary[1:, :] |= down

    result = FeatherResult()
    if seam_boundary.any():
        seam_radii = radius_map[seam_boundary]
        max_radius, min_radius = float(seam_radii.max()), float(seam_radii.min())
    else:
        max_radius = float(radius_map.max()) if radius_map.size else 0.0
        min_radius = max_radius
    result.radius_px = max_radius
    result.min_seam_radius_px = min_radius
    result.corrected_labels = labels
    result.requested_radius_px = radius
    result.overlap_capped = min_radius < radius
    if seam_boundary.any():
        result.capped_seam_fraction = float(
            np.count_nonzero(radius_map[seam_boundary] < np.float32(radius))
        ) / float(np.count_nonzero(seam_boundary))

    # Both supported distance transforms give coverage depth >= 1 wherever a camera covers.
    # The cap is therefore >= 2, and the requested radius already passed the >= 1 check above.
    half_radius_map = radius_map * 0.5

    planes = np.zeros(canvas_shape + (n_images,), dtype=np.float32)
    for i in range(n_images):
        is_owner = (labels == i).astype(np.uint8) * 255
        inside = cv2.distanceTransform(is_owner, cv2.DIST_L2, mask_type)
        outside = cv2.distanceTransform(255 - is_owner, cv2.DIST_L2, mask_type)
        # Signed distance, positive outside region i, with the half-pixel offset that puts the 50%
        # point on the geometric boundary. Selected rather than subtracted: an empty or
        # canvas-filling region saturates one of the transforms at about 1.8e19. Selecting rather
        # than subtracting also keeps the half-pixel offset on the owner side.
        phi = np.where(labels == i, 0.5 - inside, outside - 0.5)
        seam_weight = _smoothstep(0.5 - phi / radius_map)
        # Safety net at this camera's own footprint edge. The per-pixel cap keeps the band inside
        # the shared coverage, so the taper is close to 1 there but not exactly 1: the cap gives
        # coverage >= R/2 while the taper reaches 1 only at R/2 + 0.5, so host normalization below
        # is what makes the weights sum to one. It keeps a camera from contributing outside its
        # footprint on three-channel compute types, where the kernels have no zero-alpha fallback.
        coverage_weight = _smoothstep((coverage_distance[i] - 0.5) / half_radius_map)
        planes[..., i] = seam_weight * coverage_weight

    # Normalize here rather than leaving it to the kernels: only the N kernel normalizes
    # unconditionally, the three-image kernel does so only for four-channel compute, and the
    # two-image kernel synthesizes the second weight as 1-m and never normalizes at all.
    total = planes.sum(axis=2)
    divisor = np.where(total == 0.0, 1.0, total).astype(np.float32)
    planes /= divisor[..., None]

    result.weights = planes
    return result
