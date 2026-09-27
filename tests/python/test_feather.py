from __future__ import annotations

import numpy as np
import pytest

from cupano.feather import FeatherParams, build_weights, coverage_masks

UNMAPPED = 65535


def smoothstep(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def full_maps(n: int, width: int, height: int):
    return (
        [np.zeros((height, width), np.uint16) for _ in range(n)],
        [np.zeros((height, width), np.uint16) for _ in range(n)],
    )


def vertical_seam(width: int, height: int, x0: int) -> np.ndarray:
    labels = np.zeros((height, width), np.uint8)
    labels[:, x0:] = 1
    return labels


def test_two_camera_profile_matches_closed_form():
    """The signed-distance ramp must be a smoothstep centred on the geometric boundary."""
    width, height, x0, radius = 256, 16, 128, 32.0
    mx, my = full_maps(2, width, height)
    result = build_weights(
        vertical_seam(width, height, x0), mx, my, [(0, 0), (0, 0)], 2, FeatherParams(fraction=radius / width)
    )
    assert not result.hard
    assert result.radius_px == pytest.approx(radius)
    assert not result.overlap_capped

    xs = np.arange(width)
    expected = smoothstep(0.5 - (xs - x0 + 0.5) / radius)
    row = result.weights[height // 2]
    np.testing.assert_allclose(row[:, 0], expected, atol=1e-6)
    np.testing.assert_allclose(row[:, 1], 1.0 - expected, atol=1e-6)


def test_weights_sum_to_one_for_every_camera_count():
    """Weights are normalized here, not in the kernels: only the N kernel normalizes unconditionally."""
    width, height = 240, 40
    for n in (2, 3, 5, 8):
        mx, my = full_maps(n, width, height)
        labels = np.minimum(n - 1, np.arange(width) * n // width).astype(np.uint8)
        labels = np.repeat(labels[None], height, axis=0)
        result = build_weights(
            labels, mx, my, [(0, 0)] * n, n, FeatherParams(fraction=16.0 / width)
        )
        np.testing.assert_allclose(result.weights.sum(axis=2), 1.0, atol=1e-6)


def test_two_camera_weights_sum_to_one():
    """S(t) + S(1-t) == 1, so two cameras need no normalization at all."""
    width, height = 200, 24
    mx, my = full_maps(2, width, height)
    for fraction in (0.01, 0.05, 0.2, 0.5):
        result = build_weights(
            vertical_seam(width, height, 90), mx, my, [(0, 0), (0, 0)], 2, FeatherParams(fraction=fraction)
        )
        np.testing.assert_allclose(result.weights.sum(axis=2), 1.0, atol=1e-6)


def test_saturates_outside_the_band():
    width, height, x0, radius = 256, 8, 128, 32.0
    mx, my = full_maps(2, width, height)
    result = build_weights(
        vertical_seam(width, height, x0), mx, my, [(0, 0), (0, 0)], 2, FeatherParams(fraction=radius / width)
    )
    row = result.weights[height // 2]
    s = np.arange(width) - x0 + 0.5
    assert np.all(row[s <= -radius / 2, 0] == 1.0)
    assert np.all(row[s >= radius / 2, 0] == 0.0)


def test_gradient_is_bounded():
    """A hard edge anywhere is the failure this feature exists to avoid.

    The fixture has uniform full coverage on purpose, so the per-pixel cap does not engage and the
    local radius is constant. The raw ramp's slope is 1.5/R and normalization amplifies it to about
    1.13x, so the bound is 3/R. The field is 3/R_local-Lipschitz, so this does not generalize to
    geometries where the cap varies.
    """
    width, height, radius = 160, 120, 24.0
    mx, my = full_maps(3, width, height)
    labels = np.full((height, width), 2, np.uint8)
    labels[: height // 2, : width // 2] = 0
    labels[: height // 2, width // 2 :] = 1

    result = build_weights(
        labels, mx, my, [(0, 0)] * 3, 3, FeatherParams(fraction=radius / width)
    )
    limit = 3.0 / result.radius_px + 1e-5
    assert np.abs(np.diff(result.weights, axis=0)).max() <= limit
    assert np.abs(np.diff(result.weights, axis=1)).max() <= limit


def test_triple_point_splits_evenly():
    size = 161
    centre = size // 2
    mx, my = full_maps(3, size, size)
    yy, xx = np.mgrid[0:size, 0:size]
    angle = np.arctan2(yy - centre, xx - centre)
    labels = np.minimum(2, ((angle + np.pi) / (2 * np.pi) * 3).astype(np.uint8))

    result = build_weights(labels, mx, my, [(0, 0)] * 3, 3, FeatherParams(fraction=32.0 / size))
    apex = result.weights[centre, centre]
    np.testing.assert_allclose(apex / apex.sum(), 1.0 / 3.0, atol=0.05)
    # The blend kernel skips normalization when the sum is zero, so no pixel may reach it empty.
    assert result.weights.sum(axis=2).min() > 0.0


def test_weights_are_zero_outside_coverage():
    """Required for three-channel compute types, where the kernel has no alpha fallback."""
    width, height = 200, 32
    mx, my = full_maps(2, 120, height)
    result = build_weights(
        vertical_seam(width, height, 100), mx, my, [(0, 0), (80, 0)], 2, FeatherParams(fraction=16.0 / 120.0)
    )
    assert np.all(result.weights[:, 120:, 0] == 0.0)
    assert np.all(result.weights[:, :80, 1] == 0.0)
    assert result.weights.sum(axis=2).min() > 0.0


def test_radius_is_capped_by_overlap():
    """The cap is per pixel, so a 20 px overlap supports a 20 px total ramp rather than collapsing
    the whole canvas to the shallowest point on the seam."""
    width, height = 200, 16
    mx, my = full_maps(2, 110, height)
    result = build_weights(
        vertical_seam(width, height, 100), mx, my, [(0, 0), (90, 0)], 2, FeatherParams(fraction=0.9)
    )
    assert result.overlap_capped
    assert not result.hard
    # Exactly twice the shallowest coverage at the seam: a loose bound cannot tell 2*coverage
    # from 1*coverage, and the factor of two is the whole cap.
    assert result.radius_px == 20.0


def test_hard_fallback_keeps_the_uncorrected_labels():
    """The hard fallback builds its one-hot field from the labels handed in, not the corrected
    ones: that path exists to match the hard-seam mask, and the correction would move a seam the
    hard-seam mask does not have."""
    width, height, seam_x = 128, 40, 60
    params = FeatherParams(fraction=0.0)

    holed = np.zeros((height, width), np.uint16)
    # A hole inside camera 0's own region that camera 1 covers, so the correction has somewhere
    # to reassign to and the two label images genuinely differ.
    holed[8:28, 20:50] = UNMAPPED
    mx = [holed, np.zeros((height, width), np.uint16)]
    labels = vertical_seam(width, height, seam_x)

    r = build_weights(labels, mx, [m.copy() for m in mx], [(0, 0), (0, 0)], 2, params)
    assert r.hard, "zero width must take the hard fallback"

    coverage = coverage_masks((height, width), mx, [m.copy() for m in mx], [(0, 0), (0, 0)])
    would_move = (labels == 0) & (coverage[0] == 0) & (coverage[1] != 0)
    assert would_move.any(), "fixture does not exercise the correction"

    one_hot = np.zeros(labels.shape + (2,), np.float32)
    one_hot[..., 0] = labels == 0
    one_hot[..., 1] = labels == 1
    assert np.array_equal(r.weights, one_hot)
    assert np.array_equal(r.corrected_labels, labels)


def test_correction_picks_the_deepest_coverer():
    """The correction reassigns to the camera covering the pixel most deeply, not to the first one
    that covers it. The deepest coverer has the most room for its taper."""
    width, height, hole_x, hole_w = 200, 60, 40, 30
    params = FeatherParams(fraction=0.1)

    # Camera 0 owns everything but loses a block. Camera 1 grazes that block with a shallow strip;
    # camera 2 covers it deeply. Camera 1 comes first, so first and deepest disagree.
    owner = np.zeros((height, width), np.uint16)
    owner[10:30, hole_x : hole_x + hole_w] = UNMAPPED
    shallow = np.full((height, width), UNMAPPED, np.uint16)
    shallow[10:14, hole_x : hole_x + hole_w] = 0
    deep = np.full((height, width), UNMAPPED, np.uint16)
    deep[0:40, hole_x - 20 : hole_x + hole_w + 20] = 0
    mx = [owner, shallow, deep]
    labels = np.zeros((height, width), np.uint8)

    r = build_weights(labels, mx, [m.copy() for m in mx], [(0, 0)] * 3, 3, params)
    assert r.corrected_labels is not None
    assert np.all(r.corrected_labels[10:14, hole_x : hole_x + hole_w] == 2)


def test_zero_fraction_yields_exact_one_hot():
    width, height, x0 = 64, 16, 30
    mx, my = full_maps(2, width, height)
    labels = vertical_seam(width, height, x0)
    result = build_weights(labels, mx, my, [(0, 0), (0, 0)], 2, FeatherParams(fraction=0.0))
    assert result.hard
    assert result.radius_px == 0.0
    np.testing.assert_array_equal(result.weights[..., 0], (labels == 0).astype(np.float32))
    np.testing.assert_array_equal(result.weights[..., 1], (labels == 1).astype(np.float32))


def test_canvas_border_does_not_feather():
    """distanceTransform treats outside-the-image as foreground; the border is not a seam."""
    width, height, pad, x0 = 200, 32, 40, 100
    mx_small, my_small = full_maps(2, width, height)
    small = build_weights(
        vertical_seam(width, height, x0), mx_small, my_small, [(0, 0), (0, 0)], 2,
        FeatherParams(fraction=32.0 / width),
    )

    wide = width + 2 * pad
    mx_big, my_big = full_maps(2, wide, height)
    labels_big = np.zeros((height, wide), np.uint8)
    labels_big[:, x0 + pad :] = 1
    big = build_weights(
        labels_big, mx_big, my_big, [(0, 0), (0, 0)], 2, FeatherParams(fraction=small.radius_px / wide)
    )
    assert small.radius_px == pytest.approx(big.radius_px, abs=1e-3)
    np.testing.assert_allclose(
        small.weights[height // 2, :, 0], big.weights[height // 2, pad : pad + width, 0], atol=1e-6
    )


def test_eight_cameras_stay_in_range():
    width, height, n = 400, 24, 8
    mx, my = full_maps(n, width, height)
    labels = np.minimum(n - 1, (np.arange(width) * n // width)).astype(np.uint8)
    labels = np.repeat(labels[None], height, axis=0)
    result = build_weights(labels, mx, my, [(0, 0)] * n, n, FeatherParams(fraction=20.0 / width))
    assert result.weights.shape[2] == n
    assert np.isfinite(result.weights).all()
    assert result.weights.min() >= 0.0 and result.weights.max() <= 1.0
    assert result.weights.sum(axis=2).min() > 0.0


def test_empty_region_stays_finite():
    """A label with no pixels saturates one transform, at about 1.8e19 with the exact transform, 3.4e38 with the fast one, never infinity, so
    both forms stay finite. What this pins is that an empty region produces a usable field at all;
    the half-pixel offset selecting preserves is pinned by the closed-form profile test."""
    width, height = 96, 16
    mx, my = full_maps(3, width, height)
    result = build_weights(
        vertical_seam(width, height, 48), mx, my, [(0, 0)] * 3, 3, FeatherParams(fraction=16.0 / width)
    )
    assert np.isfinite(result.weights).all()
    assert np.all(result.weights[..., 2] == 0.0)
    np.testing.assert_allclose(result.weights.sum(axis=2), 1.0, atol=1e-6)


def test_uncovered_pixels_get_zero_weight():
    width, height = 200, 16
    mx, my = full_maps(2, 80, height)
    mx[0][:, 20:30] = UNMAPPED
    labels = np.zeros((height, width), np.uint8)
    labels[:, 70:] = 1
    result = build_weights(labels, mx, my, [(0, 0), (60, 0)], 2, FeatherParams(fraction=8.0 / 80.0))
    totals = result.weights.sum(axis=2)
    assert np.all(totals[:, 20:30] == 0.0)
    assert np.all(totals[:, 140:] == 0.0)


def test_rejects_malformed_input():
    mx, my = full_maps(2, 32, 8)
    labels = vertical_seam(32, 8, 16)
    with pytest.raises(ValueError):
        build_weights(labels.astype(np.float32), mx, my, [(0, 0), (0, 0)], 2)
    with pytest.raises(ValueError):
        build_weights(labels, mx, my, [(0, 0), (0, 0)], 3)
    with pytest.raises(ValueError):
        build_weights(labels, mx, my, [(0, 0), (0, 0)], 2, FeatherParams(fraction=-0.1))
    # Labels index the per-camera lists directly, so an out-of-range one must be rejected.
    bad_label = labels.copy()
    bad_label[0, 0] = 5
    with pytest.raises(ValueError):
        build_weights(bad_label, mx, my, [(0, 0), (0, 0)], 2)
    # NaN must be rejected, not silently degenerate to a hard seam.
    with pytest.raises(ValueError):
        build_weights(labels, mx, my, [(0, 0), (0, 0)], 2, FeatherParams(fraction=float("nan")))
    # An empty seam has no canvas to build a field on.
    with pytest.raises(ValueError):
        build_weights(np.zeros((0, 0), np.uint8), mx, my, [(0, 0), (0, 0)], 2)


def test_coverage_masks_follow_unmapped_sentinel():
    mx, my = full_maps(1, 20, 6)
    mx[0][2, 3] = UNMAPPED
    masks = coverage_masks((6, 40), mx, my, [(5, 0)])
    assert masks[0][0, 0] == 0
    assert masks[0][0, 5] == 255
    assert masks[0][2, 8] == 0
    assert masks[0][0, 25] == 0


def test_blanketing_third_camera_does_not_defeat_the_cap():
    """The cap must be over the cameras that actually contribute, not "covered by >= 2 cameras".

    A third camera blanketing the area inflates that count, so the cap would never engage while the
    two cameras meeting at the seam are strangled by their own tapers.
    """
    width, height, cam_w, overlap = 300, 60, 150, 10
    bx = cam_w - overlap
    params = FeatherParams(fraction=40.0 / cam_w)

    two_labels = np.zeros((height, width), np.uint8)
    two_labels[:, cam_w - overlap // 2 :] = 1
    mx = [np.zeros((height, cam_w), np.uint16), np.zeros((height, width - bx), np.uint16)]
    control = build_weights(two_labels, mx, [m.copy() for m in mx], [(0, 0), (bx, 0)], 2, params)
    assert control.overlap_capped
    assert control.min_seam_radius_px <= overlap

    three_labels = two_labels.copy()
    three_labels[height - 5 :, :] = 2
    mx3 = mx + [np.zeros((height, width), np.uint16)]
    blanketed = build_weights(
        three_labels, mx3, [m.copy() for m in mx3], [(0, 0), (bx, 0), (0, 0)], 3, params
    )
    assert blanketed.overlap_capped, "the blanketing camera hid the narrow A/B overlap"
    assert blanketed.min_seam_radius_px <= overlap


def test_non_covering_neighbour_does_not_collapse_the_cap():
    """The mirror image: a camera near another seam that covers nothing there has weight exactly
    zero, so it must not be allowed to drive the cap to zero."""
    width, height, seam_x = 600, 200, 250
    params = FeatherParams(fraction=0.5)

    two_labels = np.zeros((height, width), np.uint8)
    two_labels[:, seam_x:] = 1
    mx = [np.zeros((height, 400), np.uint16), np.zeros((height, 450), np.uint16)]
    control = build_weights(two_labels, mx, [m.copy() for m in mx], [(0, 0), (150, 0)], 2, params)
    assert not control.hard
    assert control.radius_px > 100.0

    # Same remap width, so the requested radius is unchanged; only the mapped region is small.
    island = np.full((height, 400), UNMAPPED, np.uint16)
    island[80:120, 0:60] = 0
    three_labels = two_labels.copy()
    three_labels[80:120, 300:360] = 2
    mx3 = mx + [island]
    islanded = build_weights(
        three_labels, mx3, [m.copy() for m in mx3], [(0, 0), (150, 0), (300, 0)], 3, params
    )
    assert not islanded.hard, "a non-covering neighbour collapsed the whole field"
    assert islanded.radius_px >= control.radius_px * 0.99


def ramp_width(weights: np.ndarray, y: int, x_lo: int, x_hi: int, channel: int) -> int:
    row = weights[y, x_lo:x_hi, channel]
    return int(np.count_nonzero((row > 0.02) & (row < 0.98)))


def test_grazing_neighbour_with_zero_weight_does_not_collapse_the_cap():
    """A camera whose weight at p is exactly zero must not pin the cap there. Testing reach against
    the requested radius made it pin anyway, collapsing a 400 px crossfade to 4 px."""
    width, height, seam_x = 600, 400, 250
    sliver_y, sliver_h, island_x, island_w = 198, 5, 310, 20
    params = FeatherParams(fraction=1.0)

    two_labels = vertical_seam(width, height, seam_x)
    mx = [np.zeros((height, 400), np.uint16), np.zeros((height, 400), np.uint16)]
    control = build_weights(two_labels, mx, [m.copy() for m in mx], [(0, 0), (150, 0)], 2, params)
    control_ramp = ramp_width(control.weights, sliver_y + 2, 180, 290, 0)
    assert control_ramp > 40, "fixture is wrong: the two-camera seam should feather widely"

    # Camera 2 owns an island 60 px right of the seam and drags a 5 px tall footprint back across
    # it. Same remap width as the others, so the request is unchanged.
    sliver = np.full((height, 400), UNMAPPED, np.uint16)
    sliver[sliver_y : sliver_y + sliver_h, 0:134] = 0
    three_labels = two_labels.copy()
    three_labels[sliver_y : sliver_y + sliver_h, island_x : island_x + island_w] = 2
    mx3 = mx + [sliver]
    grazed = build_weights(
        three_labels, mx3, [m.copy() for m in mx3], [(0, 0), (150, 0), (198, 0)], 3, params
    )
    assert not grazed.hard
    assert grazed.weights[sliver_y + 2, seam_x, 2] == 0.0
    grazed_ramp = ramp_width(grazed.weights, sliver_y + 2, 180, 290, 0)
    assert grazed_ramp >= control_ramp // 2, (
        f"a grazing camera with zero weight collapsed the crossfade: "
        f"{grazed_ramp} px against {control_ramp}"
    )


def test_requested_radius_reports_the_request_even_when_the_cap_bites():
    """The ROI pad is sized from the request, not from the widest seam pixel."""
    width, height, seam_x = 600, 400, 396
    params = FeatherParams(fraction=1.0)

    # An 8 px overlap, so the cap bites along the whole seam rather than at one pinch. Without
    # that, radius_px happens to equal the request and the two cannot be told apart.
    labels = vertical_seam(width, height, seam_x)
    mx = [np.zeros((height, 400), np.uint16), np.zeros((height, 400), np.uint16)]
    r = build_weights(labels, mx, [m.copy() for m in mx], [(0, 0), (392, 0)], 2, params)
    assert not r.hard
    assert r.requested_radius_px == 400.0, "the request must survive the cap"
    assert r.radius_px < 16.0, "the fixture must actually be capped everywhere"
    assert r.min_seam_radius_px == r.radius_px
    assert r.overlap_capped
    # Every seam pixel is capped, which also pins the share's denominator to the seam.
    assert r.capped_seam_fraction == 1.0


def test_capped_share_separates_a_local_pinch_from_a_general_one():
    """min_seam_radius_px is a single worst pixel and is almost always tiny on a real rig, so the
    share of the seam that was actually narrowed is reported alongside it."""
    width, height, seam_x = 600, 400, 250
    params = FeatherParams(fraction=0.1)

    labels = vertical_seam(width, height, seam_x)
    labels[0:4, 152:160] = 1
    mx = [np.zeros((height, 400), np.uint16), np.zeros((height, 400), np.uint16)]
    r = build_weights(labels, mx, [m.copy() for m in mx], [(0, 0), (150, 0)], 2, params)
    assert r.overlap_capped
    assert r.min_seam_radius_px < 8.0, "the island should pinch hard"
    assert r.radius_px == 40.0, "the rest of the seam should be untouched"
    assert 0.0 < r.capped_seam_fraction < 0.2, "a local pinch must not read as a general one"


def test_coverage_corrected_labels_keep_the_partition_total():
    """The seam label can name a camera that does not cover the pixel; correcting it is what keeps
    the partition total non-zero, and so keeps the pixel from going black."""
    width, height, seam_x = 200, 120, 100
    params = FeatherParams(fraction=0.2)

    narrow = np.full((height, 200), UNMAPPED, np.uint16)
    narrow[:, 0:60] = 0
    mx = [narrow, np.zeros((height, 200), np.uint16)]
    labels = vertical_seam(width, height, seam_x)
    r = build_weights(labels, mx, [m.copy() for m in mx], [(0, 0), (0, 0)], 2, params)

    coverage = coverage_masks((height, width), mx, [m.copy() for m in mx], [(0, 0), (0, 0)])
    any_covers = (coverage[0] != 0) | (coverage[1] != 0)
    total = r.weights.sum(axis=2)
    assert np.allclose(total[any_covers], 1.0, atol=1e-4)


def test_seam_statistics_cover_both_sides_of_a_label_change():
    """Sampling only the left side hides a camera whose footprint starts on the right flank."""
    width, height, seam_x = 400, 200, 200
    params = FeatherParams(fraction=0.5)

    right = np.full((height, 400), UNMAPPED, np.uint16)
    right[:, seam_x:] = 0
    mx = [np.zeros((height, 400), np.uint16), right]
    labels = vertical_seam(width, height, seam_x)
    r = build_weights(labels, mx, [m.copy() for m in mx], [(0, 0), (0, 0)], 2, params)
    assert r.overlap_capped, "the right flank of the seam has no coverage depth to feather into"
    assert r.min_seam_radius_px < 8.0
