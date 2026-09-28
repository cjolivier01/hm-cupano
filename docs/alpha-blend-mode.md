# Alpha blend mode

`CudaStitchPano`, `CudaStitchPano3` and `CudaStitchPanoN` can blend with a feathered crossfade
instead of a Laplacian pyramid. Select it with `BlendSettings::Alpha(fraction)` in place of the
pyramid level count:

```cpp
CudaStitchPano<uchar4, float4> pano(batch, hm::pano::BlendSettings::Alpha(0.05f), masks, ...);
```

Alpha mode requires a floating-point compute type such as `float3`, `float4`, `half3` or `half4`.
Integer input and output pixels remain supported. An integral compute type returns
`cudaErrorNotSupported` at construction because it would round the fractional feather weights
to zero or one, losing the crossfade.

`BlendSettings` converts implicitly from `int`, so existing callers that pass a level count keep
working and keep their meaning, including the convention that `0` means a hard seam.

## Why

The Laplacian blend mixes the two cameras across every spatial scale in the overlap. That hides
exposure and white-balance differences well, but it also attenuates the high-frequency detail a
downstream object detector relies on, over a band as wide as the coarsest pyramid level. A
crossfade confined to a few dozen pixels around the seam leaves the rest of the overlap carrying
unmodified source pixels.

It is also much cheaper. Alpha mode is one kernel launch over level 0 with no pyramid: at
7680x2160, half4, three cameras, six levels, the Laplacian context allocates about 875 MiB of
scratch that alpha mode does not allocate at all.

The tradeoff is real and runs the other way: a narrow crossfade cannot hide a brightness step
between cameras. There is no exposure compensation anywhere in this repository, so if the two
cameras disagree over the overlap, alpha mode will show it and the Laplacian blend will not.
Measure the mean per-channel difference over the overlap band on real footage before switching.

## The weight field

The seam labels are a total partition: every canvas pixel names exactly one owning camera. A weight
field built from `distanceTransform(label == i)` alone is inert, because that distance is zero
everywhere outside region `i`, so after the blend kernel normalizes, the owner always wins and the
seam stays hard. Widening a camera's influence past its own label needs a *signed* distance.

For each camera `i`, with `S(t) = clamp(t,0,1)^2 * (3 - 2*clamp(t,0,1))`:

```
phi_i = (label == i) ? -(inside_i - 0.5) : (outside_i - 0.5)
w_i   = S(0.5 - phi_i / R(p)) * S((coverage_i - 0.5) / (R(p) / 2))
w_i  /= sum_j w_j
```

`inside_i` and `outside_i` are Euclidean distance transforms of the label region and its
complement; `coverage_i` is the distance into camera `i`'s valid remap footprint.

Four details are load bearing:

- **The 0.5 offsets** put the 50% point exactly on the geometric boundary between the two pixel
  centres. Without them the two pixels flanking the seam step by twice the intended slope, which
  shows as a one-pixel line at small radii.
- **The coverage taper** (the second factor) is not optional. The blend kernels only drop
  zero-alpha contributors when `CHANNELS == 4`, and hstream's fp16 path is
  `CudaStitchPano<uchar4, half3>`. Without the taper, a feather reaching past a camera's footprint
  would mix in the memset black outside it and darken the seam by up to 30 code values.
- **Coverage-corrected labels.** Intersecting label with footprint would break the partition and
  leave pixels where every weight is zero, which sends the kernel down its `sumW == 0` branch.
  Instead, a pixel whose owner has no data is reassigned to the camera that covers it most deeply,
  keeping the partition total.
- **The radius is per pixel.**

      R(p) = min( R_requested, min{ max(2*coverage_i(p), 2*outside_i(p) - 1) : coverage_i(p) > 0 } )

  Camera `i` only bends the ramp at `p` if it is inside the band, and from the weight formula it is
  inside exactly when `outside_i(p) < R/2 + 0.5`. So `i` is satisfied by either of two things: `R`
  small enough to keep it out, which is `R <= 2*outside_i - 1`, or enough depth to survive its own
  taper, which is `R <= 2*coverage_i`. Taking the larger of the two per camera and the minimum
  across cameras gives the widest `R` that holds at `R` itself.

  That fixed-point framing is the point. Testing reach against `R_requested` instead is circular,
  and not conservatively so: a camera 60 px from the seam whose weight is exactly zero once the
  band narrows still pinned the cap, collapsing a 400 px crossfade to 4 px across an entire seam.
  Nor can the circularity be broken by iterating, because the map is discontinuous and
  non-increasing, so iterating oscillates between the two extremes rather than converging.

  Three other things here are load bearing. A single global cap does not survive real data: on a
  14220x4938 two-camera rig where 10.8% of the canvas is uncovered, a global minimum over the seam
  reduced a requested 439 px to 0 and turned alpha mode into a hard seam. Capping on "covered by at
  least two cameras" is wrong for N > 2, because a third camera blanketing the area inflates that
  count and the cap never engages while the two cameras meeting at the seam are strangled by their
  own tapers. And the coverage term is not optional: a camera that does not cover `p` has weight
  exactly zero there whatever `R` is, because its own taper reads `coverage_i == 0`, so without it
  a nearby seam-label island would drive the cap to zero and turn a healthy seam hard.

  Both cameras at a pixel divide by the same `R(p)`, so the partition-of-unity argument below still
  holds pointwise. `R(p)` cannot fall below a pixel: the distance transforms in use return at least
  one for any foreground pixel, so the cap leaves `R(p) >= min(2, R_requested)`, and a requested
  radius under one pixel degenerates the whole field before this point.

`S(t) + S(1-t) == 1` identically, so the *seam* factors of two cameras across a seam already sum
to one. Two things break that. Three or more overlapping regions, obviously; but also the coverage
factor, which is below one near a footprint edge (measured between 0.50 and 0.9966 for two
cameras, depending on how much of the band sits near an edge), so even the
two-image field needs normalizing. It is done here rather than in the kernels: only
`BatchedBlendKernelN` normalizes unconditionally, `BatchedBlendKernel3` does so only for
four-channel compute, and the two-image kernel takes a single-channel mask and synthesizes the
second weight as `1-m`. Normalizing once on the host makes all three consistent and makes the
kernels' own normalization a no-op.

The field is built once at construction, on the full padded canvas and then cropped. Distance
transforms are non-local, so building them from a blend-ROI crop would invent a boundary at the
crop edge.

## Feather width

`fraction` is a fraction of the narrowest camera footprint width, matching the `blend_width`
parameter this was ported from. It keeps its meaning across rigs and across `max_output_width`
rescaling, because the footprint rescales with the canvas. The default is 0.05; useful values are
well under 0.3.

The radius is the total ramp width in canvas pixels, centred on the seam, and it is capped per
pixel as described above. `feather_radius_px()` reports the *widest* crossfade any seam pixel got
and
`Result::min_seam_radius_px` the narrowest, which is what tells you whether some part of the seam
is effectively hard. Neither is a safe blend-ROI pad: the local radius grows away from a pinched
seam, so the ROI pads with `Result::requested_radius_px`, the width asked for before the cap.

`min_seam_radius_px` is a single worst pixel, and on a real rig it is almost always small: a
coverage
hole anywhere along the seam produces one. `Result::capped_seam_fraction` gives the share of seam
pixels that were narrowed at all, which is what separates a local pinch from a general one. On the
14220x4938 rig above the tightest pixel is 2 px while 97% of the seam keeps the full 439 px.
Construction logs both when the cap bites anywhere along the seam. A radius below one pixel
degenerates to the
exact one-hot partition built from the labels handed in, so `Alpha(0)` reproduces the hard-seam
*mask* byte for byte. The rendered output matches too on a rig where every label region sits
inside its owner's footprint; where it does not, the blend kernels' zero-alpha handling differs
from the hard-seam path and the two diverge, for Laplacian just as much as for alpha. There is no
per-pixel version of that fallback, because the cap cannot leave a local radius small enough to
need one.

`Params::max_px` (512 by default) clamps the requested width before the cap, so a very wide
footprint saturates rather than asking for an enormous ramp. That clamp is not reported through
`overlap_capped`, which only describes the coverage cap.

## Implementation

`cudaBatchedAlphaBlend`, `cudaBatchedAlphaBlend3` and `cudaBatchedAlphaBlendN` are level 0 of the
Laplacian path run once at full resolution: the same kernels, the same mask layouts, the same
alpha-validity semantics, with no pyramid, no context and no device allocation. They live in the
same translation units as the kernels they call, which are file-local (anonymous namespaces in the
three- and N-camera units).

Two things about the ROI are easy to get backwards, and both cost real pixels.

The ROI is sized from `Result::corrected_labels`, not from the seam handed in. Correction moves a
seam to the edge of any coverage hole inside an overlap, and a ROI built from the original labels
does not know that seam is there.

The coverage guard, `blend_roi::hard_baseline_covers_soft_owners_outside_write`, takes the
*original* labels instead. It protects the hard baseline in `cudaBlendHardSeam`, which is built
from the originals, and a corrected owner covers by construction, so checking corrected labels
would only ever fail where no camera covers at all.

The two are independent, not one subsuming the other, and the argument for the pair runs:
correction fires exactly where the original owner has no data and some other camera does, and the
guard fails exactly when
some pixel *outside* the write ROI has an original owner with no data. So a guard pass means
corrected and original agree everywhere outside the ROI, while the corrected-label ROI plus the
feather pad covers the whole band of every corrected seam inside it. The guard alone is not
enough, because it never looks inside the ROI: a hole landing there passes it, and the seam the
correction moves to that hole's edge still feathers outward. Sizing the ROI from the corrected
labels also grows it over such a hole, which can re-enable minimizing where the guard used to veto
it.

A consequence worth knowing: on a rig with uncovered canvas outside the write ROI the guard turns
minimizing off entirely. On `vegas-calheat-1`, 10.8% uncovered, expect the minimized path not to
engage.

Alpha mode has no pyramid, so `BlendSettings::roi_levels()` reports zero levels to
`blend_roi::select_regions`, which zeroes the blend ROI's pyramid margin. The write ROI is padded
by `max(overlap_padding, ceil(requested_radius_px/2) + 1)` so it covers the crossfade. The
request, not the seam maximum: the local radius grows away from a pinched seam, so the band can
reach far past the widest radius any seam pixel got; outside alpha mode that
extra term is zero, leaving existing callers' ROIs untouched.

`cupano/feather.py` mirrors the C++ step for step and calls the same `cv2.distanceTransform`.
`scripts/compare_feather_parity.py --all` checks the two element by element over seven rigs (2, 3
and 8 cameras, a blanketing third, a grazing zero-weight camera, a coverage hole, the hard
fallback) and reports bit-identical weights, labels and metadata on all of them. Expect one
float32 ULP on some fraction of elements on some builds rather than exact equality: OpenCV's
`DIST_MASK_PRECISE` is not reproducible run to run, so neither side can promise bit-identity.
`Params::fast` selects the 5x5
chamfer approximation, which differs between OpenCV builds by up to 3e-04 on its own, so the two
sides are not comparable there; leave it off when they have to agree. `cupano/pano.py` exposes the
same choice as `blend_mode="alpha"` plus
`feather_fraction`.

## Tests

- `//src/pano:featherMask_test` and `tests/python/test_feather.py` check the weight field on the
  CPU with no GPU: the closed-form ramp, partition of unity, exact saturation outside the band, a
  gradient bound, triple points splitting evenly and summing to one, zero weight outside coverage,
  the per-pixel overlap cap (including that a blanketing third camera does not defeat it), exact
  one-hot at zero width, and canvas-border independence.

  The gradient bound is `3/R` and is asserted only on fixtures with uniform coverage, where the
  local radius is constant. The field is `3/R_local`-Lipschitz, not `3/radius_px`, so where the cap
  varies — around coverage holes and thin overlaps — steps are legitimately much larger: a camera
  with no data at all must reach zero weight over however many pixels the hole spans, exactly as
  the hard-seam path does.
- `CudaPano{,3,N}AlphaTest` cover the stitchers end to end: zero feather reproduces the hard seam
  byte for byte, a constant input is preserved exactly, the crossfade is confined to the seam band,
  minimized and full-canvas blends agree, and compact and reference workspaces agree.
