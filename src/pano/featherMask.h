#pragma once

#include <opencv2/core.hpp>
#include <string>
#include <vector>

namespace hm {
namespace pano {
namespace feather {

/**
 * @brief Inputs controlling the feathered crossfade width.
 */
struct Params {
  /// Crossfade width as a fraction of the narrowest camera footprint width.
  float fraction = 0.05f;
  /// Absolute clamp on the resulting radius, in canvas pixels.
  float max_px = 512.0f;
  /// Use the 5x5 chamfer approximation (~2% error) instead of the exact Euclidean transform.
  bool fast = false;
};

/**
 * @brief A per-camera blend weight field plus the radius that was actually used.
 */
struct Result {
  /// CV_32FC(n_images), canvas sized, interleaved [H x W x N]. Empty when `error` is set.
  cv::Mat weights;
  /// Widest crossfade any seam pixel got, in canvas pixels. Total ramp width, centred on the seam.
  float radius_px = 0.0f;
  /// The width that was asked for, after max_px but before the coverage cap. The local radius can
  /// exceed radius_px away from a pinched seam, so this is the only safe bound for a blend-ROI pad.
  float requested_radius_px = 0.0f;
  /// Narrowest crossfade any seam pixel got. Below about 4 px the transition is visually hard.
  float min_seam_radius_px = 0.0f;
  /// True when any seam pixel was reduced to fit inside its contributors' shared coverage.
  bool overlap_capped = false;
  /// Share of seam pixels that were reduced, in [0, 1]. `min_seam_radius_px` is a single worst
  /// pixel and is usually tiny on a real rig; this says whether that is a curiosity or the norm.
  float capped_seam_fraction = 0.0f;
  /// True when the radius collapsed below a pixel and `weights` is the exact one-hot partition.
  bool hard = false;
  /// The labels the weight field was actually built from: `seam_index` with every pixel whose
  /// owner has no data there reassigned to the camera covering it most deeply. A coverage hole
  /// inside an overlap moves a seam, so a blend ROI derived from the original labels can miss the
  /// crossfade entirely. Empty on the error paths. On the hard fallback it is the caller's own
  /// `seam_index`, buffer included, because that path deliberately discards the correction, so
  /// treat it as read-only.
  cv::Mat corrected_labels;
  /// Non-empty when the inputs were unusable.
  std::string error;
};

/**
 * @brief Builds feathered per-camera blend weights from a hard seam label image.
 *
 * The seam label image partitions the canvas: every pixel names exactly one owning camera. A weight
 * field derived from `distanceTransform(label == i)` alone is inert, because that distance is zero
 * everywhere outside region `i`, so after the blend kernel normalizes, the owner always wins and the
 * seam stays hard. Widening a camera's influence past its own label therefore needs a *signed*
 * distance:
 *
 *     phi_i = (label == i) ? -(inside_i - 0.5) : (outside_i - 0.5)
 *     w_i   = S(0.5 - phi_i / R) * S((coverage_i - 0.5) / (R / 2))
 *
 * with `S` the smoothstep `t*t*(3-2t)` on a clamped argument. `S(t) + S(1-t) == 1` identically, so
 * the *seam* factors of two cameras across a seam already sum to one. The coverage factor breaks
 * that: near a footprint edge it is below one (measured between 0.50 and 0.9966 for two cameras, depending on how much
 * of the band sits near an edge), so the field needs normalizing at N = 2 as well, not only where three or more regions
 * meet.
 *
 * The 0.5 offsets place the 50% point exactly on the geometric boundary between the two pixel
 * centres. Without them the two pixels flanking the seam step by twice the intended slope, which is
 * visible as a one-pixel line at small radii.
 *
 * The second factor tapers each camera to zero at the edge of its own remap footprint. It is not
 * optional: the blend kernels only drop zero-alpha contributors when CHANNELS == 4, so on a
 * three-channel compute type a feather that reached past a camera's footprint would mix in the
 * memset black outside it and darken the seam.
 *
 * @param seam_index  CV_8U labels in [0, n_images), canvas sized. Must be the padded, canonicalized
 *                    seam (i.e. the output of CanvasManagerN::convertMaskMat), never a blend-ROI
 *                    crop: distance transforms are non-local, so cropping first produces a hard edge
 *                    at the crop boundary. Crop the returned weights instead.
 * @param remap_x     Per-camera CV_16U column maps; 65535 marks an unmapped source pixel.
 * @param remap_y     Per-camera CV_16U row maps.
 * @param positions   Per-camera top-left placement of its remap on the canvas.
 * @param n_images    Number of cameras.
 * @param params      Width controls.
 */
Result build_weights(
    const cv::Mat& seam_index,
    const std::vector<cv::Mat>& remap_x,
    const std::vector<cv::Mat>& remap_y,
    const std::vector<cv::Point>& positions,
    int n_images,
    const Params& params);

/// Canvas-sized CV_8U coverage masks (255 where camera i has a mapped source pixel). Exposed for
/// tests and diagnostics.
std::vector<cv::Mat> coverage_masks(
    cv::Size canvas,
    const std::vector<cv::Mat>& remap_x,
    const std::vector<cv::Mat>& remap_y,
    const std::vector<cv::Point>& positions);

} // namespace feather
} // namespace pano
} // namespace hm
