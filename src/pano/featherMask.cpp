#include "cupano/pano/featherMask.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <limits>

namespace hm {
namespace pano {
namespace feather {

namespace {

constexpr uint16_t kUnmappedPositionValue = 65535;
// Below one pixel the crossfade cannot be represented and the mask degenerates to the hard seam.
constexpr float kMinUsefulRadius = 1.0f;

int distance_mask(bool fast) {
  return fast ? cv::DIST_MASK_5 : cv::DIST_MASK_PRECISE;
}

// smoothstep(clamp(t, 0, 1)), in place.
void smoothstep_inplace(cv::Mat& t) {
  cv::min(t, 1.0, t);
  cv::max(t, 0.0, t);
  const cv::Mat ramp = 3.0 - 2.0 * t;
  t = t.mul(t).mul(ramp);
}

cv::Mat one_hot_planes(const cv::Mat& labels, int n_images) {
  std::vector<cv::Mat> planes(n_images);
  for (int i = 0; i < n_images; ++i) {
    planes[i] = cv::Mat::zeros(labels.size(), CV_32F);
    planes[i].setTo(1.0f, labels == i);
  }
  cv::Mat merged;
  cv::merge(planes, merged);
  return merged;
}

} // namespace

std::vector<cv::Mat> coverage_masks(
    cv::Size canvas,
    const std::vector<cv::Mat>& remap_x,
    const std::vector<cv::Mat>& remap_y,
    const std::vector<cv::Point>& positions) {
  const int n = static_cast<int>(remap_x.size());
  std::vector<cv::Mat> coverage(n);
  for (int i = 0; i < n; ++i) {
    coverage[i] = cv::Mat::zeros(canvas, CV_8U);
    const cv::Rect placed(positions[i].x, positions[i].y, remap_x[i].cols, remap_x[i].rows);
    const cv::Rect clipped = placed & cv::Rect(0, 0, canvas.width, canvas.height);
    if (clipped.area() <= 0) {
      continue;
    }
    const cv::Point local(clipped.x - placed.x, clipped.y - placed.y);
    cv::Mat dest = coverage[i](clipped);
    for (int y = 0; y < clipped.height; ++y) {
      const uint16_t* mx = remap_x[i].ptr<uint16_t>(local.y + y) + local.x;
      const uint16_t* my = remap_y[i].ptr<uint16_t>(local.y + y) + local.x;
      uint8_t* out = dest.ptr<uint8_t>(y);
      for (int x = 0; x < clipped.width; ++x) {
        out[x] = (mx[x] != kUnmappedPositionValue && my[x] != kUnmappedPositionValue) ? 255 : 0;
      }
    }
  }
  return coverage;
}

Result build_weights(
    const cv::Mat& seam_index,
    const std::vector<cv::Mat>& remap_x,
    const std::vector<cv::Mat>& remap_y,
    const std::vector<cv::Point>& positions,
    int n_images,
    const Params& params) {
  Result result;
  if (seam_index.empty() || seam_index.type() != CV_8U) {
    result.error = "Feather mask requires a CV_8U seam index image";
    return result;
  }
  if (n_images < 2 || static_cast<int>(remap_x.size()) != n_images || static_cast<int>(remap_y.size()) != n_images ||
      static_cast<int>(positions.size()) != n_images) {
    result.error = "Feather mask input sizes do not match n_images";
    return result;
  }
  if (!(params.fraction >= 0.0f) || !(params.max_px >= 0.0f)) {
    result.error = "Feather mask requires a non-negative fraction and max_px";
    return result;
  }
  // coverage_masks reads these through raw row pointers, so check the element type and that the
  // two maps agree in size before indexing either.
  for (int i = 0; i < n_images; ++i) {
    if (remap_x[i].type() != CV_16U || remap_y[i].type() != CV_16U) {
      result.error = "Feather mask requires CV_16U remap maps";
      return result;
    }
    if (remap_x[i].size() != remap_y[i].size()) {
      result.error = "Feather mask remap x and y maps must be the same size";
      return result;
    }
  }

  double max_label = 0.0;
  cv::minMaxLoc(seam_index, nullptr, &max_label);
  if (max_label >= static_cast<double>(n_images)) {
    result.error = "Feather mask seam index contains a label outside [0, n_images)";
    return result;
  }

  // Width is a fraction of the narrowest camera footprint, so it scales with the canvas.
  int narrowest = std::numeric_limits<int>::max();
  for (int i = 0; i < n_images; ++i) {
    narrowest = std::min(narrowest, remap_x[i].cols);
  }
  const float radius = std::min(params.fraction * static_cast<float>(narrowest), params.max_px);
  if (!(radius >= kMinUsefulRadius)) {
    // Subpixel widths reproduce the hard seam exactly, including its uncorrected labels.
    // No coverage or distance fields are needed for this one-hot partition.
    result.weights = one_hot_planes(seam_index, n_images);
    result.corrected_labels = seam_index;
    result.hard = true;
    return result;
  }

  const cv::Size canvas = seam_index.size();
  const int mask_type = distance_mask(params.fast);

  // Per-camera coverage, and the distance into it. Used to taper each camera at its own footprint
  // edge, and to cap the radius at what the real overlap can support.
  const std::vector<cv::Mat> coverage = coverage_masks(canvas, remap_x, remap_y, positions);
  std::vector<cv::Mat> coverage_distance(n_images);
  for (int i = 0; i < n_images; ++i) {
    cv::distanceTransform(coverage[i], coverage_distance[i], cv::DIST_L2, mask_type);
  }

  // Coverage-corrected labels. Where a pixel's owner has no source data, hand it to the camera that
  // covers it most deeply. This keeps the label image a total partition, which is what guarantees
  // the blend kernel always sees a non-zero weight sum. Pixels no camera covers keep their original
  // label; every contributor there is transparent, so the kernel emits transparent black.
  cv::Mat labels = seam_index.clone();
  for (int y = 0; y < canvas.height; ++y) {
    uint8_t* row = labels.ptr<uint8_t>(y);
    for (int x = 0; x < canvas.width; ++x) {
      const int owner = row[x];
      if (owner < n_images && coverage[owner].at<uint8_t>(y, x)) {
        continue;
      }
      int best = -1;
      float best_depth = 0.0f;
      for (int i = 0; i < n_images; ++i) {
        if (!coverage[i].at<uint8_t>(y, x)) {
          continue;
        }
        const float depth = coverage_distance[i].at<float>(y, x);
        if (best < 0 || depth > best_depth) {
          best = i;
          best_depth = depth;
        }
      }
      if (best >= 0) {
        row[x] = static_cast<uint8_t>(best);
      }
    }
  }

  // Cap the crossfade per pixel at what the cameras that actually contribute there can support.
  //
  // A global minimum does not survive real data: it samples pixels no camera covers (whose owner
  // depth is zero) and lets one pinhole collapse the whole canvas. On a 14220x4938 two-camera rig
  // that reduced a requested 439 px to 0 and turned alpha mode into a hard seam.
  //
  // Each camera must either have enough coverage depth for the resulting local radius or lie
  // outside its reach. The bound below resolves both conditions at that same radius.
  //
  // Both cameras at a pixel divide by the same radius, so S(t) + S(1-t) == 1 still holds pointwise.
  cv::Mat cap(canvas, CV_32F, cv::Scalar(std::numeric_limits<float>::max()));
  // Recomputed rather than cached for the weight loop below: one canvas float32 per camera is
  // 2.1 GiB at N=8 on a 14220x4938 canvas, against about 5% of a one-time construction step.
  for (int i = 0; i < n_images; ++i) {
    cv::Mat not_owner;
    cv::bitwise_not(labels == i, not_owner);
    cv::Mat outside;
    cv::distanceTransform(not_owner, outside, cv::DIST_L2, mask_type);

    // Camera i only bends the ramp at p if it is inside the band, and from the weight formula
    // below it is inside exactly when outside_i < R/2 + 0.5. So i is satisfied by either of two
    // things: R small enough to keep it out (R <= 2*outside_i - 1), or enough depth to survive
    // its own taper (R <= 2*coverage_i). Taking the larger of the two per camera and the minimum
    // across cameras is the widest R that holds at R itself.
    //
    // Testing reach against the *requested* radius instead would be circular, and not
    // conservatively so: a camera 60 px away whose weight is exactly zero once the band narrows
    // would still pin the cap, collapsing a 400 px seam to 4 px.
    cv::Mat allowed = coverage_distance[i] * 2.0;
    cv::Mat out_of_reach = outside * 2.0 - 1.0;
    cv::max(allowed, out_of_reach, allowed);

    // A camera with no data at p contributes nothing there for any R, because its taper reads
    // coverage_distance == 0. Letting it into the minimum would strangle a good seam; a
    // seam-label island near another seam is enough to trigger that.
    cv::Mat candidate = cap.clone();
    allowed.copyTo(candidate, coverage[i]);
    cv::min(cap, candidate, cap);
  }

  // cap holds a radius already, and stays at FLT_MAX where no camera covers.
  cv::Mat radius_map;
  cv::min(cap, static_cast<double>(radius), radius_map);

  // Report against the seam, where the crossfade actually happens. A canvas-wide maximum would
  // stay at the requested width even when every seam pixel is pinched.
  cv::Mat seam_boundary = cv::Mat::zeros(canvas, CV_8U);
  for (int y = 0; y < canvas.height; ++y) {
    const uint8_t* row = labels.ptr<uint8_t>(y);
    const uint8_t* below = (y + 1 < canvas.height) ? labels.ptr<uint8_t>(y + 1) : nullptr;
    uint8_t* mark = seam_boundary.ptr<uint8_t>(y);
    for (int x = 0; x < canvas.width; ++x) {
      // Both sides, matching blend_roi::seam_boundary_bbox. A one-sided mask makes the reported
      // radius depend on which way the label happens to change.
      if (x + 1 < canvas.width && row[x + 1] != row[x]) {
        mark[x] = 255;
        mark[x + 1] = 255;
      }
      if (below != nullptr && below[x] != row[x]) {
        mark[x] = 255;
        seam_boundary.ptr<uint8_t>(y + 1)[x] = 255;
      }
    }
  }
  double seam_min = 0.0;
  double seam_max = 0.0;
  const bool has_seam = cv::countNonZero(seam_boundary) > 0;
  if (has_seam) {
    cv::minMaxLoc(radius_map, &seam_min, &seam_max, nullptr, nullptr, seam_boundary);
  } else {
    cv::minMaxLoc(radius_map, nullptr, &seam_max);
    seam_min = seam_max;
  }
  result.corrected_labels = labels;
  result.requested_radius_px = radius;
  result.radius_px = static_cast<float>(seam_max);
  result.min_seam_radius_px = static_cast<float>(seam_min);
  result.overlap_capped = seam_min < static_cast<double>(radius);
  if (has_seam) {
    cv::Mat capped_at_seam;
    cv::compare(radius_map, static_cast<double>(radius), capped_at_seam, cv::CMP_LT);
    cv::bitwise_and(capped_at_seam, seam_boundary, capped_at_seam);
    result.capped_seam_fraction =
        static_cast<float>(cv::countNonZero(capped_at_seam)) / static_cast<float>(cv::countNonZero(seam_boundary));
  }

  // Both supported distance transforms give coverage depth >= 1 wherever a camera covers.
  // The cap is therefore >= 2, and the requested radius already passed the >= 1 check above.
  cv::Mat half_radius_map = radius_map * 0.5;

  std::vector<cv::Mat> planes(n_images);
  for (int i = 0; i < n_images; ++i) {
    const cv::Mat is_owner = (labels == i);
    cv::Mat inside;
    cv::distanceTransform(is_owner, inside, cv::DIST_L2, mask_type);
    cv::Mat not_owner;
    cv::bitwise_not(is_owner, not_owner);
    cv::Mat outside;
    cv::distanceTransform(not_owner, outside, cv::DIST_L2, mask_type);

    // Signed distance, positive outside region i, with the half-pixel offset that puts the 50%
    // point on the geometric boundary. Selected rather than subtracted: an empty or canvas-filling
    // region saturates one of the transforms at about 1.8e19. Selecting rather than subtracting
    // also keeps the half-pixel offset on the owner side, which subtracting would lose.
    cv::Mat phi = outside - 0.5;
    const cv::Mat inside_signed = 0.5 - inside;
    inside_signed.copyTo(phi, is_owner);

    cv::Mat seam_weight;
    cv::divide(phi, radius_map, seam_weight);
    seam_weight = 0.5 - seam_weight;
    smoothstep_inplace(seam_weight);

    // Taper at this camera's own footprint edge. The per-pixel cap keeps the band inside coverage
    // the contributors share, so this is close to 1 there but not exactly 1 - the half-pixel offset
    // means it cannot reach 1 at the cap boundary, and host normalization below is what makes the
    // weights sum to one anyway. It is load bearing outside the band and for N > 2, and it is what
    // keeps a camera from contributing outside its footprint on three-channel compute types, where
    // the blend kernels have no zero-alpha fallback.
    cv::Mat coverage_weight;
    cv::divide(coverage_distance[i] - 0.5, half_radius_map, coverage_weight);
    smoothstep_inplace(coverage_weight);

    planes[i] = seam_weight.mul(coverage_weight);
  }

  // Normalize here rather than leaving it to the kernels. Only BatchedBlendKernelN normalizes;
  // BatchedBlendKernel3 does so only for four-channel compute, and the two-image kernel never does
  // - it takes a single-channel mask and synthesizes the second weight as 1-m, which would discard
  // the other camera's taper entirely. Normalizing once here makes all three consistent, and makes
  // the kernels' own normalization a no-op.
  cv::Mat total = cv::Mat::zeros(canvas, CV_32F);
  for (const cv::Mat& plane : planes) {
    total += plane;
  }
  // cv::divide yields NaN for 0/0 on floats rather than zero, so substitute a unit divisor where
  // nothing covers. Every plane is already zero there, so the result stays zero.
  cv::Mat divisor = total.clone();
  divisor.setTo(1.0f, total == 0.0f);
  for (cv::Mat& plane : planes) {
    cv::divide(plane, divisor, plane);
  }

  cv::merge(planes, result.weights);
  return result;
}

} // namespace feather
} // namespace pano
} // namespace hm
