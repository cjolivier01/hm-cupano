#pragma once

#include <opencv2/core.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace hm {
namespace pano {
namespace blend_roi {

struct Regions {
  cv::Rect blend;
  cv::Rect write;
};

struct RemapRoi {
  // ROI within the remap map, in image-local coordinates.
  cv::Rect roi;
  // Destination offset for the full remap map, in blend-ROI coordinates.
  int offset_x{0};
  int offset_y{0};
};

inline int scaled_overlap_padding(size_t original_canvas_width, int scaled_canvas_width, int base_pad = 128) {
  if (original_canvas_width == 0 || scaled_canvas_width <= 0 ||
      static_cast<size_t>(scaled_canvas_width) >= original_canvas_width) {
    return base_pad;
  }
  return std::max(
      1,
      static_cast<int>(std::floor(
          static_cast<double>(base_pad) * static_cast<double>(scaled_canvas_width) /
          static_cast<double>(original_canvas_width))));
}

inline int pyramid_margin(int num_levels) {
  // Conservative margin in level-0 pixels to keep pyramid boundary effects out of the write-back ROI.
  if (num_levels <= 0)
    return 0;
  const int shift = std::min(num_levels, 30);
  return 1 << shift;
}

inline int pyramid_alignment(int num_levels) {
  // The Laplacian pyramid downsamples by 2 for each level > 0. Aligning the ROI top-left to this phase
  // makes an ROI-local pyramid sample the same level coordinates as a full-canvas pyramid.
  if (num_levels <= 1)
    return 1;
  const int shift = std::min(num_levels - 1, 30);
  return 1 << shift;
}

inline int effective_pyramid_levels(int width, int height, int requested_levels) {
  if (width <= 0 || height <= 0 || requested_levels <= 0)
    return 0;

  int level_width = width;
  int level_height = height;
  for (int level = 1; level < requested_levels; ++level) {
    if (level_width < 2 || level_height < 2)
      return level;
    level_width = (level_width + 1) / 2;
    level_height = (level_height + 1) / 2;
  }
  return requested_levels;
}

inline std::optional<cv::Rect> seam_boundary_bbox(const cv::Mat& seam_index) {
  CV_Assert(seam_index.type() == CV_8U);
  const int w = seam_index.cols;
  const int h = seam_index.rows;
  if (w <= 0 || h <= 0)
    return std::nullopt;

  int min_x = w;
  int min_y = h;
  int max_x = -1;
  int max_y = -1;

  const auto update = [&](int x, int y) {
    min_x = std::min(min_x, x);
    min_y = std::min(min_y, y);
    max_x = std::max(max_x, x);
    max_y = std::max(max_y, y);
  };

  for (int y = 0; y < h; ++y) {
    const uint8_t* row = seam_index.ptr<uint8_t>(y);
    const uint8_t* row_down = (y + 1 < h) ? seam_index.ptr<uint8_t>(y + 1) : nullptr;
    for (int x = 0; x < w; ++x) {
      const uint8_t value = row[x];
      if (x + 1 < w && value != row[x + 1]) {
        update(x, y);
        update(x + 1, y);
      }
      if (row_down && value != row_down[x]) {
        update(x, y);
        update(x, y + 1);
      }
    }
  }

  if (max_x < 0 || max_y < 0)
    return std::nullopt;
  return cv::Rect(min_x, min_y, max_x - min_x + 1, max_y - min_y + 1);
}

inline cv::Rect expand_and_clamp(const cv::Rect& rect, int pad, int max_w, int max_h) {
  if (rect.width <= 0 || rect.height <= 0)
    return {};
  const int x0 = std::max(0, rect.x - pad);
  const int y0 = std::max(0, rect.y - pad);
  const int x1 = std::min(max_w, rect.x + rect.width + pad);
  const int y1 = std::min(max_h, rect.y + rect.height + pad);
  return cv::Rect(x0, y0, std::max(0, x1 - x0), std::max(0, y1 - y0));
}

inline cv::Rect align_and_clamp(const cv::Rect& rect, int align, int max_w, int max_h) {
  if (rect.width <= 0 || rect.height <= 0)
    return {};
  if (align <= 1)
    return rect;

  const int x0 = std::max(0, (rect.x / align) * align);
  const int y0 = std::max(0, (rect.y / align) * align);
  const int x1_unclamped = ((rect.x + rect.width + align - 1) / align) * align;
  const int y1_unclamped = ((rect.y + rect.height + align - 1) / align) * align;
  const int x1 = std::min(max_w, x1_unclamped);
  const int y1 = std::min(max_h, y1_unclamped);
  return cv::Rect(x0, y0, std::max(0, x1 - x0), std::max(0, y1 - y0));
}

inline Regions select_regions(const cv::Mat& seam_index, int num_levels, int overlap_pad) {
  const auto boundary_bbox = seam_boundary_bbox(seam_index);
  if (!boundary_bbox.has_value())
    return {};

  Regions regions;
  regions.write = expand_and_clamp(*boundary_bbox, overlap_pad, seam_index.cols, seam_index.rows);
  regions.blend = expand_and_clamp(regions.write, pyramid_margin(num_levels), seam_index.cols, seam_index.rows);
  regions.blend = align_and_clamp(regions.blend, pyramid_alignment(num_levels), seam_index.cols, seam_index.rows);
  const int64_t blend_area = static_cast<int64_t>(regions.blend.width) * regions.blend.height;
  const int64_t canvas_area = static_cast<int64_t>(seam_index.cols) * seam_index.rows;
  // A minimized frame also pays for a full-canvas hard-seam baseline. Reject near-full ROIs where the
  // saved pyramid/remap work is too small to justify that extra pass.
  constexpr int64_t kMaxBlendAreaPercent = 90;
  if (blend_area * 100 >= canvas_area * kMaxBlendAreaPercent ||
      effective_pyramid_levels(regions.blend.width, regions.blend.height, num_levels) !=
          effective_pyramid_levels(seam_index.cols, seam_index.rows, num_levels)) {
    return {};
  }
  return regions;
}

inline RemapRoi remap_roi(const cv::Point& position, const cv::Size& size, const cv::Rect& blend_roi_canvas) {
  RemapRoi result;
  result.offset_x = position.x - blend_roi_canvas.x;
  result.offset_y = position.y - blend_roi_canvas.y;

  const cv::Rect intersection = cv::Rect(position, size) & blend_roi_canvas;
  if (intersection.width > 0 && intersection.height > 0) {
    result.roi =
        cv::Rect(intersection.x - position.x, intersection.y - position.y, intersection.width, intersection.height);
  }
  return result;
}

inline bool hard_baseline_covers_soft_owners_outside_write(
    const cv::Mat& seam_index,
    const std::vector<cv::Point>& positions,
    const std::vector<cv::Mat>& remap_x,
    const std::vector<cv::Mat>& remap_y,
    const cv::Rect& write_roi_canvas) {
  CV_Assert(seam_index.type() == CV_8U);
  CV_Assert(positions.size() == remap_x.size() && positions.size() == remap_y.size());
  constexpr uint16_t kUnmappedPositionValue = 65535;

  for (int y = 0; y < seam_index.rows; ++y) {
    const uint8_t* seam_row = seam_index.ptr<uint8_t>(y);
    for (int x = 0; x < seam_index.cols; ++x) {
      if (write_roi_canvas.contains(cv::Point(x, y)))
        continue;

      const size_t owner = seam_row[x];
      if (owner >= positions.size())
        return false;
      const int local_x = x - positions[owner].x;
      const int local_y = y - positions[owner].y;
      if (local_x < 0 || local_y < 0 || local_x >= remap_x[owner].cols || local_y >= remap_x[owner].rows ||
          remap_x[owner].size() != remap_y[owner].size()) {
        return false;
      }
      if (remap_x[owner].ptr<uint16_t>(local_y)[local_x] == kUnmappedPositionValue ||
          remap_y[owner].ptr<uint16_t>(local_y)[local_x] == kUnmappedPositionValue) {
        return false;
      }
    }
  }
  return true;
}

} // namespace blend_roi
} // namespace pano
} // namespace hm
