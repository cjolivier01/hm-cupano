#pragma once

#include <opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

#include "cupano/pano/canvasManager.h"

namespace hm {
namespace pano {

// Minimal N-image canvas manager (focuses on placement; blending ROI kept full canvas for simplicity).
class CanvasManagerN {
 public:
  CanvasManagerN(CanvasInfo canvas_info, bool minimize_blend, int overlap_pad = 128)
      : canvas_info_(std::move(canvas_info)), _minimize_blend(minimize_blend), _overlap_pad(overlap_pad) {
    _remappers.resize(canvas_info_.positions.size());
  }

  void set_remap_size(int idx, const cv::Size& sz) {
    if (idx >= 0 && idx < static_cast<int>(_remappers.size())) {
      _remappers[idx].width = sz.width;
      _remappers[idx].height = sz.height;
      _remappers[idx].xpos = canvas_info_.positions[idx].x;
    }
  }

  // For now, blending ROI is the full canvas to keep logic generic.
  void updateMinimizeBlend(const std::vector<cv::Size>& /*remapped_sizes*/) {}

  // Pad seam masks to full canvas size (replicate right/bottom borders). We do
  // not crop here; the N-stitcher currently blends over the full canvas.
  cv::Mat convertMaskMat(const cv::Mat& mask) const {
    int padw = 0;
    int padh = 0;
    const int mwidth = mask.cols;
    const int mheight = mask.rows;

    assert(mwidth <= canvas_info_.width);
    assert(mheight <= canvas_info_.height);

    if (mwidth < canvas_info_.width) {
      padw = canvas_info_.width - mwidth;
    }
    if (mheight < canvas_info_.height) {
      padh = canvas_info_.height - mheight;
    }

    if (padw > 0 || padh > 0) {
      cv::Mat paddedMask;
      cv::copyMakeBorder(mask, paddedMask, 0, padh, 0, padw, cv::BORDER_REPLICATE);
      return paddedMask;
    }
    return mask;
  }

  constexpr int overlap_padding() const {
    return _overlap_pad;
  }
  constexpr int canvas_width() const {
    return canvas_info_.width;
  }
  constexpr int canvas_height() const {
    return canvas_info_.height;
  }
  constexpr const std::vector<cv::Point>& canvas_positions() const {
    return canvas_info_.positions;
  }

  const std::vector<Remapper>& remappers() const {
    return _remappers;
  }

 private:
  CanvasInfo canvas_info_;
  bool _minimize_blend{false};
  int _overlap_pad{0};
  std::vector<Remapper> _remappers;
};

} // namespace pano
} // namespace hm
