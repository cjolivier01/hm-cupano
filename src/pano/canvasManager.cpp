#include "canvasManager.h"

namespace hm {
namespace pano {

CanvasManager::CanvasManager(CanvasInfo canvas_info, bool minimize_blend, int overlap_pad)
    : _x1(0),
      _y1(0),
      _x2(0),
      _y2(0),
      canvas_info_(canvas_info),
      _overlapping_width(0),
      _minimize_blend(minimize_blend),
      _overlap_pad(overlap_pad) {}

void CanvasManager::updateMinimizeBlend(const cv::Size& remapped_size_1, const cv::Size& remapped_size_2) {
  // Ensure that we have at least two positions in the canvas info.
  assert(canvas_info_.positions.size() >= 2);

  // Extract the positions where images will be placed.
  _x1 = canvas_info_.positions[0].x;
  _y1 = canvas_info_.positions[0].y;
  _x2 = canvas_info_.positions[1].x;
  _y2 = canvas_info_.positions[1].y;

  // Compute the overlap width based on the first image's width minus the X of the second image.
  int width_1 = _remapper_1.width;
  _overlapping_width = width_1 - _x2;
  assert(width_1 > _x2); // Must have positive overlap.

  if (_minimize_blend) {
    // Assign X offsets for the two remappers for minimal blending region.
    _remapper_1.xpos = _x1;
    // Start overlapping right away in the second remapper.
    _remapper_2.xpos = _x1 + _overlap_pad;

    // Define the seam box (the region to be blended) with extra padding.
    int box_x1 = _x2 - _overlap_pad;
    int box_y1 = std::max(0, std::min(_y1, _y2) - _overlap_pad);
    int box_x2 = width_1 + _overlap_pad;
    int box_y2 =
        std::min(canvas_info_.height, std::max(_y1 + _remapper_1.height, _y2 + _remapper_2.height) + _overlap_pad);
    _padded_blended_tlbr = {box_x1, box_y1, box_x2, box_y2};

    // Validate the seam box within the canvas boundaries.
    assert(box_x1 >= 0);
    assert(box_x2 <= canvas_info_.width);

    // Compute ROIs for the blend region in each image.
    roi_blend_1 = {_x2 - _overlap_pad, 0, remapped_size_1.width - _x2 - _overlap_pad, remapped_size_1.height};
    roi_blend_2 = {0, 0, _overlapping_width + _overlap_pad, remapped_size_2.height};
  }
}

cv::Mat CanvasManager::convertMaskMat(const cv::Mat& mask) {
  int padw = 0, padh = 0;
  int mwidth = mask.cols;
  int mheight = mask.rows;

  // Ensure mask is not larger than the canvas.
  assert(mwidth <= canvas_info_.width);
  assert(mheight <= canvas_info_.height);

  // Determine horizontal/vertical padding for the mask.
  if (mwidth < canvas_info_.width)
    padw = canvas_info_.width - mwidth;
  if (mheight < canvas_info_.height)
    padh = canvas_info_.height - mheight;

  cv::Mat paddedMask;
  if (padw > 0 || padh > 0) {
    // Replicate border pixels on right/bottom to expand the mask.
    cv::copyMakeBorder(mask, paddedMask, 0, padh, 0, padw, cv::BORDER_REPLICATE);
  } else {
    paddedMask = mask;
  }

  // Double-check final dimensions.
  assert(paddedMask.cols == canvas_info_.width);
  assert(paddedMask.rows == canvas_info_.height);

  // If minimizing blend, crop the mask to the overlap region.
  if (_minimize_blend) {
    int x_start = canvas_info_.positions[1].x - _overlap_pad;
    int x_end = _remapper_1.width + _overlap_pad;
    assert(x_start >= 0 && x_end <= paddedMask.cols);

    cv::Rect roi(x_start, 0, x_end - x_start, paddedMask.rows);
    return paddedMask(roi);
  }

  return paddedMask;
}

} // namespace pano
} // namespace hm
