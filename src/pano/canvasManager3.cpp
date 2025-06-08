#include "canvasManager3.h"
#include <algorithm>
#include <cassert>

namespace hm {
namespace pano {

CanvasManager3::CanvasManager3(CanvasInfo canvas_info, bool minimize_blend, int overlap_pad)
    : canvas_info_(std::move(canvas_info)), _minimize_blend(minimize_blend), _overlap_pad(overlap_pad) {
  assert(canvas_info_.positions.size() == 3);
}

void CanvasManager3::updateMinimizeBlend(
    const cv::Size& remapped_size_0,
    const cv::Size& remapped_size_1,
    const cv::Size& remapped_size_2) {
  assert(canvas_info_.positions.size() == 3);

  int x0 = canvas_info_.positions[0].x;
  int y0 = canvas_info_.positions[0].y;
  int x1 = canvas_info_.positions[1].x;
  int y1 = canvas_info_.positions[1].y;
  int x2 = canvas_info_.positions[2].x;
  int y2 = canvas_info_.positions[2].y;

  _remapper_0.width = remapped_size_0.width;
  _remapper_0.height = remapped_size_0.height;
  _remapper_1.width = remapped_size_1.width;
  _remapper_1.height = remapped_size_1.height;
  _remapper_2.width = remapped_size_2.width;
  _remapper_2.height = remapped_size_2.height;

  // Compute the overlap between image0 and image1:
  // Overlap occurs when the right edge of remapped0 (x0 + w0) > x1
  if (_remapper_0.width + x0 > x1) {
    _overlapping_width01 = (_remapper_0.width + x0) - x1;
    assert(_overlapping_width01 > 0);
  } else {
    _overlapping_width01 = 0;
  }

  // Likewise, overlap between image1 and image2:
  if (_remapper_1.width + x1 > x2) {
    _overlapping_width12 = (_remapper_1.width + x1) - x2;
    assert(_overlapping_width12 > 0);
  } else {
    _overlapping_width12 = 0;
  }

  if (false && _minimize_blend) {
    // 1) Where to place each remapper’s “xpos”:
    _remapper_0.xpos = x0; // image0 goes at (x0,y0)
    _remapper_1.xpos = x1; // image1 at (x1,y1)
    _remapper_2.xpos = x2; // image2 at (x2,y2)

    // 2) For the ROI in remapped0 vs. remapped1:
    //    The blending strip for (0 vs. 1) has width (_overlapping_width01 + 2*_overlap_pad),
    //    and sits horizontally at “x1 - _overlap_pad” (in remapped0’s coords).
    if (_overlapping_width01 > 0) {
      int blendW01 = _overlapping_width01 + 2 * _overlap_pad;
      int start01_x = x1 - _overlap_pad - x0; // in 0’s local coords
      remapped_image_roi_blend_0 = {start01_x, 0, blendW01, remapped_size_0.height};
      remapped_image_roi_blend_1 = {0, 0, blendW01, remapped_size_1.height};
    } else {
      remapped_image_roi_blend_0 = {0, 0, 0, 0};
      remapped_image_roi_blend_1 = {0, 0, 0, 0};
    }

    // 3) For the ROI in remapped1 vs. remapped2:
    if (_overlapping_width12 > 0) {
      int blendW12 = _overlapping_width12 + 2 * _overlap_pad;
      int start12_x = x2 - _overlap_pad - x1; // in 1’s local coords
      remapped_image_roi_blend_1 = {
          std::max(remapped_image_roi_blend_1.x, start12_x), 0, blendW12, remapped_size_1.height};
      remapped_image_roi_blend_2 = {0, 0, blendW12, remapped_size_2.height};
    } else {
      remapped_image_roi_blend_2 = {0, 0, 0, 0};
    }

    // 4) Finally, where should we place the **3‐way** blended region on the canvas?
    //    We take the intersection of the two overlap strips (0|1) and (1|2) in **canvas coords**:
    // int global_blend_x0 = std::max(x0 + remapped_image_roi_blend_0.x, x1 + remapped_image_roi_blend_1.x);
    int global_blend_y0 = std::max(std::max(y0, y1), y2); // assume y0,y1,y2 share same vertical alignment
    // width = intersection of [x1 - pad, x0 + w0 + pad] with [x2 - pad, x1 + w1 + pad]
    int left_bound = std::max(x1 - _overlap_pad, x2 - _overlap_pad);
    // int right_bound = std::min((x0 + _remapper_0.width) + _overlap_pad, (x1 + _remapper_1.width) + _overlap_pad);

    left_bound = std::max(0, left_bound);

    // int blendW = std::max(0, right_bound - left_bound);
    _x_blend_start = left_bound;
    _y_blend_start = global_blend_y0;
  } else {

    // 1) Where to place each remapper’s “xpos”:
    _remapper_0.xpos = x0; // image0 goes at (x0,y0)
    _remapper_1.xpos = x1; // image1 at (x1,y1)
    _remapper_2.xpos = x2; // image2 at (x2,y2)

    remapped_image_roi_blend_0 = {0, 0, remapped_size_0.width, remapped_size_0.height};
    remapped_image_roi_blend_1 = {0, 0, remapped_size_1.width, remapped_size_1.height};
    remapped_image_roi_blend_2 = {0, 0, remapped_size_2.width, remapped_size_2.height};

    _x_blend_start = 0;
    _y_blend_start = 0;
  }
}

cv::Mat CanvasManager3::convertMaskMat(const cv::Mat& mask) {
  // The input “mask” is assumed to be a 3‐channel image already of size (canvas_width × canvas_height).
  // If _minimize_blend is true, we crop to the 3‐way overlapping rectangular region:
  if (_minimize_blend) {
    int x_start = _x_blend_start;
    int y_start = _y_blend_start;
    int w = 0, h = 0;
    // The vertical extent is just the maximum height over y0, y1, y2:
    int y0 = canvas_positions()[0].y;
    int y1 = canvas_positions()[1].y;
    int y2 = canvas_positions()[2].y;
    int h0 = _remapper_0.height, h1 = _remapper_1.height, h2 = _remapper_2.height;
    h = std::max({y0 + h0, y1 + h1, y2 + h2}) - y_start;
    w = 0;
    // horizontal extent = blendW computed earlier:
    w = (int)(mask.cols) - x_start;
    cv::Rect roi(x_start, y_start, w, h);
    return mask(roi).clone();
  }
  // Otherwise, return the full mask.
  return mask;
}

} // namespace pano
} // namespace hm
