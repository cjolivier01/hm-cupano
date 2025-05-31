#pragma once

#include <opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

namespace hm {
namespace pano {

/**
 * @brief Holds basic canvas info for THREE‐image stitching.
 *        "positions" must have exactly 3 cv::Points: top-left corners of image0,1,2.
 */
struct CanvasInfo3 {
  int width{0};
  int height{0};
  std::vector<cv::Point> positions; // size() == 3
};

/**
 * @brief Remapper struct for each of the three images.
 */
struct Remapper3 {
  int width{0};
  int height{0};
  int xpos{0}; // where to place on canvas
};

/**
 * @class CanvasManager3
 * @brief Manages placement + blending ROIs for three‐image stitching.
 *
 * If `_minimize_blend` is true, it will compute minimal overlapping ROIs among
 * (image0 vs. image1) and (image1 vs. image2) and define:
 *   remapped_image_roi_blend_0  (where image0 overlaps image1),
 *   remapped_image_roi_blend_1  (where image1 overlaps both 0 and 2),
 *   remapped_image_roi_blend_2  (where image2 overlaps image1).
 *
 * The final blended region (where all three blend) will be the intersection of
 * those two ROI strips (i.e. where image0|1|2 all overlap).  We pass that
 * 3‐channel mask (cropped) to the 3‐image blend.
 */
class CanvasManager3 {
 public:
  CanvasManager3(CanvasInfo3 canvas_info, bool minimize_blend, int overlap_pad = 128);

  /**
   * @brief After remapping each image, call this to compute minimal blend‐ROIs:
   * @param remapped_size_0 size of the remapped image0
   * @param remapped_size_1 size of the remapped image1
   * @param remapped_size_2 size of the remapped image2
   */
  void updateMinimizeBlend(
      const cv::Size& remapped_size_0,
      const cv::Size& remapped_size_1,
      const cv::Size& remapped_size_2);

  /**
   * @brief Given a 3‐channel mask of full canvas size, either crop it
   *        to the minimal “3‐way overlapping rectangle,” or return the full mask.
   */
  cv::Mat convertMaskMat(const cv::Mat& mask);

  // ROI for blending portion of remapped image0 (wrt its local coords)
  cv::Rect2i remapped_image_roi_blend_0;

  // ROI for blending portion of remapped image1
  cv::Rect2i remapped_image_roi_blend_1;

  // ROI for blending portion of remapped image2
  cv::Rect2i remapped_image_roi_blend_2;

  constexpr int overlap_padding() const {
    return _overlap_pad;
  }
  constexpr int overlapping_width01() const {
    return _overlapping_width01; // overlap of 0 vs. 1
  }
  constexpr int overlapping_width12() const {
    return _overlapping_width12; // overlap of 1 vs. 2
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

  Remapper3 _remapper_0;
  Remapper3 _remapper_1;
  Remapper3 _remapper_2;

  // When blending, we need to know where to place the 3‐way blended pixels on canvas:
  int _x_blend_start{0}, _y_blend_start{0};

 private:
  CanvasInfo3 canvas_info_;
  bool _minimize_blend{false};
  int _overlap_pad{0};

  // The widths where image0|1 overlap, and where image1|2 overlap:
  int _overlapping_width01{0};
  int _overlapping_width12{0};
};
} // namespace pano
} // namespace hm
