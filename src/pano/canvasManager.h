#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core/types.hpp>

#include <vector>

namespace hm {
namespace pano {

/**
 * @brief Structure to hold basic canvas information for stitching or blending.
 *
 * This structure keeps track of a canvas size (width and height) and
 * the anchor positions within the canvas where images should be placed.
 */
struct CanvasInfo {
  /// The width of the overall canvas.
  int width{0};
  /// The height of the overall canvas.
  int height{0};
  /**
   * @brief A collection of top-left positions (as `cv::Point`) for placing images.
   *
   * Typically, `positions[0]` is the top-left corner where the first image
   * will be placed, and `positions[1]` is for the second image, etc.
   */
  std::vector<cv::Point> positions;
};

/**
 * @brief Structure to hold parameters of a remapper, which is responsible
 * for mapping pixel coordinates from a source image to target coordinates.
 */
struct Remapper {
  /// Width of the remapped image.
  int width{0};
  /// Height of the remapped image.
  int height{0};
  /**
   * @brief The X position in the final canvas where this image is placed.
   *
   * Set by blending logic or custom offset logic.
   */
  int xpos{0};
};

/**
 * @class CanvasManager
 * @brief Manages the stitching canvas and blending region parameters.
 *
 * The CanvasManager encapsulates logic for determining how two images
 * should be placed on a canvas, whether to minimize the blending region,
 * and how to convert or crop the mask if blending is enabled.
 */
class CanvasManager {
 public:
  /**
   * @brief Default constructor that initializes the manager with canvas info and blending parameters.
   *
   * @param canvas_info  The width, height, and anchor positions for the canvas.
   * @param minimize_blend Whether or not to minimize the blend region.
   * @param overlap_pad   Extra padding to apply in the overlapping region.
   */
  CanvasManager(CanvasInfo canvas_info, bool minimize_blend, int overlap_pad = 128);

  /**
   * @brief If `_minimize_blend` is true, updates internal parameters for minimal blending overlap.
   *
   * This function calculates the overlap between the two remappers and sets
   * various ROI (region of interest) parameters that define where blending
   * actually takes place.
   *
   * @param remapped_size_1 The size of image 1 after remapping.
   * @param remapped_size_2 The size of image 2 after remapping.
   */
  void updateMinimizeBlend(const cv::Size& remapped_size_1, const cv::Size& remapped_size_2);

  /**
   * @brief Converts a given mask (e.g., the seam mask) to match the canvas dimensions.
   *
   * If `_minimize_blend` is true, it will crop the mask to the region where
   * the images overlap plus padding. Otherwise, it returns a mask that is
   * expanded to the full canvas size.
   *
   * @param mask The input mask to be converted or cropped.
   * @return A `cv::Mat` mask that matches the canvas layout.
   */
  cv::Mat convertMaskMat(const cv::Mat& mask);

  /**
   * @brief The region of interest for blending the first image.
   */
  cv::Rect2i remapped_image_roi_blend_1;
  /**
   * @brief The region of interest for blending the second image.
   */
  cv::Rect2i remapped_image_roi_blend_2;

  /**
   * @brief Returns the overlap padding that was set in the constructor.
   */
  constexpr int overlap_padding() const {
    return _overlap_pad;
  }

  /**
   * @brief Returns the overlapping width computed by `updateMinimizeBlend()`.
   */
  constexpr int overlapping_width() const {
    return _overlapping_width;
  }

  /**
   * @brief Returns the width of the entire stitching canvas.
   */
  constexpr int canvas_width() const {
    return canvas_info_.width;
  }

  /**
   * @brief Returns the height of the entire stitching canvas.
   */
  constexpr int canvas_height() const {
    return canvas_info_.height;
  }

  constexpr const std::vector<cv::Point>& canvas_positions() const {
    return canvas_info_.positions;
  }

  /// Public members holding the two remappers (for two images).
  Remapper _remapper_1;
  Remapper _remapper_2;

  /**
   * @brief Additional members used in the blending logic.
   *
   * `_x1,_y1` are the top-left of image 1, `_x2,_y2` are for image 2.
   * `_padded_blended_tlbr` is a vector of [x1, y1, x2, y2] bounding the padded region.
   */
  int _x1{0}, _y1{0}, _x2{0}, _y2{0};
  // std::vector<int> _padded_blended_tlbr;

 private:
  /// Internal storage of the canvas layout info (width, height, positions).
  CanvasInfo canvas_info_;
  /// Internal tracking of the overlap width (updated by `updateMinimizeBlend`).
  int _overlapping_width{0};
  /// Whether or not to minimize the blending region.
  bool _minimize_blend{false};
  /// Overlap padding (defaults to 128).
  int _overlap_pad{0};
};

} // namespace pano
} // namespace hm
