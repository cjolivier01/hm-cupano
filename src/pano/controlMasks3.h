#pragma once

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

namespace hm {
namespace pano {

/**
 * @brief Holds three spatial positions (in pixels or units) for the three TIFF images,
 *        plus their per-pixel remap maps (X/Y for each of the three), and a single
 *        **3-channel** seam mask (`whole_seam_mask_image`) where each pixel (R,G,B) 
 *        holds the blending weights (m₀, m₁, m₂) for image0/1/2.
 */
struct SpatialTiff3 {
  float xpos;
  float ypos;
};

class ControlMasks3 {
 public:
  ControlMasks3() = default;
  ControlMasks3(const std::string& game_dir);

  bool load(const std::string& game_dir);
  bool is_valid() const;

  size_t canvas_width() const;
  size_t canvas_height() const;

  // Per-pixel remapping for image #0 (16U each)
  cv::Mat img0_col;
  cv::Mat img0_row;

  // Per-pixel remapping for image #1
  cv::Mat img1_col;
  cv::Mat img1_row;

  // Per-pixel remapping for image #2
  cv::Mat img2_col;
  cv::Mat img2_row;

  // A **3‐channel** seam mask where each pixel’s channels sum to 1
  // (e.g. R=weight0, G=weight1, B=weight2).
  cv::Mat whole_seam_mask_image;

  // Exactly three positions for the three images
  std::vector<SpatialTiff3> positions;
};

} // namespace pano
} // namespace hm
