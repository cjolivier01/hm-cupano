#pragma once

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

#include "cupano/pano/controlMasks.h"

namespace hm {
namespace pano {

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
  std::vector<SpatialTiff> positions;
};

} // namespace pano
} // namespace hm
