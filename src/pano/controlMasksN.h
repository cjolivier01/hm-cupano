#pragma once

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

#include "cupano/pano/controlMasks.h"

namespace hm {
namespace pano {

// Generic N-image control masks loader.
class ControlMasksN {
 public:
  ControlMasksN() = default;
  ControlMasksN(const std::string& dir, int n_images) { load(dir, n_images); }

  // Loads mapping_XXXX(_x|_y).tif for i in [0..N-1] and seam_file.png (indexed).
  bool load(const std::string& dir, int n_images);

  bool is_valid() const;

  size_t canvas_width() const;
  size_t canvas_height() const;

  // Per-image remap col/row (CV_16U)
  std::vector<cv::Mat> img_col; // size N
  std::vector<cv::Mat> img_row; // size N

  // Single-channel indexed seam; split into N channels at use time.
  cv::Mat whole_seam_mask_indexed; // CV_8U indexed labels [0..N-1]

  // Positions size N
  std::vector<SpatialTiff> positions;

  // Convert indexed seam to N-channel one-hot mask (CV_8UCN)
  static cv::Mat split_to_channels(const cv::Mat& indexed, int n_images);
};

} // namespace pano
} // namespace hm

