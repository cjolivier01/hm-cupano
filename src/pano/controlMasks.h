#pragma once

#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

namespace hm {
namespace pano {

/**
 * @brief Structure to store the spatial positions (in some coordinate system) of TIFF images.
 *
 * Typically used to store geospatial X/Y offsets in the case of GeoTIFF files.
 */
struct SpatialTiff {
  /// The X position in pixels (or another unit, depending on context).
  float xpos;
  /// The Y position in pixels (or another unit, depending on context).
  float ypos;
};

/**
 * @brief Holds reference images (like row/col images), a seam mask image, and their positions.
 *
 * The `ControlMasks` struct typically loads TIFF images that describe how to map
 * pixel coordinates (for example, as row/col transformations), plus a seam mask
 * image and position metadata for georeferencing or alignment.
 */
class ControlMasks {

public:
  ControlMasks() = default;
  ControlMasks(std::string game_dir);

  /**
   * @brief Loads the required images and position data from a directory.
   *
   * @param game_dir The directory from which to load the images/masks.
   * @return true if all images loaded successfully, false otherwise.
   */
  bool load(std::string game_dir);

  bool is_valid() const ;

  /// Column mapping for image1 (e.g., from row/col transformations).
  cv::Mat img1_col;
  /// Row mapping for image1.
  cv::Mat img1_row;
  /// Column mapping for image2.
  cv::Mat img2_col;
  /// Row mapping for image2.
  cv::Mat img2_row;

  /// A large seam mask that indicates where blending should happen.
  cv::Mat whole_seam_mask_image;
  /**
   * @brief Contains two `SpatialTiff` structures describing the positions of the two images.
   *
   * `positions[0]` corresponds to the first image, and `positions[1]` to the second.
   */
  std::vector<SpatialTiff> positions;
};

} // namespace pano
} // namespace hm

