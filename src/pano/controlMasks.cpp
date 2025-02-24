#include "controlMasks.h"

#include <tiffio.h> // For reading TIFF metadata

namespace hm {
namespace pano {

/**
 * @brief Internal structure used to store metadata from TIFF tags.
 */
namespace {

struct TiffInfo {
  bool validResolution = false; ///< Whether resolution tags were valid
  float xResolution = 0.0f; ///< Horizontal resolution
  float yResolution = 0.0f; ///< Vertical resolution

  uint16_t resolutionUnit = 0; ///< Resolution unit (inch, centimeter, etc.)

  bool hasGeoTiePoints = false; ///< Whether geo tie points were found
  float xPosition{0}; ///< The X position from TIFF metadata
  float yPosition{0}; ///< The Y position from TIFF metadata
};

/**
 * @brief Normalizes a list of positions such that the minimum X/Y becomes 0.
 *
 * All positions are translated so that the smallest X and Y values shift to 0.
 *
 * @param positions A vector of `SpatialTiff` positions to normalize.
 * @return The translated positions (moved so that min_x and min_y are 0).
 */
std::vector<SpatialTiff> normalize_positions(std::vector<SpatialTiff>&& positions) {
  float min_x = std::numeric_limits<float>::max();
  float min_y = std::numeric_limits<float>::max();

  // Find minimal X and Y values.
  std::for_each(positions.begin(), positions.end(), [&](const SpatialTiff& sp) {
    min_x = std::min(min_x, sp.xpos);
    min_y = std::min(min_y, sp.ypos);
  });

  // Subtract out the minimum to shift everything.
  std::for_each(positions.begin(), positions.end(), [&](SpatialTiff& sp) {
    sp.xpos -= min_x;
    sp.ypos -= min_y;
  });

  return positions;
}

/**
 * @brief Reads TIFF metadata such as resolution and X/Y positions from a file.
 *
 * @param filename The TIFF file to read.
 * @return A `TiffInfo` struct containing various metadata fields.
 */
TiffInfo getTiffInfo(const std::string& filename) {
  TiffInfo info;
  TIFF* tif = TIFFOpen(filename.c_str(), "r");
  if (!tif) {
    std::cerr << "Error: Could not open file " << filename << std::endl;
    return info;
  }

  // Get Resolution
  float xres = 0.0f, yres = 0.0f;
  if (TIFFGetField(tif, TIFFTAG_XRESOLUTION, &xres) && TIFFGetField(tif, TIFFTAG_YRESOLUTION, &yres)) {
    info.xResolution = xres;
    info.yResolution = yres;
    info.validResolution = true;
  }

  // Resolution Unit
  uint16_t resUnit = 0;
  if (TIFFGetField(tif, TIFFTAG_RESOLUTIONUNIT, &resUnit)) {
    info.resolutionUnit = resUnit;
  }

  // X/Y Position
  float xpos = 0.0f, ypos = 0.0f;
  if (TIFFGetField(tif, TIFFTAG_XPOSITION, &xpos)) {
    std::cout << "X Position: " << xpos << std::endl;
    info.xPosition = xpos;
  } else {
    std::cout << "No X Position information found." << std::endl;
  }

  if (TIFFGetField(tif, TIFFTAG_YPOSITION, &ypos)) {
    std::cout << "Y Position: " << ypos << std::endl;
    info.yPosition = ypos;
  } else {
    std::cout << "No Y Position information found." << std::endl;
  }

  TIFFClose(tif);
  return info;
}

/**
 * @brief Creates a `SpatialTiff` from TIFF metadata, multiplying position by resolution.
 *
 * @param filename The TIFF file path.
 * @return A `SpatialTiff` representing X/Y positions in pixel-space (or some unit).
 */
SpatialTiff get_geo_tiff(const std::string& filename) {
  TiffInfo info = getTiffInfo(filename);
  return SpatialTiff{.xpos = info.xPosition * info.xResolution, .ypos = info.yPosition * info.yResolution};
}

/**
 * @brief Loads a seam mask from disk in grayscale, then processes min/max values for binary usage.
 *
 * This function reads the image as `IMREAD_GRAYSCALE`, finds min and max pixel
 * values, then sets max-locations to 0 and min-locations to 1, effectively
 * producing an inverted binary mask.
 *
 * @param filename The path to the seam mask image.
 * @return A processed 8-bit single-channel seam mask.
 */
cv::Mat load_seam_mask(const std::string& filename) {
  cv::Mat seam_mask = cv::imread(filename, cv::IMREAD_GRAYSCALE);
  if (!seam_mask.empty()) {
    double minVal, maxVal;
    cv::Point minLoc, maxLoc;
    cv::minMaxLoc(seam_mask, &minVal, &maxVal, &minLoc, &maxLoc);

    // Create masks for min and max values
    cv::Mat minMask = (seam_mask == static_cast<int>(minVal));
    cv::Mat maxMask = (seam_mask == static_cast<int>(maxVal));

    // Invert: set max locations to 0 and min locations to 1
    seam_mask.setTo(0, maxMask);
    seam_mask.setTo(1, minMask);
  }
  return seam_mask;
}

} // namespace

ControlMasks::ControlMasks(std::string game_dir) {
  // Caller should check is_valid()
  (void)load(std::move(game_dir));
}

bool ControlMasks::is_valid() const {
  return !img1_col.empty() && !img1_row.empty() && !img2_col.empty() && !img2_row.empty() &&
      !whole_seam_mask_image.empty() && positions.size() == 2;
}

size_t ControlMasks::canvas_width() const {
  return std::max(positions.at(0).xpos + img1_col.cols, positions.at(1).xpos + img2_col.cols);
}

size_t ControlMasks::canvas_height() const {
  return std::max(positions.at(0).ypos + img1_col.rows, positions.at(1).ypos + img2_col.rows);
}

bool ControlMasks::load(std::string game_dir) {
  if (!game_dir.empty() && game_dir.back() != '/') {
    game_dir += '/';
  }

  // Construct the file paths we want to load.
  std::string mapping_0_pos = game_dir + "mapping_0000.tif";
  std::string mapping_0_x = game_dir + "mapping_0000_x.tif";
  std::string mapping_0_y = game_dir + "mapping_0000_y.tif";
  std::string mapping_1_pos = game_dir + "mapping_0001.tif";
  std::string mapping_1_x = game_dir + "mapping_0001_x.tif";
  std::string mapping_1_y = game_dir + "mapping_0001_y.tif";
  std::string whole_seam_mask = game_dir + "seam_file.png";

  // Load column/row transformations for the first image.
  img1_col = cv::imread(mapping_0_x, cv::IMREAD_ANYDEPTH);
  if (img1_col.empty()) {
    std::cerr << "Unable to load seam or masking file: " << mapping_0_x << std::endl;
    return false;
  }
  assert(img1_col.type() == CV_16U);
  img1_row = cv::imread(mapping_0_y, cv::IMREAD_ANYDEPTH);
  if (img1_row.empty()) {
    std::cerr << "Unable to load seam or masking file: " << mapping_0_y << std::endl;
    return false;
  }

  // Load column/row transformations for the second image.
  img2_col = cv::imread(mapping_1_x, cv::IMREAD_ANYDEPTH);
  if (img2_col.empty()) {
    std::cerr << "Unable to load seam or masking file: " << mapping_1_x << std::endl;
    return false;
  }
  img2_row = cv::imread(mapping_1_y, cv::IMREAD_ANYDEPTH);
  if (img2_row.empty()) {
    std::cerr << "Unable to load seam or masking file: " << mapping_1_y << std::endl;
    return false;
  }

  // Load and process the seam mask.
  whole_seam_mask_image = load_seam_mask(whole_seam_mask);
  if (whole_seam_mask_image.empty()) {
    std::cerr << "Unable to load seam or masking file: " << whole_seam_mask << std::endl;
    return false;
  }

  // Determine the geospatial positions of both images, then normalize them to start at (0,0).
  positions = normalize_positions({get_geo_tiff(mapping_0_pos), get_geo_tiff(mapping_1_pos)});

  return true;
}

} // namespace pano
} // namespace hm
