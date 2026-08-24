#include "controlMasks.h"

#include <opencv2/imgproc.hpp>
#include <png.h>
#include <tiffio.h> // For reading TIFF metadata
#include <tiffio.h> // For TIFF metadata

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace hm {
namespace pano {

/**
 * @brief Internal structure used to store metadata from TIFF tags.
 */
namespace {

constexpr uint16_t kUnmappedPositionValue = 65535;

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

cv::Mat resize_remap_preserving_unmapped(const cv::Mat& src, const cv::Size& size) {
  cv::Mat resized;
  cv::resize(src, resized, size, 0.0, 0.0, cv::INTER_NEAREST);
  cv::Mat invalid_mask(size, CV_8U, cv::Scalar(0));
  const double scale_x = static_cast<double>(src.cols) / static_cast<double>(size.width);
  const double scale_y = static_cast<double>(src.rows) / static_cast<double>(size.height);
  std::vector<int> column_prefix(static_cast<size_t>(src.cols) + 1, 0);
  for (int y = 0; y < size.height; ++y) {
    const int y0 = std::clamp(static_cast<int>(std::floor(y * scale_y)), 0, src.rows - 1);
    const int y1 = std::clamp(static_cast<int>(std::ceil((y + 1) * scale_y)), y0 + 1, src.rows);
    column_prefix[0] = 0;
    for (int source_x = 0; source_x < src.cols; ++source_x) {
      bool has_unmapped = false;
      for (int source_y = y0; source_y < y1; ++source_y) {
        if (src.ptr<uint16_t>(source_y)[source_x] == kUnmappedPositionValue) {
          has_unmapped = true;
          break;
        }
      }
      column_prefix[static_cast<size_t>(source_x) + 1] =
          column_prefix[static_cast<size_t>(source_x)] + (has_unmapped ? 1 : 0);
    }
    for (int x = 0; x < size.width; ++x) {
      const int x0 = std::clamp(static_cast<int>(std::floor(x * scale_x)), 0, src.cols - 1);
      const int x1 = std::clamp(static_cast<int>(std::ceil((x + 1) * scale_x)), x0 + 1, src.cols);
      if (column_prefix[static_cast<size_t>(x1)] - column_prefix[static_cast<size_t>(x0)] > 0) {
        invalid_mask.at<uint8_t>(y, x) = 255;
      }
    }
  }
  resized.setTo(kUnmappedPositionValue, invalid_mask);
  return resized;
}

cv::Mat resize_mask_nearest(const cv::Mat& src, const cv::Size& size) {
  cv::Mat resized;
  cv::resize(src, resized, size, 0.0, 0.0, cv::INTER_NEAREST);
  return resized;
}

struct ScaledPlacement {
  SpatialTiff position;
  cv::Size size;
};

std::optional<cv::Size> read_tiff_size(const std::string& filename, bool require_uint16 = true) {
  TIFF* tif = TIFFOpen(filename.c_str(), "r");
  if (!tif) {
    return std::nullopt;
  }
  uint32_t width = 0;
  uint32_t height = 0;
  uint16_t samples = 0;
  uint16_t bits = 0;
  uint16_t sample_format = SAMPLEFORMAT_UINT;
  const bool ok = TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width) && TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &height);
  TIFFGetFieldDefaulted(tif, TIFFTAG_SAMPLESPERPIXEL, &samples);
  TIFFGetFieldDefaulted(tif, TIFFTAG_BITSPERSAMPLE, &bits);
  TIFFGetFieldDefaulted(tif, TIFFTAG_SAMPLEFORMAT, &sample_format);
  TIFFClose(tif);
  if (!ok || width == 0 || height == 0 || width > static_cast<uint32_t>(std::numeric_limits<int>::max()) ||
      height > static_cast<uint32_t>(std::numeric_limits<int>::max()) ||
      (require_uint16 && (samples != 1 || bits != 16 || sample_format != SAMPLEFORMAT_UINT))) {
    return std::nullopt;
  }
  return cv::Size(static_cast<int>(width), static_cast<int>(height));
}

ScaledPlacement scaled_placement(const SpatialTiff& position, const cv::Size& size, double scale) {
  const auto scaled_x = static_cast<int>(std::floor(position.xpos * scale));
  const auto scaled_y = static_cast<int>(std::floor(position.ypos * scale));
  const auto scaled_right = static_cast<int>(std::ceil((position.xpos + size.width) * scale));
  const auto scaled_bottom = static_cast<int>(std::ceil((position.ypos + size.height) * scale));
  return ScaledPlacement{
      .position = SpatialTiff{.xpos = static_cast<float>(scaled_x), .ypos = static_cast<float>(scaled_y)},
      .size = cv::Size(std::max(1, scaled_right - scaled_x), std::max(1, scaled_bottom - scaled_y))};
}

cv::Size canvas_size(const std::vector<ScaledPlacement>& placements) {
  int width = 1;
  int height = 1;
  for (const ScaledPlacement& placement : placements) {
    width = std::max(width, static_cast<int>(placement.position.xpos) + placement.size.width);
    height = std::max(height, static_cast<int>(placement.position.ypos) + placement.size.height);
  }
  return cv::Size(width, height);
}

double scale_to_fit_max_width(
    const std::vector<SpatialTiff>& positions,
    const std::vector<cv::Size>& sizes,
    size_t native_width,
    int max_output_width) {
  double low = 0.0;
  double high = static_cast<double>(max_output_width) / static_cast<double>(native_width);
  std::vector<ScaledPlacement> direct_placements;
  direct_placements.reserve(positions.size());
  for (size_t i = 0; i < positions.size(); ++i) {
    direct_placements.push_back(scaled_placement(positions[i], sizes[i], high));
  }
  if (canvas_size(direct_placements).width <= max_output_width) {
    return high;
  }
  for (int iteration = 0; iteration < 32; ++iteration) {
    const double mid = (low + high) / 2.0;
    std::vector<ScaledPlacement> placements;
    placements.reserve(positions.size());
    for (size_t i = 0; i < positions.size(); ++i) {
      placements.push_back(scaled_placement(positions[i], sizes[i], mid));
    }
    if (canvas_size(placements).width <= max_output_width) {
      low = mid;
    } else {
      high = mid;
    }
  }
  return low > 0.0 ? low : high;
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
    // std::cout << "X Position: " << xpos << std::endl;
    info.xPosition = xpos;
    info.hasGeoTiePoints = true;
  } else {
    std::cout << "No X Position information found." << std::endl;
  }

  if (TIFFGetField(tif, TIFFTAG_YPOSITION, &ypos)) {
    // std::cout << "Y Position: " << ypos << std::endl;
    info.yPosition = ypos;
  } else {
    std::cout << "No Y Position information found." << std::endl;
    info.hasGeoTiePoints = false;
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
std::optional<SpatialTiff> get_geo_tiff(const std::string& filename) {
  TiffInfo info = getTiffInfo(filename);
  if (!info.validResolution || !info.hasGeoTiePoints || !std::isfinite(info.xResolution) ||
      !std::isfinite(info.yResolution) || info.xResolution <= 0.0f || info.yResolution <= 0.0f ||
      !std::isfinite(info.xPosition) || !std::isfinite(info.yPosition)) {
    return std::nullopt;
  }
  const float xpos = info.xPosition * info.xResolution;
  const float ypos = info.yPosition * info.yResolution;
  if (!std::isfinite(xpos) || !std::isfinite(ypos)) {
    return std::nullopt;
  }
  return SpatialTiff{.xpos = xpos, .ypos = ypos};
}

/**
 * Load a paletted (indexed) PNG as a single‐channel 8-bit Mat of palette‐indices.
 * Throws std::runtime_error on any error (non-paletted, file not found, etc.).
 *
 * Requirements:
 *  • libpng installed, and you link with -lpng
 *  • The PNG *must* be 8-bit paletted (PNG_COLOR_TYPE_PALETTE, bit_depth=8).
 *    If it is not paletted, this will error out.
 *
 * Usage:
 *    cv::Mat idx = imreadPalettedAsIndex("my_palette.png");
 *    // idx.type()==CV_8U, idx.cols×idx.rows = image size
 */
std::optional<cv::Mat> imreadPalettedAsIndex(const std::string& filename) {
  // 1) Open the file in binary mode
  FILE* fp = fopen(filename.c_str(), "rb");
  if (!fp) {
    throw std::runtime_error("Cannot open file: " + filename);
  }

  // 2) Read and check the 8-byte PNG signature
  png_byte sig[8];
  if (fread(sig, 1, 8, fp) != 8) {
    fclose(fp);
    throw std::runtime_error("Failed to read PNG signature.");
  }
  if (png_sig_cmp(sig, 0, 8)) {
    fclose(fp);
    throw std::runtime_error("File is not recognized as a PNG.");
  }

  // 3) Create libpng read structs
  png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
  if (!png_ptr) {
    fclose(fp);
    throw std::runtime_error("png_create_read_struct failed.");
  }

  png_infop info_ptr = png_create_info_struct(png_ptr);
  if (!info_ptr) {
    png_destroy_read_struct(&png_ptr, (png_infopp) nullptr, (png_infopp) nullptr);
    fclose(fp);
    throw std::runtime_error("png_create_info_struct failed.");
  }

  if (setjmp(png_jmpbuf(png_ptr))) {
    // If we get here, libpng encountered an error
    png_destroy_read_struct(&png_ptr, &info_ptr, (png_infopp) nullptr);
    fclose(fp);
    throw std::runtime_error("Error during PNG init_io or read_info.");
  }

  png_init_io(png_ptr, fp);
  png_set_sig_bytes(png_ptr, 8); // we already read 8 signature bytes

  // 4) Read PNG metadata (header)
  png_read_info(png_ptr, info_ptr);

  // 5) Check that this PNG is 8-bit paletted (indexed) format
  png_uint_32 width = png_get_image_width(png_ptr, info_ptr);
  png_uint_32 height = png_get_image_height(png_ptr, info_ptr);
  png_byte color_type = png_get_color_type(png_ptr, info_ptr);
  png_byte bit_depth = png_get_bit_depth(png_ptr, info_ptr);

  if (color_type != PNG_COLOR_TYPE_PALETTE || bit_depth != 8) {
    png_destroy_read_struct(&png_ptr, &info_ptr, (png_infopp) nullptr);
    fclose(fp);
    return std::nullopt;
  }

  // 6) Ensure no transforms (we want raw indices). Do NOT call png_set_palette_to_rgb()
  //    and do NOT call any transform that expands the palette. We just need the raw bytes.
  //    So we skip any png_set_* that would expand the data.

  // 7) Allocate row pointers and image buffer
  std::vector<png_bytep> row_ptrs(height);
  // Each row is `width` bytes, each byte is an index [0..255] into the palette
  std::vector<png_byte> raw_data(width * height);
  for (png_uint_32 y = 0; y < height; y++) {
    row_ptrs[y] = raw_data.data() + (y * width);
  }

  // 8) Read the image into our row pointers
  png_read_image(png_ptr, row_ptrs.data());

  // 9) Clean up libpng structs
  png_destroy_read_struct(&png_ptr, &info_ptr, (png_infopp) nullptr);
  fclose(fp);

  // 10) Wrap raw_data into a single‐channel Mat (CV_8U)
  cv::Mat indexed((int)height, (int)width, CV_8U, raw_data.data());
  // We must clone because raw_data is a local vector—once we leave this scope, raw_data goes away.
  return indexed.clone();
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
std::optional<cv::Mat> load_seam_mask(const std::string& filename) {
  if (!std::filesystem::exists(filename)) {
    std::cerr << "Could not find seam file: " << filename << std::endl;
    return std::nullopt;
  }
  std::optional<cv::Mat> opt_seam_mask = imreadPalettedAsIndex(filename);
  cv::Mat seam_mask;
  if (!opt_seam_mask) {
    seam_mask = cv::imread(filename, cv::IMREAD_GRAYSCALE);
  } else {
    seam_mask = std::move(*opt_seam_mask);
  }
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

bool binary_seam_has_both_classes(const cv::Mat& seam_mask) {
  if (seam_mask.empty()) {
    return false;
  }
  double min_val = 0.0;
  double max_val = 0.0;
  cv::minMaxLoc(seam_mask, &min_val, &max_val);
  return min_val == 0.0 && max_val == 1.0;
}

void clear_control_masks(ControlMasks& masks) {
  masks.img1_col.release();
  masks.img1_row.release();
  masks.img2_col.release();
  masks.img2_row.release();
  masks.whole_seam_mask_image.release();
  masks.positions.clear();
}

} // namespace

ControlMasks::ControlMasks(std::string game_dir, int max_output_width) {
  // Caller should check is_valid()
  (void)load(std::move(game_dir), max_output_width);
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

bool ControlMasks::scale_to_max_output_width(int max_output_width) {
  if (!is_valid() || max_output_width <= 0 || canvas_width() <= static_cast<size_t>(max_output_width)) {
    return is_valid();
  }

  const size_t native_width = canvas_width();
  const double scale =
      scale_to_fit_max_width(positions, {img1_col.size(), img2_col.size()}, native_width, max_output_width);
  const std::vector<ScaledPlacement> placements{
      scaled_placement(positions[0], img1_col.size(), scale),
      scaled_placement(positions[1], img2_col.size(), scale),
  };
  img1_col = resize_remap_preserving_unmapped(img1_col, placements[0].size);
  img1_row = resize_remap_preserving_unmapped(img1_row, img1_col.size());
  img2_col = resize_remap_preserving_unmapped(img2_col, placements[1].size);
  img2_row = resize_remap_preserving_unmapped(img2_row, img2_col.size());
  whole_seam_mask_image = resize_mask_nearest(whole_seam_mask_image, canvas_size(placements));
  if (!binary_seam_has_both_classes(whole_seam_mask_image)) {
    clear_control_masks(*this);
    return false;
  }
  positions = {placements[0].position, placements[1].position};
  return true;
}

bool ControlMasks::load(std::string game_dir, int max_output_width) {
  clear_control_masks(*this);
  if (!game_dir.empty() && game_dir.back() != '/') {
    game_dir += '/';
  }

  // Construct the file paths we want to load.
  // The mapping files are saved by Hugin's 'nona' app,
  // while the seam file is saved by either enblend or multiblend
  std::string mapping_0_pos = game_dir + "mapping_0000.tif";
  std::string mapping_0_x = game_dir + "mapping_0000_x.tif";
  std::string mapping_0_y = game_dir + "mapping_0000_y.tif";
  std::string mapping_1_pos = game_dir + "mapping_0001.tif";
  std::string mapping_1_x = game_dir + "mapping_0001_x.tif";
  std::string mapping_1_y = game_dir + "mapping_0001_y.tif";
  std::string whole_seam_mask = game_dir + "seam_file.png";

  const auto p0 = get_geo_tiff(mapping_0_pos);
  const auto p1 = get_geo_tiff(mapping_1_pos);
  if (!p0 || !p1) {
    std::cerr << "Unable to load mapping placement metadata from " << mapping_0_pos << " / " << mapping_1_pos
              << std::endl;
    clear_control_masks(*this);
    return false;
  }
  positions = normalize_positions({*p0, *p1});
  const auto img1_position_size = read_tiff_size(mapping_0_pos, /*require_uint16=*/false);
  const auto img2_position_size = read_tiff_size(mapping_1_pos, /*require_uint16=*/false);
  const auto img1_size = read_tiff_size(mapping_0_x);
  const auto img2_size = read_tiff_size(mapping_1_x);
  const auto img1_row_size = read_tiff_size(mapping_0_y);
  const auto img2_row_size = read_tiff_size(mapping_1_y);
  if (!img1_position_size || !img2_position_size || !img1_size || !img2_size || !img1_row_size || !img2_row_size) {
    std::cerr << "Unable to load remap metadata from " << mapping_0_x << " / " << mapping_1_x << std::endl;
    clear_control_masks(*this);
    return false;
  }
  if (*img1_position_size != *img1_size || *img1_size != *img1_row_size || *img2_position_size != *img2_size ||
      *img2_size != *img2_row_size) {
    std::cerr << "Mapping placement and remap dimensions do not match" << std::endl;
    clear_control_masks(*this);
    return false;
  }
  std::vector<ScaledPlacement> placements{
      ScaledPlacement{.position = positions[0], .size = *img1_size},
      ScaledPlacement{.position = positions[1], .size = *img2_size},
  };
  const cv::Size native_canvas_size = canvas_size(placements);
  if (max_output_width > 0 && native_canvas_size.width > max_output_width) {
    std::cerr << "Control mask canvas " << native_canvas_size.width << "x" << native_canvas_size.height
              << " exceeds max_output_width " << max_output_width << "; regenerate capped mapping TIFFs" << std::endl;
    clear_control_masks(*this);
    return false;
  }

  // Load column/row transformations for the first image.
  img1_col = cv::imread(mapping_0_x, cv::IMREAD_ANYDEPTH);
  if (img1_col.empty()) {
    std::cerr << "Unable to load seam or masking file: " << mapping_0_x << std::endl;
    clear_control_masks(*this);
    return false;
  }
  assert(img1_col.type() == CV_16U);
  if (img1_col.size() != placements[0].size) {
    img1_col = resize_remap_preserving_unmapped(img1_col, placements[0].size);
  }
  img1_row = cv::imread(mapping_0_y, cv::IMREAD_ANYDEPTH);
  if (img1_row.empty()) {
    std::cerr << "Unable to load seam or masking file: " << mapping_0_y << std::endl;
    clear_control_masks(*this);
    return false;
  }
  if (img1_row.size() != placements[0].size) {
    img1_row = resize_remap_preserving_unmapped(img1_row, placements[0].size);
  }

  // Load column/row transformations for the second image.
  img2_col = cv::imread(mapping_1_x, cv::IMREAD_ANYDEPTH);
  if (img2_col.empty()) {
    std::cerr << "Unable to load seam or masking file: " << mapping_1_x << std::endl;
    clear_control_masks(*this);
    return false;
  }
  if (img2_col.size() != placements[1].size) {
    img2_col = resize_remap_preserving_unmapped(img2_col, placements[1].size);
  }
  img2_row = cv::imread(mapping_1_y, cv::IMREAD_ANYDEPTH);
  if (img2_row.empty()) {
    std::cerr << "Unable to load seam or masking file: " << mapping_1_y << std::endl;
    clear_control_masks(*this);
    return false;
  }
  if (img2_row.size() != placements[1].size) {
    img2_row = resize_remap_preserving_unmapped(img2_row, placements[1].size);
  }

  // Load and process the seam mask.
  std::optional<cv::Mat> optional_whole_seam_mask_image;
  try {
    optional_whole_seam_mask_image = load_seam_mask(whole_seam_mask);
  } catch (const std::exception& e) {
    std::cerr << "Unable to load seam or masking file: " << whole_seam_mask << " (" << e.what() << ")" << std::endl;
    clear_control_masks(*this);
    return false;
  }
  if (!optional_whole_seam_mask_image || optional_whole_seam_mask_image->empty()) {
    std::cerr << "Unable to load seam or masking file: " << whole_seam_mask << std::endl;
    clear_control_masks(*this);
    return false;
  }
  whole_seam_mask_image = std::move(*optional_whole_seam_mask_image);
  const cv::Size effective_canvas_size = canvas_size(placements);
  if (whole_seam_mask_image.size() != effective_canvas_size) {
    whole_seam_mask_image = resize_mask_nearest(whole_seam_mask_image, effective_canvas_size);
  }
  if (!binary_seam_has_both_classes(whole_seam_mask_image)) {
    std::cerr << "Seam mask lost one or more image classes: " << whole_seam_mask << std::endl;
    clear_control_masks(*this);
    return false;
  }
  positions = {placements[0].position, placements[1].position};

  return true;
}

} // namespace pano
} // namespace hm
