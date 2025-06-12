#include "controlMasks.h"

#include <png.h>
#include <tiffio.h> // For reading TIFF metadata
#include <tiffio.h> // For TIFF metadata

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
    // std::cout << "X Position: " << xpos << std::endl;
    info.xPosition = xpos;
  } else {
    std::cout << "No X Position information found." << std::endl;
  }

  if (TIFFGetField(tif, TIFFTAG_YPOSITION, &ypos)) {
    // std::cout << "Y Position: " << ypos << std::endl;
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
  // The mapping files are saved by Hugin's 'nona' app,
  // while the seam file is saved by either enblend or multiblend
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
  auto optional_whole_seam_mask_image = load_seam_mask(whole_seam_mask);
  if (!optional_whole_seam_mask_image || optional_whole_seam_mask_image->empty()) {
    std::cerr << "Unable to load seam or masking file: " << whole_seam_mask << std::endl;
    return false;
  }
  whole_seam_mask_image = std::move(*optional_whole_seam_mask_image);

  // Determine the geospatial positions of both images, then normalize them to start at (0,0).
  positions = normalize_positions({get_geo_tiff(mapping_0_pos), get_geo_tiff(mapping_1_pos)});

  return true;
}

} // namespace pano
} // namespace hm
