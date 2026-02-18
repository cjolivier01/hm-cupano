#include "controlMasks3.h"
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <png.h>
#include <tiffio.h> // For TIFF metadata
#include <stdexcept>
#include <string>
#include <vector>

namespace hm {
namespace pano {

/**
 * A small helper to load and normalize TIFF tag positions into “SpatialTiff3”:
 */
namespace {

struct TiffInfo3 {
  bool validResolution = false;
  float xResolution = 0.0f;
  float yResolution = 0.0f;
  uint16_t resolutionUnit = 0;

  bool hasGeoTiePoints = false;
  float xPosition = 0.0f;
  float yPosition = 0.0f;
};

static SpatialTiff get_geo_tiff3(const std::string& filename) {
  TiffInfo3 info;
  TIFF* tif = TIFFOpen(filename.c_str(), "r");
  if (!tif) {
    std::cerr << "Error: Could not open file " << filename << std::endl;
    return {0, 0};
  }

  float xres = 0.0f, yres = 0.0f;
  if (TIFFGetField(tif, TIFFTAG_XRESOLUTION, &xres) && TIFFGetField(tif, TIFFTAG_YRESOLUTION, &yres)) {
    info.xResolution = xres;
    info.yResolution = yres;
    info.validResolution = true;
  }
  uint16_t resUnit = 0;
  if (TIFFGetField(tif, TIFFTAG_RESOLUTIONUNIT, &resUnit)) {
    info.resolutionUnit = resUnit;
  }

  float xpos = 0.0f, ypos = 0.0f;
  if (TIFFGetField(tif, TIFFTAG_XPOSITION, &xpos)) {
    info.xPosition = xpos;
  }
  if (TIFFGetField(tif, TIFFTAG_YPOSITION, &ypos)) {
    info.yPosition = ypos;
  }

  TIFFClose(tif);
  return SpatialTiff{.xpos = info.xPosition * info.xResolution, .ypos = info.yPosition * info.yResolution};
}

/**
 * Normalize a list of three positions so that the minimum X and Y become zero.
 */
static std::vector<SpatialTiff> normalize_positions3(std::vector<SpatialTiff>&& positions) {
  float min_x = std::numeric_limits<float>::max();
  float min_y = std::numeric_limits<float>::max();
  for (auto& sp : positions) {
    min_x = std::min(min_x, sp.xpos);
    min_y = std::min(min_y, sp.ypos);
  }
  for (auto& sp : positions) {
    sp.xpos -= min_x;
    sp.ypos -= min_y;
  }
  return positions;
}

std::vector<int> get_unique_values(const cv::Mat& gray) {
  CV_Assert(gray.type() == CV_8U); // must be single‐channel 8‐bit

  std::set<int> uniq;
  for (int y = 0; y < gray.rows; y++) {
    const uchar* rowPtr = gray.ptr<uchar>(y);
    for (int x = 0; x < gray.cols; x++) {
      uniq.insert(rowPtr[x]);
    }
  }

  // copy the set into a sorted vector
  return std::vector<int>(uniq.begin(), uniq.end());
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
cv::Mat imreadPalettedAsIndex(const std::string& filename) {
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
    throw std::runtime_error("PNG is not 8-bit paletted (indexed).");
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

cv::Mat load_seam_mask3(const std::string& filename) {
  // cv::Mat seam_mask = cv::imread(filename, cv::IMREAD_ANYDEPTH);
  cv::Mat seam_mask = imreadPalettedAsIndex(filename);
  // cv::Mat seam_mask_dest = seam_mask.clone();
  if (!seam_mask.empty()) {
    std::vector<int> unique_values = get_unique_values(seam_mask);
    assert(*unique_values.begin() == 0);
    assert(*unique_values.rbegin() == unique_values.size() - 1);
    assert(unique_values.size() == 3);
  }
  return seam_mask;
}

} // namespace

ControlMasks3::ControlMasks3(const std::string& game_dir) {
  load(game_dir);
}

cv::Mat ControlMasks3::split_to_channels(const cv::Mat& seam_mask) {
  CV_Assert(seam_mask.type() == CV_8U);
  cv::Mat seam_mask_dest = cv::Mat(cv::Size(seam_mask.cols, seam_mask.rows), CV_8UC3, cv::Scalar(0, 0, 0));

  std::vector<cv::Mat> channels;
  cv::split(seam_mask_dest, channels);

  // `seam_mask` stores image indices. Some masks may not contain all labels, so we
  // build a fixed 3-channel one-hot encoding for indices {0,1,2} without requiring
  // all labels to appear.
  channels[0].setTo(1, seam_mask == 0);
  channels[1].setTo(1, seam_mask == 1);
  channels[2].setTo(1, seam_mask == 2);

  // Merge channels back
  cv::merge(channels, seam_mask_dest);
  return seam_mask_dest;
}

bool ControlMasks3::load(const std::string& game_dir_in) {
  std::string game_dir = game_dir_in;
  if (!game_dir.empty() && game_dir.back() != '/') {
    game_dir += '/';
  }
  // TIFF filenames for the three images:
  std::string mapping0_pos = game_dir + "mapping_0000.tif";
  std::string mapping0_x = game_dir + "mapping_0000_x.tif";
  std::string mapping0_y = game_dir + "mapping_0000_y.tif";
  std::string mapping1_pos = game_dir + "mapping_0001.tif";
  std::string mapping1_x = game_dir + "mapping_0001_x.tif";
  std::string mapping1_y = game_dir + "mapping_0001_y.tif";
  std::string mapping2_pos = game_dir + "mapping_0002.tif";
  std::string mapping2_x = game_dir + "mapping_0002_x.tif";
  std::string mapping2_y = game_dir + "mapping_0002_y.tif";
  std::string seam_filename = game_dir + "seam_file.png"; // we assume 3‐channel PNG

  // Load remap (X/Y) for image0:
  img0_col = cv::imread(mapping0_x, cv::IMREAD_ANYDEPTH);
  if (img0_col.empty()) {
    std::cerr << "Unable to load remap0_x: " << mapping0_x << std::endl;
    return false;
  }
  img0_row = cv::imread(mapping0_y, cv::IMREAD_ANYDEPTH);
  if (img0_row.empty()) {
    std::cerr << "Unable to load remap0_y: " << mapping0_y << std::endl;
    return false;
  }

  // Load remap for image1:
  img1_col = cv::imread(mapping1_x, cv::IMREAD_ANYDEPTH);
  if (img1_col.empty()) {
    std::cerr << "Unable to load remap1_x: " << mapping1_x << std::endl;
    return false;
  }
  img1_row = cv::imread(mapping1_y, cv::IMREAD_ANYDEPTH);
  if (img1_row.empty()) {
    std::cerr << "Unable to load remap1_y: " << mapping1_y << std::endl;
    return false;
  }

  // Load remap for image2:
  img2_col = cv::imread(mapping2_x, cv::IMREAD_ANYDEPTH);
  if (img2_col.empty()) {
    std::cerr << "Unable to load remap2_x: " << mapping2_x << std::endl;
    return false;
  }
  img2_row = cv::imread(mapping2_y, cv::IMREAD_ANYDEPTH);
  if (img2_row.empty()) {
    std::cerr << "Unable to load remap2_y: " << mapping2_y << std::endl;
    return false;
  }

  // Load the seam mask:
  whole_seam_mask_image = load_seam_mask3(seam_filename);

  if (whole_seam_mask_image.empty()) {
    std::cerr << "Unable to load seam mask: " << seam_filename << std::endl;
    return false;
  }

  // Load geospatial positions from TIFF tags:
  SpatialTiff p0 = get_geo_tiff3(mapping0_pos);
  SpatialTiff p1 = get_geo_tiff3(mapping1_pos);
  SpatialTiff p2 = get_geo_tiff3(mapping2_pos);
  positions = normalize_positions3({p0, p1, p2});

  return true;
}

bool ControlMasks3::is_valid() const {
  return (
      !img0_col.empty() && !img0_row.empty() && !img1_col.empty() && !img1_row.empty() && !img2_col.empty() &&
      !img2_row.empty() && !whole_seam_mask_image.empty() && positions.size() == 3);
}

size_t ControlMasks3::canvas_width() const {
  // Use the maximum X‐extent of the three images
  float w0 = positions[0].xpos + img0_col.cols;
  float w1 = positions[1].xpos + img1_col.cols;
  float w2 = positions[2].xpos + img2_col.cols;
  return static_cast<size_t>(std::max({w0, w1, w2}));
}

size_t ControlMasks3::canvas_height() const {
  // Use maximum Y‐extent
  float h0 = positions[0].ypos + img0_col.rows;
  float h1 = positions[1].ypos + img1_col.rows;
  float h2 = positions[2].ypos + img2_col.rows;
  return static_cast<size_t>(std::max({h0, h1, h2}));
}

} // namespace pano
} // namespace hm
