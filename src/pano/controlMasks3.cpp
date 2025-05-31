#include "controlMasks3.h"
#include <tiffio.h> // For TIFF metadata

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

static SpatialTiff3 get_geo_tiff3(const std::string& filename) {
  TiffInfo3 info;
  TIFF* tif = TIFFOpen(filename.c_str(), "r");
  if (!tif) {
    std::cerr << "Error: Could not open file " << filename << std::endl;
    return {0, 0};
  }

  float xres = 0.0f, yres = 0.0f;
  if (TIFFGetField(tif, TIFFTAG_XRESOLUTION, &xres) &&
      TIFFGetField(tif, TIFFTAG_YRESOLUTION, &yres)) {
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
  return SpatialTiff3{
      .xpos = info.xPosition * info.xResolution,
      .ypos = info.yPosition * info.yResolution};
}

/**
 * Normalize a list of three positions so that the minimum X and Y become zero.
 */
static std::vector<SpatialTiff3> normalize_positions3(std::vector<SpatialTiff3>&& positions) {
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

/**
 * @brief Loads a **3‐channel** seam mask from disk (e.g. a PNG).  
 *        We assume the user supplies it as a 3‐channel color image where
 *        each channel is already 0/255 (or 0/1) to indicate weights.  
 *        We simply load as CV_8UC3 and (if needed) invert or normalize.
 */
static cv::Mat load_seam_mask3(const std::string& filename) {
  cv::Mat mask = cv::imread(filename, cv::IMREAD_COLOR); // CV_8UC3
  if (!mask.empty()) {
    // If the mask’s channels are “0 / 255,” normalize them to “0 / 1.”
    cv::Mat float_mask;
    mask.convertTo(float_mask, CV_32F, 1.0f / 255.0f);
    // Then convert back to 8-bit [0..255] so that 1.0→255 (for CudaMat<uchar>).
    float_mask.convertTo(mask, CV_8UC3, 255.0f);
  }
  return mask;
}

} // namespace

ControlMasks3::ControlMasks3(const std::string& game_dir) {
  load(game_dir);
}

bool ControlMasks3::load(const std::string& game_dir_in) {
  std::string game_dir = game_dir_in;
  if (!game_dir.empty() && game_dir.back() != '/') {
    game_dir += '/';
  }
  // TIFF filenames for the three images:
  std::string mapping0_pos   = game_dir + "mapping_0000.tif";
  std::string mapping0_x     = game_dir + "mapping_0000_x.tif";
  std::string mapping0_y     = game_dir + "mapping_0000_y.tif";
  std::string mapping1_pos   = game_dir + "mapping_0001.tif";
  std::string mapping1_x     = game_dir + "mapping_0001_x.tif";
  std::string mapping1_y     = game_dir + "mapping_0001_y.tif";
  std::string mapping2_pos   = game_dir + "mapping_0002.tif";
  std::string mapping2_x     = game_dir + "mapping_0002_x.tif";
  std::string mapping2_y     = game_dir + "mapping_0002_y.tif";
  std::string seam_filename  = game_dir + "seam_file.png";   // we assume 3‐channel PNG

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

  // Load the **3‐channel** seam mask:
  whole_seam_mask_image = load_seam_mask3(seam_filename);
  if (whole_seam_mask_image.empty()) {
    std::cerr << "Unable to load 3‐channel seam mask: " << seam_filename << std::endl;
    return false;
  }

  // Load geospatial positions from TIFF tags:
  SpatialTiff3 p0 = get_geo_tiff3(mapping0_pos);
  SpatialTiff3 p1 = get_geo_tiff3(mapping1_pos);
  SpatialTiff3 p2 = get_geo_tiff3(mapping2_pos);
  positions = normalize_positions3({p0, p1, p2});

  return true;
}

bool ControlMasks3::is_valid() const {
  return (!img0_col.empty() && !img0_row.empty() &&
          !img1_col.empty() && !img1_row.empty() &&
          !img2_col.empty() && !img2_row.empty() &&
          !whole_seam_mask_image.empty() &&
          positions.size() == 3);
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
