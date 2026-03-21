#include "controlMasksN.h"

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <png.h>
#include <tiffio.h>
#include <array>
#include <set>
#include <stdexcept>

namespace hm {
namespace pano {
namespace {

struct TiffInfoN {
  float xResolution = 0.0f;
  float yResolution = 0.0f;
  float xPosition = 0.0f;
  float yPosition = 0.0f;
};

static SpatialTiff get_geo_tiffN(const std::string& filename) {
  TiffInfoN info;
  TIFF* tif = TIFFOpen(filename.c_str(), "r");
  if (!tif) {
    std::cerr << "Error: Could not open file " << filename << std::endl;
    return {0, 0};
  }
  float xres = 0.0f, yres = 0.0f;
  TIFFGetField(tif, TIFFTAG_XRESOLUTION, &xres);
  TIFFGetField(tif, TIFFTAG_YRESOLUTION, &yres);
  info.xResolution = xres;
  info.yResolution = yres;
  float xpos = 0.0f, ypos = 0.0f;
  TIFFGetField(tif, TIFFTAG_XPOSITION, &xpos);
  TIFFGetField(tif, TIFFTAG_YPOSITION, &ypos);
  info.xPosition = xpos;
  info.yPosition = ypos;
  TIFFClose(tif);
  return SpatialTiff{.xpos = info.xPosition * info.xResolution, .ypos = info.yPosition * info.yResolution};
}

static std::vector<SpatialTiff> normalize_positionsN(std::vector<SpatialTiff>&& positions) {
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

static std::vector<int> get_unique_values(const cv::Mat& gray) {
  CV_Assert(gray.type() == CV_8U);
  std::set<int> uniq;
  for (int y = 0; y < gray.rows; y++) {
    const uchar* rowPtr = gray.ptr<uchar>(y);
    for (int x = 0; x < gray.cols; x++) {
      uniq.insert(rowPtr[x]);
    }
  }
  return std::vector<int>(uniq.begin(), uniq.end());
}

static cv::Mat imreadPalettedAsIndex(const std::string& filename) {
  FILE* fp = fopen(filename.c_str(), "rb");
  if (!fp) {
    throw std::runtime_error("Cannot open file: " + filename);
  }
  png_byte sig[8];
  if (fread(sig, 1, 8, fp) != 8) {
    fclose(fp);
    throw std::runtime_error("Failed to read PNG signature.");
  }
  if (png_sig_cmp(sig, 0, 8)) {
    fclose(fp);
    throw std::runtime_error("File is not recognized as a PNG.");
  }
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
    png_destroy_read_struct(&png_ptr, &info_ptr, (png_infopp) nullptr);
    fclose(fp);
    throw std::runtime_error("Error during PNG read.");
  }
  png_init_io(png_ptr, fp);
  png_set_sig_bytes(png_ptr, 8);
  png_read_info(png_ptr, info_ptr);
  png_uint_32 width = png_get_image_width(png_ptr, info_ptr);
  png_uint_32 height = png_get_image_height(png_ptr, info_ptr);
  png_byte color_type = png_get_color_type(png_ptr, info_ptr);
  png_byte bit_depth = png_get_bit_depth(png_ptr, info_ptr);
  if (color_type != PNG_COLOR_TYPE_PALETTE || bit_depth != 8) {
    png_destroy_read_struct(&png_ptr, &info_ptr, (png_infopp) nullptr);
    fclose(fp);
    throw std::runtime_error("PNG is not 8-bit paletted (indexed).");
  }
  std::vector<png_bytep> row_ptrs(height);
  std::vector<png_byte> raw_data(width * height);
  for (png_uint_32 y = 0; y < height; y++) {
    row_ptrs[y] = raw_data.data() + (y * width);
  }
  png_read_image(png_ptr, row_ptrs.data());
  png_destroy_read_struct(&png_ptr, &info_ptr, (png_infopp) nullptr);
  fclose(fp);
  cv::Mat indexed((int)height, (int)width, CV_8U, raw_data.data());
  return indexed.clone();
}

} // namespace

bool ControlMasksN::load(const std::string& dirIn, int n_images) {
  std::string dir = dirIn;
  if (!dir.empty() && dir.back() != '/')
    dir += '/';

  img_col.resize(n_images);
  img_row.resize(n_images);
  positions.clear();
  positions.reserve(n_images);

  for (int i = 0; i < n_images; ++i) {
    char buf_pos[64], buf_x[64], buf_y[64];
    snprintf(buf_pos, sizeof(buf_pos), "mapping_%04d.tif", i);
    snprintf(buf_x, sizeof(buf_x), "mapping_%04d_x.tif", i);
    snprintf(buf_y, sizeof(buf_y), "mapping_%04d_y.tif", i);
    std::string mapping_pos = dir + buf_pos;
    std::string mapping_x = dir + buf_x;
    std::string mapping_y = dir + buf_y;

    img_col[i] = cv::imread(mapping_x, cv::IMREAD_ANYDEPTH);
    img_row[i] = cv::imread(mapping_y, cv::IMREAD_ANYDEPTH);
    if (img_col[i].empty() || img_row[i].empty()) {
      std::cerr << "Unable to load remap for index " << i << " from " << mapping_x << " / " << mapping_y << std::endl;
      return false;
    }
    positions.push_back(get_geo_tiffN(mapping_pos));
  }

  positions = normalize_positionsN(std::move(positions));

  std::string seam_filename = dir + "seam_file.png";
  try {
    whole_seam_mask_indexed = imreadPalettedAsIndex(seam_filename);
  } catch (const std::exception& e) {
    // Many 2-view seam masks are plain grayscale (e.g. {0,255}), not paletted PNGs.
    // Fall back to OpenCV's grayscale loader in that case.
    whole_seam_mask_indexed = cv::imread(seam_filename, cv::IMREAD_GRAYSCALE);
    if (whole_seam_mask_indexed.empty()) {
      std::cerr << "Unable to load seam mask: " << seam_filename << " (" << e.what() << ")" << std::endl;
      return false;
    }
  }

  auto uniq = get_unique_values(whole_seam_mask_indexed);
  if (uniq.empty() || static_cast<int>(uniq.size()) != n_images) {
    std::cerr << "Seam mask classes (" << uniq.size() << ") != n_images (" << n_images << "): " << seam_filename
              << std::endl;
    return false;
  }

  // Remap seam labels to contiguous indices [0..n_images-1] so downstream kernels
  // (and hard-seam dest-map logic) can assume canonical values.
  //
  // Example: a 2-view grayscale seam may be {0,255}. We remap to {0,1}.
  if (uniq.front() != 0 || uniq.back() != n_images - 1) {
    std::array<uint8_t, 256> lut{};
    for (int i = 0; i < n_images; ++i) {
      lut[static_cast<uint8_t>(uniq[i])] = static_cast<uint8_t>(i);
    }
    for (int y = 0; y < whole_seam_mask_indexed.rows; ++y) {
      uint8_t* rowPtr = whole_seam_mask_indexed.ptr<uint8_t>(y);
      for (int x = 0; x < whole_seam_mask_indexed.cols; ++x) {
        rowPtr[x] = lut[rowPtr[x]];
      }
    }
  }

  return true;
}

bool ControlMasksN::is_valid() const {
  if (img_col.empty() || img_row.empty() || img_col.size() != img_row.size())
    return false;
  if (positions.size() != img_col.size())
    return false;
  if (whole_seam_mask_indexed.empty())
    return false;
  return true;
}

size_t ControlMasksN::canvas_width() const {
  float maxw = 0;
  for (size_t i = 0; i < img_col.size(); ++i) {
    maxw = std::max(maxw, positions[i].xpos + img_col[i].cols);
  }
  return static_cast<size_t>(maxw);
}

size_t ControlMasksN::canvas_height() const {
  float maxh = 0;
  for (size_t i = 0; i < img_row.size(); ++i) {
    maxh = std::max(maxh, positions[i].ypos + img_row[i].rows);
  }
  return static_cast<size_t>(maxh);
}

cv::Mat ControlMasksN::split_to_channels(const cv::Mat& indexed, int n_images) {
  CV_Assert(indexed.type() == CV_8U);
  cv::Mat out(indexed.size(), CV_MAKETYPE(CV_8U, n_images), cv::Scalar(0));
  std::vector<cv::Mat> channels;
  cv::split(out, channels);
  for (int i = 0; i < n_images; ++i) {
    cv::Mat mask = (indexed == i);
    channels[i].setTo(1, mask);
  }
  cv::merge(channels, out);
  return out;
}

} // namespace pano
} // namespace hm
