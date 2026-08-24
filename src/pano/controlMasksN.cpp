#include "controlMasksN.h"

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <png.h>
#include <tiffio.h>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <optional>
#include <set>
#include <stdexcept>

namespace hm {
namespace pano {
namespace {

constexpr uint16_t kUnmappedPositionValue = 65535;

struct TiffInfoN {
  float xResolution = 0.0f;
  float yResolution = 0.0f;
  float xPosition = 0.0f;
  float yPosition = 0.0f;
};

static std::optional<SpatialTiff> get_geo_tiffN(const std::string& filename) {
  TiffInfoN info;
  TIFF* tif = TIFFOpen(filename.c_str(), "r");
  if (!tif) {
    std::cerr << "Error: Could not open file " << filename << std::endl;
    return std::nullopt;
  }
  float xres = 0.0f, yres = 0.0f;
  const bool has_resolution =
      TIFFGetField(tif, TIFFTAG_XRESOLUTION, &xres) && TIFFGetField(tif, TIFFTAG_YRESOLUTION, &yres);
  info.xResolution = xres;
  info.yResolution = yres;
  float xpos = 0.0f, ypos = 0.0f;
  const bool has_position = TIFFGetField(tif, TIFFTAG_XPOSITION, &xpos) && TIFFGetField(tif, TIFFTAG_YPOSITION, &ypos);
  info.xPosition = xpos;
  info.yPosition = ypos;
  TIFFClose(tif);
  if (!has_resolution || !has_position) {
    return std::nullopt;
  }
  return SpatialTiff{.xpos = info.xPosition * info.xResolution, .ypos = info.yPosition * info.yResolution};
}

static std::optional<cv::Size> read_tiff_size(const std::string& filename) {
  TIFF* tif = TIFFOpen(filename.c_str(), "r");
  if (!tif) {
    return std::nullopt;
  }
  uint32_t width = 0;
  uint32_t height = 0;
  const bool ok = TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width) && TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &height);
  TIFFClose(tif);
  if (!ok || width == 0 || height == 0) {
    return std::nullopt;
  }
  return cv::Size(static_cast<int>(width), static_cast<int>(height));
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

static bool indexed_seam_has_all_classes(const cv::Mat& indexed, int n_images) {
  const auto uniq = get_unique_values(indexed);
  return !uniq.empty() && static_cast<int>(uniq.size()) == n_images && uniq.front() == 0 && uniq.back() == n_images - 1;
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

cv::Mat resize_nearest(const cv::Mat& src, const cv::Size& size) {
  cv::Mat resized;
  cv::resize(src, resized, size, 0.0, 0.0, cv::INTER_NEAREST);
  return resized;
}

struct ScaledPlacementN {
  SpatialTiff position;
  cv::Size size;
};

ScaledPlacementN scaled_placementN(const SpatialTiff& position, const cv::Size& size, double scale) {
  const auto scaled_x = static_cast<int>(std::floor(position.xpos * scale));
  const auto scaled_y = static_cast<int>(std::floor(position.ypos * scale));
  const auto scaled_right = static_cast<int>(std::ceil((position.xpos + size.width) * scale));
  const auto scaled_bottom = static_cast<int>(std::ceil((position.ypos + size.height) * scale));
  return ScaledPlacementN{
      .position = SpatialTiff{.xpos = static_cast<float>(scaled_x), .ypos = static_cast<float>(scaled_y)},
      .size = cv::Size(std::max(1, scaled_right - scaled_x), std::max(1, scaled_bottom - scaled_y))};
}

cv::Size canvas_sizeN(const std::vector<ScaledPlacementN>& placements) {
  int width = 1;
  int height = 1;
  for (const ScaledPlacementN& placement : placements) {
    width = std::max(width, static_cast<int>(placement.position.xpos) + placement.size.width);
    height = std::max(height, static_cast<int>(placement.position.ypos) + placement.size.height);
  }
  return cv::Size(width, height);
}

void clear_control_masksN(ControlMasksN& masks) {
  masks.img_col.clear();
  masks.img_row.clear();
  masks.positions.clear();
  masks.whole_seam_mask_indexed.release();
}

double scale_to_fit_max_widthN(
    const std::vector<SpatialTiff>& positions,
    const std::vector<cv::Size>& sizes,
    size_t native_width,
    int max_output_width) {
  double low = 0.0;
  double high = static_cast<double>(max_output_width) / static_cast<double>(native_width);
  std::vector<ScaledPlacementN> direct_placements;
  direct_placements.reserve(positions.size());
  for (size_t i = 0; i < positions.size(); ++i) {
    direct_placements.push_back(scaled_placementN(positions[i], sizes[i], high));
  }
  if (canvas_sizeN(direct_placements).width <= max_output_width) {
    return high;
  }
  for (int iteration = 0; iteration < 32; ++iteration) {
    const double mid = (low + high) / 2.0;
    std::vector<ScaledPlacementN> placements;
    placements.reserve(positions.size());
    for (size_t i = 0; i < positions.size(); ++i) {
      placements.push_back(scaled_placementN(positions[i], sizes[i], mid));
    }
    if (canvas_sizeN(placements).width <= max_output_width) {
      low = mid;
    } else {
      high = mid;
    }
  }
  return low > 0.0 ? low : high;
}

} // namespace

bool ControlMasksN::load(const std::string& dirIn, int n_images, int max_output_width) {
  clear_control_masksN(*this);
  std::string dir = dirIn;
  if (!dir.empty() && dir.back() != '/')
    dir += '/';

  img_col.resize(n_images);
  img_row.resize(n_images);
  positions.clear();
  positions.reserve(n_images);
  std::vector<cv::Size> native_sizes;
  native_sizes.reserve(n_images);
  std::vector<std::string> mapping_x_paths;
  std::vector<std::string> mapping_y_paths;
  mapping_x_paths.reserve(n_images);
  mapping_y_paths.reserve(n_images);

  for (int i = 0; i < n_images; ++i) {
    char buf_pos[64], buf_x[64], buf_y[64];
    snprintf(buf_pos, sizeof(buf_pos), "mapping_%04d.tif", i);
    snprintf(buf_x, sizeof(buf_x), "mapping_%04d_x.tif", i);
    snprintf(buf_y, sizeof(buf_y), "mapping_%04d_y.tif", i);
    std::string mapping_pos = dir + buf_pos;
    std::string mapping_x = dir + buf_x;
    std::string mapping_y = dir + buf_y;

    const auto size = read_tiff_size(mapping_x);
    if (!size) {
      std::cerr << "Unable to load remap metadata for index " << i << " from " << mapping_x << std::endl;
      clear_control_masksN(*this);
      return false;
    }
    native_sizes.push_back(*size);
    mapping_x_paths.push_back(mapping_x);
    mapping_y_paths.push_back(mapping_y);
    const auto position = get_geo_tiffN(mapping_pos);
    if (!position) {
      std::cerr << "Unable to load mapping placement metadata for index " << i << " from " << mapping_pos << std::endl;
      clear_control_masksN(*this);
      return false;
    }
    positions.push_back(*position);
  }

  positions = normalize_positionsN(std::move(positions));
  std::vector<ScaledPlacementN> placements;
  placements.reserve(native_sizes.size());
  for (size_t i = 0; i < native_sizes.size(); ++i) {
    placements.push_back(ScaledPlacementN{.position = positions[i], .size = native_sizes[i]});
  }
  const cv::Size native_canvas_size = canvas_sizeN(placements);
  if (max_output_width > 0 && native_canvas_size.width > max_output_width) {
    std::cerr << "Control mask canvas " << native_canvas_size.width << "x" << native_canvas_size.height
              << " exceeds max_output_width " << max_output_width << "; regenerate capped mapping TIFFs" << std::endl;
    clear_control_masksN(*this);
    return false;
  }

  for (int i = 0; i < n_images; ++i) {
    img_col[i] = cv::imread(mapping_x_paths[i], cv::IMREAD_ANYDEPTH);
    if (img_col[i].empty()) {
      std::cerr << "Unable to load remap for index " << i << " from " << mapping_x_paths[i] << std::endl;
      clear_control_masksN(*this);
      return false;
    }
    if (img_col[i].size() != placements[i].size) {
      img_col[i] = resize_remap_preserving_unmapped(img_col[i], placements[i].size);
    }
    img_row[i] = cv::imread(mapping_y_paths[i], cv::IMREAD_ANYDEPTH);
    if (img_row[i].empty()) {
      std::cerr << "Unable to load remap for index " << i << " from " << mapping_y_paths[i] << std::endl;
      clear_control_masksN(*this);
      return false;
    }
    if (img_row[i].size() != placements[i].size) {
      img_row[i] = resize_remap_preserving_unmapped(img_row[i], placements[i].size);
    }
    positions[i] = placements[i].position;
  }

  std::string seam_filename = dir + "seam_file.png";
  try {
    whole_seam_mask_indexed = imreadPalettedAsIndex(seam_filename);
  } catch (const std::exception& e) {
    // Many 2-view seam masks are plain grayscale (e.g. {0,255}), not paletted PNGs.
    // Fall back to OpenCV's grayscale loader in that case.
    whole_seam_mask_indexed = cv::imread(seam_filename, cv::IMREAD_GRAYSCALE);
    if (whole_seam_mask_indexed.empty()) {
      std::cerr << "Unable to load seam mask: " << seam_filename << " (" << e.what() << ")" << std::endl;
      clear_control_masksN(*this);
      return false;
    }
  }
  const cv::Size effective_canvas_size = canvas_sizeN(placements);
  if (whole_seam_mask_indexed.size() != effective_canvas_size) {
    whole_seam_mask_indexed = resize_nearest(whole_seam_mask_indexed, effective_canvas_size);
  }

  auto uniq = get_unique_values(whole_seam_mask_indexed);
  if (uniq.empty() || static_cast<int>(uniq.size()) != n_images) {
    std::cerr << "Seam mask classes (" << uniq.size() << ") != n_images (" << n_images << "): " << seam_filename
              << std::endl;
    clear_control_masksN(*this);
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

bool ControlMasksN::scale_to_max_output_width(int max_output_width) {
  if (!is_valid() || max_output_width <= 0 || canvas_width() <= static_cast<size_t>(max_output_width)) {
    return is_valid();
  }
  const size_t native_width = canvas_width();
  std::vector<cv::Size> native_sizes;
  native_sizes.reserve(img_col.size());
  for (const auto& remap : img_col) {
    native_sizes.push_back(remap.size());
  }
  const double scale = scale_to_fit_max_widthN(positions, native_sizes, native_width, max_output_width);
  std::vector<ScaledPlacementN> placements;
  placements.reserve(img_col.size());
  for (size_t i = 0; i < img_col.size(); ++i) {
    placements.push_back(scaled_placementN(positions[i], img_col[i].size(), scale));
  }
  for (size_t i = 0; i < img_col.size(); ++i) {
    img_col[i] = resize_remap_preserving_unmapped(img_col[i], placements[i].size);
    img_row[i] = resize_remap_preserving_unmapped(img_row[i], img_col[i].size());
    positions[i] = placements[i].position;
  }
  whole_seam_mask_indexed = resize_nearest(whole_seam_mask_indexed, canvas_sizeN(placements));
  if (!indexed_seam_has_all_classes(whole_seam_mask_indexed, static_cast<int>(img_col.size()))) {
    img_col.clear();
    img_row.clear();
    positions.clear();
    whole_seam_mask_indexed.release();
    return false;
  }
  return true;
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
