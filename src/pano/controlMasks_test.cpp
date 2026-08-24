#include "cupano/pano/canvasManager.h"
#include "cupano/pano/controlMasks.h"
#include "cupano/pano/controlMasks3.h"
#include "cupano/pano/controlMasksN.h"

#include <gtest/gtest.h>
#include <tiffio.h>
#include <filesystem>
#include <string>

namespace hm::pano {
namespace {

constexpr uint16_t kUnmapped = 65535;

cv::Mat remap(int rows, int cols, uint16_t base) {
  cv::Mat out(rows, cols, CV_16U);
  for (int y = 0; y < rows; ++y) {
    for (int x = 0; x < cols; ++x) {
      out.at<uint16_t>(y, x) = static_cast<uint16_t>(base + y * cols + x);
    }
  }
  return out;
}

bool write_tiff(const std::filesystem::path& path, uint32_t width, uint32_t height, float xpos, float ypos) {
  TIFF* tif = TIFFOpen(path.c_str(), "w");
  if (!tif) {
    return false;
  }
  TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, width);
  TIFFSetField(tif, TIFFTAG_IMAGELENGTH, height);
  TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, 1);
  TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, 8);
  TIFFSetField(tif, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
  TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
  TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
  TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP, height);
  TIFFSetField(tif, TIFFTAG_XRESOLUTION, 1.0f);
  TIFFSetField(tif, TIFFTAG_YRESOLUTION, 1.0f);
  TIFFSetField(tif, TIFFTAG_XPOSITION, xpos);
  TIFFSetField(tif, TIFFTAG_YPOSITION, ypos);
  std::vector<uint8_t> row(width, 0);
  bool ok = true;
  for (uint32_t y = 0; y < height; ++y) {
    ok = TIFFWriteScanline(tif, row.data(), y, 0) >= 0 && ok;
  }
  TIFFClose(tif);
  return ok;
}

bool write_bad_tiff(const std::filesystem::path& path, uint32_t width, uint32_t height) {
  TIFF* tif = TIFFOpen(path.c_str(), "w");
  if (!tif) {
    return false;
  }
  TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, width);
  TIFFSetField(tif, TIFFTAG_IMAGELENGTH, height);
  TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, 1);
  TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, 8);
  TIFFSetField(tif, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
  TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
  TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
  TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP, height);
  std::vector<uint8_t> row(width, 0);
  bool ok = true;
  for (uint32_t y = 0; y < height; ++y) {
    ok = TIFFWriteScanline(tif, row.data(), y, 0) >= 0 && ok;
  }
  TIFFClose(tif);
  return ok;
}

bool write_control_masks_files(const std::filesystem::path& dir, bool valid_positions) {
  std::filesystem::create_directories(dir);
  const bool positions_ok = valid_positions
      ? (write_tiff(dir / "mapping_0000.tif", 8, 4, 0.0f, 0.0f) &&
         write_tiff(dir / "mapping_0001.tif", 8, 4, 8.0f, 0.0f))
      : (write_bad_tiff(dir / "mapping_0000.tif", 8, 4) && write_bad_tiff(dir / "mapping_0001.tif", 8, 4));
  if (!positions_ok) {
    return false;
  }
  cv::imwrite((dir / "mapping_0000_x.tif").string(), remap(4, 8, 10));
  cv::imwrite((dir / "mapping_0000_y.tif").string(), remap(4, 8, 100));
  cv::imwrite((dir / "mapping_0001_x.tif").string(), remap(4, 8, 200));
  cv::imwrite((dir / "mapping_0001_y.tif").string(), remap(4, 8, 300));
  cv::Mat seam(4, 16, CV_8U, cv::Scalar(0));
  seam.colRange(8, 16).setTo(255);
  return cv::imwrite((dir / "seam_file.png").string(), seam);
}

TEST(ControlMasksTest, ScaleToMaxOutputWidthPreservesUnmappedSentinel) {
  ControlMasks masks;
  masks.img1_col = remap(8, 8, 10);
  masks.img1_row = remap(8, 8, 100);
  masks.img2_col = remap(8, 8, 200);
  masks.img2_row = remap(8, 8, 300);
  masks.img1_col.at<uint16_t>(7, 7) = kUnmapped;
  masks.img1_row.at<uint16_t>(7, 7) = kUnmapped;
  masks.img2_col.at<uint16_t>(7, 7) = kUnmapped;
  masks.img2_row.at<uint16_t>(7, 7) = kUnmapped;
  masks.whole_seam_mask_image = cv::Mat(8, 16, CV_8U, cv::Scalar(0));
  masks.whole_seam_mask_image.colRange(8, 16).setTo(1);
  masks.positions = {{0.0f, 0.0f}, {8.0f, 0.0f}};

  masks.scale_to_max_output_width(2);

  EXPECT_EQ(masks.canvas_width(), 2u);
  EXPECT_EQ(masks.canvas_height(), 1u);
  EXPECT_EQ(masks.img1_col.size(), cv::Size(1, 1));
  EXPECT_EQ(masks.img2_col.size(), cv::Size(1, 1));
  EXPECT_EQ(masks.positions[1].xpos, 1.0f);
  EXPECT_EQ(cv::countNonZero(masks.img1_col == kUnmapped), 1);
  EXPECT_EQ(cv::countNonZero(masks.img1_row == kUnmapped), 1);
  EXPECT_EQ(cv::countNonZero(masks.img2_col == kUnmapped), 1);
  EXPECT_EQ(cv::countNonZero(masks.img2_row == kUnmapped), 1);
}

TEST(ControlMasksTest, ScaleToMaxOutputWidthRejectsCollapsedSeamClass) {
  ControlMasks masks;
  masks.img1_col = remap(4, 40, 10);
  masks.img1_row = remap(4, 40, 100);
  masks.img2_col = remap(4, 40, 200);
  masks.img2_row = remap(4, 40, 300);
  masks.whole_seam_mask_image = cv::Mat(4, 80, CV_8U, cv::Scalar(0));
  masks.whole_seam_mask_image.colRange(40, 80).setTo(1);
  masks.positions = {{0.0f, 0.0f}, {40.0f, 0.0f}};

  EXPECT_FALSE(masks.scale_to_max_output_width(1));
  EXPECT_FALSE(masks.is_valid());
}

TEST(ControlMasksTest, FailedLoadClearsPreviousState) {
  const std::filesystem::path root =
      std::filesystem::temp_directory_path() / ("cupano-control-masks-test-" + std::to_string(::getpid()));
  const std::filesystem::path valid = root / "valid";
  const std::filesystem::path invalid = root / "invalid";
  std::filesystem::remove_all(root);
  ASSERT_TRUE(write_control_masks_files(valid, true));
  ASSERT_TRUE(write_control_masks_files(invalid, false));

  ControlMasks masks;
  EXPECT_TRUE(masks.load(valid.string()));
  EXPECT_TRUE(masks.is_valid());
  EXPECT_FALSE(masks.load(invalid.string()));
  EXPECT_FALSE(masks.is_valid());

  std::filesystem::remove_all(root);
}

TEST(ControlMasksTest, ScaleToMaxOutputWidthPreservesSparseUnmappedSentinel) {
  ControlMasks masks;
  masks.img1_col = remap(32, 32, 10);
  masks.img1_row = remap(32, 32, 100);
  masks.img2_col = remap(32, 32, 200);
  masks.img2_row = remap(32, 32, 300);
  masks.img1_col.at<uint16_t>(0, 0) = kUnmapped;
  masks.img1_row.at<uint16_t>(0, 0) = kUnmapped;
  masks.img2_col.at<uint16_t>(31, 31) = kUnmapped;
  masks.img2_row.at<uint16_t>(31, 31) = kUnmapped;
  masks.whole_seam_mask_image = cv::Mat(32, 64, CV_8U, cv::Scalar(0));
  masks.whole_seam_mask_image.colRange(32, 64).setTo(1);
  masks.positions = {{0.0f, 0.0f}, {32.0f, 0.0f}};

  masks.scale_to_max_output_width(2);

  EXPECT_EQ(masks.canvas_width(), 2u);
  EXPECT_EQ(masks.canvas_height(), 1u);
  EXPECT_EQ(cv::countNonZero(masks.img1_col == kUnmapped), 1);
  EXPECT_EQ(cv::countNonZero(masks.img1_row == kUnmapped), 1);
  EXPECT_EQ(cv::countNonZero(masks.img2_col == kUnmapped), 1);
  EXPECT_EQ(cv::countNonZero(masks.img2_row == kUnmapped), 1);
}

TEST(ControlMasksTest, ScaleToMaxOutputWidthKeepsSeamAlignedWithRoundedCanvas) {
  ControlMasks masks;
  masks.img1_col = remap(7, 11, 10);
  masks.img1_row = remap(7, 11, 100);
  masks.img2_col = remap(7, 11, 200);
  masks.img2_row = remap(7, 11, 300);
  masks.whole_seam_mask_image = cv::Mat(7, 17, CV_8U, cv::Scalar(0));
  masks.whole_seam_mask_image.colRange(8, 17).setTo(1);
  masks.positions = {{0.0f, 0.0f}, {6.0f, 0.0f}};

  masks.scale_to_max_output_width(10);

  EXPECT_EQ(masks.canvas_width(), 10u);
  EXPECT_EQ(masks.whole_seam_mask_image.size(), cv::Size(10, static_cast<int>(masks.canvas_height())));
  EXPECT_EQ(masks.positions[1].xpos, 3.0f);
  EXPECT_EQ(masks.img1_col.cols, 7);
  EXPECT_EQ(masks.img2_col.cols, 7);
}

TEST(ControlMasksNTest, ScaleToMaxOutputWidthPreservesIndexedSeam) {
  ControlMasksN masks;
  masks.img_col = {remap(4, 8, 10), remap(4, 8, 100), remap(4, 8, 200)};
  masks.img_row = {remap(4, 8, 300), remap(4, 8, 400), remap(4, 8, 500)};
  masks.img_col[2].at<uint16_t>(3, 3) = kUnmapped;
  masks.img_row[2].at<uint16_t>(3, 3) = kUnmapped;
  masks.whole_seam_mask_indexed = cv::Mat(4, 20, CV_8U, cv::Scalar(0));
  masks.whole_seam_mask_indexed.colRange(8, 16).setTo(1);
  masks.whole_seam_mask_indexed.colRange(16, 20).setTo(2);
  masks.positions = {{0.0f, 0.0f}, {6.0f, 0.0f}, {12.0f, 0.0f}};

  masks.scale_to_max_output_width(10);

  EXPECT_EQ(masks.canvas_width(), 10u);
  EXPECT_EQ(masks.canvas_height(), 2u);
  EXPECT_EQ(masks.positions[1].xpos, 3.0f);
  EXPECT_EQ(masks.positions[2].xpos, 6.0f);
  EXPECT_EQ(cv::countNonZero(masks.img_col[2] == kUnmapped), 1);
  EXPECT_EQ(cv::countNonZero(masks.img_row[2] == kUnmapped), 1);
  EXPECT_EQ(cv::countNonZero(masks.whole_seam_mask_indexed == 2), 4);
}

TEST(ControlMasksNTest, ScaleToMaxOutputWidthKeepsIndexedSeamAlignedWithRoundedCanvas) {
  ControlMasksN masks;
  masks.img_col = {remap(7, 11, 10), remap(7, 11, 100), remap(7, 11, 200)};
  masks.img_row = {remap(7, 11, 300), remap(7, 11, 400), remap(7, 11, 500)};
  masks.whole_seam_mask_indexed = cv::Mat(7, 23, CV_8U, cv::Scalar(0));
  masks.whole_seam_mask_indexed.colRange(8, 16).setTo(1);
  masks.whole_seam_mask_indexed.colRange(16, 23).setTo(2);
  masks.positions = {{0.0f, 0.0f}, {6.0f, 0.0f}, {12.0f, 0.0f}};

  masks.scale_to_max_output_width(13);

  EXPECT_EQ(masks.canvas_width(), 13u);
  EXPECT_EQ(masks.whole_seam_mask_indexed.size(), cv::Size(13, static_cast<int>(masks.canvas_height())));
  EXPECT_EQ(masks.positions[1].xpos, 3.0f);
  EXPECT_EQ(masks.positions[2].xpos, 6.0f);
}

TEST(ControlMasksNTest, ScaleToMaxOutputWidthRejectsCollapsedSeamClass) {
  ControlMasksN masks;
  masks.img_col = {remap(4, 40, 10), remap(4, 40, 100), remap(4, 40, 200)};
  masks.img_row = {remap(4, 40, 300), remap(4, 40, 400), remap(4, 40, 500)};
  masks.whole_seam_mask_indexed = cv::Mat(4, 120, CV_8U, cv::Scalar(0));
  masks.whole_seam_mask_indexed.colRange(40, 80).setTo(1);
  masks.whole_seam_mask_indexed.colRange(80, 120).setTo(2);
  masks.positions = {{0.0f, 0.0f}, {40.0f, 0.0f}, {80.0f, 0.0f}};

  EXPECT_FALSE(masks.scale_to_max_output_width(2));
  EXPECT_FALSE(masks.is_valid());
}

TEST(ControlMasks3Test, ScaleToMaxOutputWidthRejectsCollapsedSeamClass) {
  ControlMasks3 masks;
  masks.img0_col = remap(4, 40, 10);
  masks.img0_row = remap(4, 40, 300);
  masks.img1_col = remap(4, 40, 100);
  masks.img1_row = remap(4, 40, 400);
  masks.img2_col = remap(4, 40, 200);
  masks.img2_row = remap(4, 40, 500);
  masks.whole_seam_mask_image = cv::Mat(4, 120, CV_8U, cv::Scalar(0));
  masks.whole_seam_mask_image.colRange(40, 80).setTo(1);
  masks.whole_seam_mask_image.colRange(80, 120).setTo(2);
  masks.positions = {{0.0f, 0.0f}, {40.0f, 0.0f}, {80.0f, 0.0f}};

  EXPECT_FALSE(masks.scale_to_max_output_width(2));
  EXPECT_FALSE(masks.is_valid());
}

TEST(CanvasManagerTest, MinimizeBlendClampsPaddingToScaledCanvas) {
  CanvasInfo canvas_info;
  canvas_info.width = 10;
  canvas_info.height = 4;
  canvas_info.positions = {{0, 0}, {3, 0}};
  CanvasManager manager(canvas_info, true);
  manager._remapper_1.width = 8;
  manager._remapper_1.height = 4;

  manager.updateMinimizeBlend(cv::Size(8, 4), cv::Size(7, 4));

  EXPECT_EQ(manager.overlap_padding(), 2);
  EXPECT_EQ(manager.remapped_image_roi_blend_1.x, 1);
  EXPECT_EQ(manager.remapped_image_roi_blend_1.width, 7);
  EXPECT_EQ(manager.remapped_image_roi_blend_2.width, 7);
}

} // namespace
} // namespace hm::pano
