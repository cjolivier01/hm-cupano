#include "cupano/pano/canvasManager.h"
#include "cupano/pano/controlMasks.h"
#include "cupano/pano/controlMasks3.h"
#include "cupano/pano/controlMasksN.h"

#include <gtest/gtest.h>

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
