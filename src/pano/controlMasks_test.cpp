#include "cupano/pano/controlMasks.h"
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
  masks.img1_col = remap(4, 8, 10);
  masks.img1_row = remap(4, 8, 100);
  masks.img2_col = remap(4, 8, 200);
  masks.img2_row = remap(4, 8, 300);
  masks.img1_col.at<uint16_t>(1, 1) = kUnmapped;
  masks.img1_row.at<uint16_t>(1, 1) = kUnmapped;
  masks.img2_col.at<uint16_t>(2, 2) = kUnmapped;
  masks.img2_row.at<uint16_t>(2, 2) = kUnmapped;
  masks.whole_seam_mask_image = cv::Mat(4, 12, CV_8U, cv::Scalar(0));
  masks.whole_seam_mask_image.colRange(6, 12).setTo(1);
  masks.positions = {{0.0f, 0.0f}, {8.0f, 0.0f}};

  masks.scale_to_max_output_width(8);

  EXPECT_EQ(masks.canvas_width(), 8u);
  EXPECT_EQ(masks.canvas_height(), 2u);
  EXPECT_EQ(masks.img1_col.size(), cv::Size(4, 2));
  EXPECT_EQ(masks.img2_col.size(), cv::Size(4, 2));
  EXPECT_EQ(masks.positions[1].xpos, 4.0f);
  EXPECT_EQ(cv::countNonZero(masks.img1_col == kUnmapped), 1);
  EXPECT_EQ(cv::countNonZero(masks.img1_row == kUnmapped), 1);
  EXPECT_EQ(cv::countNonZero(masks.img2_col == kUnmapped), 1);
  EXPECT_EQ(cv::countNonZero(masks.img2_row == kUnmapped), 1);
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

} // namespace
} // namespace hm::pano
